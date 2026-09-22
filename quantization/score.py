#!/usr/bin/env python3
"""Layer-A scorer: per-capability divergence of a candidate checkpoint vs a bf16 reference.

Two modes:
  reference  bf16 engine: generate + freeze continuations for `generate` items,
             tokenize `score` items, cache top-K logprobs over the frozen token ids
  compare    candidate engine: teacher-force the SAME token ids, diff against the cache

Everything is done on token ids (via the engine's /tokenize), never on characters, so
reference and candidate always score identical positions.

Gate (TODO 2.6): `compare` against the reference engine itself must give KL ~ 0 and
top-1 100%. Score on a dedicated engine with MTP off (never on one that serves users).
"""
import argparse, json, math, os, sys, threading, urllib.request, urllib.error
from collections import defaultdict

VOCAB = 248320
MAX_CHUNK = 1024          # positions x VOCAB x 4B ; ~0.95 GiB of full-vocab logits per
                          # window. The bf16 TP=2 engine has ~3 GiB of headroom per card.


def post(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}") from None


def tokenize(base, model, it):
    """Token ids of the context (chat-templated prompt) or of a raw `score` text."""
    if it.get("mode") == "generate":
        body = {"model": model, "messages": [{"role": "user", "content": it["prompt"]}],
                "add_generation_prompt": True}
        for k in ("tools", "chat_template_kwargs"):
            if it.get(k):
                body[k] = it[k]
    else:
        body = {"model": model, "prompt": it["text"]}
    return post(f"{base}/tokenize", body)["tokens"]


def generate(base, model, ctx_ids, max_tokens):
    """Greedy continuation of ctx_ids, as token ids. Frozen into the reference."""
    r = post(f"{base}/v1/completions", {
        "model": model, "prompt": ctx_ids, "max_tokens": max_tokens, "temperature": 0,
        "return_token_ids": True, "skip_special_tokens": False})
    ch = r["choices"][0]
    ids = ch.get("token_ids")
    if ids is None:       # older server: re-tokenize the text (special tokens kept above)
        ids = post(f"{base}/tokenize", {"model": model, "prompt": ch["text"],
                                        "add_special_tokens": False})["tokens"]
    return ids, ch.get("finish_reason")


def parse_pos(d, tok_id, topk):
    """One prompt_logprobs entry -> {id, lp, rank, top1, top:{id:lp}}.

    The observed token is looked up BY ID. vLLM returns top-k plus the observed token,
    so when the observed token is inside the top-k the dict has exactly k entries and
    nothing about ordering or rank identifies it.
    """
    act = d.get(str(tok_id))
    if act is None:
        raise RuntimeError(f"observed token {tok_id} missing from prompt_logprobs entry")
    top = {tid: v["logprob"] for tid, v in d.items() if v["rank"] and v["rank"] <= topk}
    top1 = next((tid for tid, v in d.items() if v["rank"] == 1), None)
    return {"id": tok_id, "lp": act["logprob"], "rank": act["rank"], "top1": top1, "top": top}


# One prompt_logprobs window in flight at a time: each materialises chunk x VOCAB x 4 B of
# logits outside the engine's memory budget. Four concurrent 1024-token windows OOM-killed
# the bf16 TP=2 engine on 2026-09-21. Generation stays concurrent.
_SCORE_LOCK = threading.Lock()


def score_ids(base, model, ids, n_ctx, topk, chunk=MAX_CHUNK):
    """Teacher-forced scores for ids[n_ctx:]. Sliding token windows, stride chunk/2:
    every position past the first window is scored with >= chunk/2 tokens of context."""
    out = {}
    start = done = 0
    stride = max(chunk // 2, 1)
    while done < len(ids):
        win = ids[start:start + chunk]
        with _SCORE_LOCK:
            r = post(f"{base}/v1/completions", {
                "model": model, "prompt": win, "max_tokens": 1,
                "temperature": 0, "prompt_logprobs": topk})
        pl = r["choices"][0].get("prompt_logprobs") or []
        if len(pl) != len(win):
            raise RuntimeError(f"prompt_logprobs has {len(pl)} entries for {len(win)} tokens")
        for j, d in enumerate(pl):
            g = start + j
            if g < done or g < n_ctx or not d:     # d is None for the very first token
                continue
            out[g] = parse_pos(d, ids[g], topk)
        done = start + len(win)
        start += stride
    return [out.get(g) for g in range(n_ctx, len(ids))]


def kl_trunc(p_top, q_top):
    """KL over the partition {reference top-K tokens, everything else}.

    Coarse-graining makes this a lower bound on the full-vocab KL. A reference token
    absent from the candidate's top-K is floored at the candidate's smallest reported
    logprob (its true value is no higher), and counted so the caller can report how
    often the floor was hit. Returns (kl, n_floored)."""
    floor = min(q_top.values())
    kl = p_sum = q_sum = 0.0
    floored = 0
    for tid, plp in p_top.items():
        qlp = q_top.get(tid)
        if qlp is None:
            qlp = floor; floored += 1
        p = math.exp(plp)
        kl += p * (plp - qlp)
        p_sum += p; q_sum += math.exp(qlp)
    p_tail, q_tail = max(1.0 - p_sum, 0.0), max(1.0 - q_sum, 1e-12)
    if p_tail > 1e-12:
        kl += p_tail * math.log(p_tail / q_tail)
    return max(kl, 0.0), floored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["reference", "compare"])
    ap.add_argument("--base", default="http://localhost:9010")
    ap.add_argument("--model", default="score")
    ap.add_argument("--set", default="prompts/frozen-v1.jsonl")
    ap.add_argument("--ref", default="results/reference-bf16.jsonl")
    ap.add_argument("--out", default="results/candidate.json")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=MAX_CHUNK)
    ap.add_argument("--gen-max", type=int, default=1024, help="max generated tokens per item")
    ap.add_argument("--limit", type=int, default=0, help="first N items only (smoke test)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent items (reference generation)")
    a = ap.parse_args()

    if a.mode == "reference":
        if not os.path.exists(a.set):
            sys.exit(f"frozen set missing: {a.set} (see prompts/SCHEMA.md)")
        items = [json.loads(l) for l in open(a.set) if l.strip() and not l.startswith("#")]
        if a.limit:
            items = items[:a.limit]
        os.makedirs(os.path.dirname(a.ref) or ".", exist_ok=True)
        def one(it):
            ids = tokenize(a.base, a.model, it)
            n_ctx, fr = 0, None
            if it.get("mode") == "generate":
                n_ctx = len(ids)
                cont, fr = generate(a.base, a.model, ids, a.gen_max)
                ids = ids + cont
            pos = score_ids(a.base, a.model, ids, n_ctx, a.topk, a.chunk)
            return {"id": it["id"], "tag": it["tag"], "ids": ids, "n_ctx": n_ctx, "chunk": a.chunk,
                    "finish_reason": fr, "pos": pos}
        # resume: items already in the reference file are kept, never regenerated
        done = set()
        if os.path.exists(a.ref):
            for l in open(a.ref):
                try:
                    done.add(json.loads(l)["id"])
                except Exception:
                    break
        items = [it for it in items if it["id"] not in done]
        print(f"reference: {len(done)} items already done, {len(items)} to go", flush=True)
        from concurrent.futures import ThreadPoolExecutor
        with open(a.ref, "a") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
            for rec in ex.map(one, items):          # map keeps the frozen order
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"ref {rec['id']:>16s} ctx {rec['n_ctx']:5d} scored {len(rec['pos']):5d}"
                      f"{'  TRUNCATED at --gen-max' if rec['finish_reason'] == 'length' else ''}", flush=True)
        print(f"wrote {a.ref}")
        return

    # compare: the reference file is the source of truth for WHAT is scored
    refs = [json.loads(l) for l in open(a.ref) if l.strip()]
    per_item = {}
    if a.limit:
        refs = refs[:a.limit]
    agg = defaultdict(lambda: {"kl": 0.0, "n": 0, "top1": 0, "rank_sum": 0, "dlp": 0.0,
                               "floored": 0, "support": 0})
    for r in refs:
        # score with the SAME windows as the reference: context per position must match
        cand = score_ids(a.base, a.model, r["ids"], r["n_ctx"], a.topk, r.get("chunk", a.chunk))
        if len(cand) != len(r["pos"]):
            sys.exit(f"{r['id']}: {len(cand)} candidate positions vs {len(r['pos'])} reference")
        g = agg[r["tag"]]
        it = {"kl": 0.0, "n": 0, "top1": 0}
        for rp, cp in zip(r["pos"], cand):
            if not rp or not cp:
                continue
            kl, fl = kl_trunc(rp["top"], cp["top"])
            g["kl"] += kl
            it["kl"] += kl; it["n"] += 1; it["top1"] += rp["top1"] == cp["top1"]
            g["floored"] += fl
            g["support"] += len(rp["top"])
            g["n"] += 1
            g["rank_sum"] += cp["rank"]
            g["dlp"] += cp["lp"] - rp["lp"]
            if rp["top1"] == cp["top1"]:
                g["top1"] += 1
        per_item[r["id"]] = {"tag": r["tag"], **it}
        print(f"cmp {r['id']:>16s}", flush=True)

    res = {t: {"mean_kl": v["kl"] / max(v["n"], 1),
               "top1_agree": v["top1"] / max(v["n"], 1),
               "mean_rank": v["rank_sum"] / max(v["n"], 1),
               "mean_dlogprob": v["dlp"] / max(v["n"], 1),
               "kl_floored_frac": v["floored"] / max(v["support"], 1),
               "positions": v["n"]} for t, v in agg.items()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    json.dump(per_item, open(a.out.replace(".json", ".items.json"), "w"))   # for paired tests
    print(f"\n{'tag':14s} {'mean KL':>9s} {'top-1':>8s} {'d-logp':>8s} {'floored':>8s} {'positions':>10s}")
    for t, v in sorted(res.items()):
        print(f"{t:14s} {v['mean_kl']:9.5f} {v['top1_agree']*100:7.2f}% {v['mean_dlogprob']:8.4f} "
              f"{v['kl_floored_frac']*100:7.2f}% {v['positions']:10d}")


if __name__ == "__main__":
    main()
