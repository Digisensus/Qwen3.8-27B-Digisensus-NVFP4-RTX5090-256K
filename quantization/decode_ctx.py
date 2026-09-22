#!/usr/bin/env python3
"""Decode speed at long context on REAL text. Random-token prompts make MTP acceptance (and so
decode tok/s) erratic; here the context is a long document of real public conversations
(HuggingFaceH4/ultrachat_200k) followed by a question, answered with thinking off.
Per context length: single stream, and `--concurrency` streams at once (different documents),
decode tok/s = (tokens - 1) / (t_last - t_first), plus MTP acceptance from /metrics.
Usage: python3 decode_ctx.py --port 9003 --parquet <ultrachat test_sft parquet>
"""
import argparse, json, re, statistics as st, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


def metric(port, name):
    t = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=10).read().decode()
    return sum(float(x) for x in re.findall(rf"^{re.escape(name)}\{{[^}}]*\}} ([0-9.e+]+)$", t, re.M))


def ntok(port, text):
    q = urllib.request.Request(f"http://localhost:{port}/tokenize", json.dumps({"model": "qwen", "prompt": text}).encode(), {"Content-Type": "application/json"})
    return len(json.load(urllib.request.urlopen(q, timeout=120))["tokens"])


def build_doc(convs, start, target, port):
    parts, i = [], start
    est = 0
    while est < target:
        c = convs[i % len(convs)]; i += 1
        parts.append("\n".join(f"{m['role'].upper()}: {m['content']}" for m in c))
        est += len(parts[-1]) // 4
    doc = "\n\n---\n\n".join(parts)
    while ntok(port, doc) > target:          # trim to the token target
        doc = doc[: int(len(doc) * 0.97)]
    return doc


def one(port, doc, max_tokens):
    body = {"model": "qwen", "messages": [{"role": "user", "content": doc + "\n\n---\n\nSummarise the main topics of the conversations above as a numbered list, one sentence each."}],
            "max_tokens": max_tokens, "min_tokens": max_tokens, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    q = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; toks = 0
    for line in urllib.request.urlopen(q, timeout=1800):
        line = line.decode().strip()
        if not line.startswith("data: {"):
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            toks = d["usage"]["completion_tokens"]
        dl = (d.get("choices") or [{}])[0].get("delta") or {}
        if dl.get("content"):
            now = time.time(); first = first or now; last = now
    if first is None:
        return None            # request failed / produced nothing: counted, not measured
    return {"ttft": first - t0, "tps": (toks - 1) / max(last - first, 1e-6), "toks": toks, "first": first, "last": last}


def union_len(iv):
    """Total time during which at least one stream was receiving tokens."""
    tot, end = 0.0, None
    for a, b in sorted(iv):
        if end is None or a > end:
            tot += b - a; end = b
        elif b > end:
            tot += b - end; end = b
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="9003"); ap.add_argument("--parquet", required=True)
    ap.add_argument("--contexts", default="1024,8192,16384,32768,65536,131072,258000")
    ap.add_argument("--max-tokens", type=int, default=384); ap.add_argument("--pool", type=int, default=266197)
    ap.add_argument("--only-conc", action="store_true", help="skip the single-stream rows")
    ap.add_argument("--only-single", action="store_true"); ap.add_argument("--reps", type=int, default=1); ap.add_argument("--max-conc", type=int, default=8)
    ap.add_argument("--pool-from-engine", action="store_true", help="read the KV pool size from the engine log line via --pool")
    a = ap.parse_args()
    convs = [list(m) for m in pd.read_parquet(a.parquet)["messages"]]
    print("| context | decode, 1 user (tok/s) | MTP accept | concurrency | decode per user (tok/s) | total decode, all users (tok/s) | aggregate incl. prefill (tok/s) | MTP accept |")
    print("|---|---|---|---|---|---|---|---|")
    for L in map(int, a.contexts.split(",")):
        C = max(1, min(a.max_conc, a.pool // (L + a.max_tokens + 64)))
        docs = [build_doc(convs, 400 * k, L, a.port) for k in range(C + 1)]
        row = []
        for label, ds in (("single", [] if a.only_conc else docs[:1] * a.reps), ("conc", [] if a.only_single else (docs[1:C + 1] if C > 1 else []))):
            if not ds:
                row += ["1", "—", "—", "—", "—"] if label == "conc" else ["—", "—"]; continue
            d0, a0 = metric(a.port, "vllm:spec_decode_num_draft_tokens_total"), metric(a.port, "vllm:spec_decode_num_accepted_tokens_total")
            TP0 = (metric(a.port, "vllm:inter_token_latency_seconds_sum"), metric(a.port, "vllm:generation_tokens_total"), metric(a.port, "vllm:inter_token_latency_seconds_count"))
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=1 if label == "single" else len(ds)) as ex:
                rs = list(ex.map(lambda d: one(a.port, d, a.max_tokens), ds))
            wall = time.time() - t0
            failed = sum(r is None for r in rs); rs = [r for r in rs if r]
            if failed: print(f"  ({failed} of {len(ds)} requests failed at {L})", flush=True)
            if not rs:
                row += ["—", "—"] if label == "single" else [str(len(ds)), "failed", "—", "—", "—"]; continue
            d1, a1 = metric(a.port, "vllm:spec_decode_num_draft_tokens_total"), metric(a.port, "vllm:spec_decode_num_accepted_tokens_total")
            acc = f"{100 * (a1 - a0) / max(d1 - d0, 1):.0f}%"
            if label == "single":
                eng = (metric(a.port, "vllm:inter_token_latency_seconds_sum"), metric(a.port, "vllm:generation_tokens_total"), metric(a.port, "vllm:inter_token_latency_seconds_count"))
                steps = eng[2] - TP0[2]; secs = eng[0] - TP0[0]; toks = eng[1] - TP0[1]
                row += ["/".join(f"{r['tps']:.0f}" for r in rs) + f" (engine: {1000*secs/max(steps,1):.1f} ms/step, {toks/max(steps,1):.2f} tok/step = {toks/max(secs,1e-6):.0f} tok/s)", acc]
            else:
                total = sum(r["toks"] - 1 for r in rs) / union_len([(r["first"], r["last"]) for r in rs])
                row += [str(len(ds)), f"{st.median(r['tps'] for r in rs):.0f}", f"{total:.0f}", f"{sum(r['toks'] for r in rs) / wall:.0f}", acc]
        name = f"{L // 1024}k" if L % 1024 == 0 else f"{L:,}"
        print(f"| {name} | " + " | ".join(row) + " |", flush=True)


if __name__ == "__main__":
    main()
