#!/usr/bin/env python3
"""Layer B: does a candidate think longer / truncate more than the baseline?

Published Qwen3.5 results: quantized models think longer (~2x truncation). Layer-A KL
cannot see that — it scores bf16's text, not the candidate's own trajectories. This runs
real thinking-ON requests on two engines and compares, per request:
  reasoning tokens, content tokens, finish_reason=length, answer validity.
Modes: greedy (temperature 0 — paired, same request same settings) and the request's own
logged sampling params (x runs). Sources: real traffic (thinking on) + frozen reasoning items.
Privacy: prints counts only.

Usage: python3 harness/think_ab.py --a 9001:reference --b 9003:r6 [--n-traffic 16] [--runs 2]
"""
import argparse, collections, copy, glob, hashlib, json, statistics as st, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import replay_audit as ra


def load(n_traffic, n_frozen):
    reqs, seen = [], set()
    for f in sorted(glob.glob(ra.LOGS)):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            q = r.get("request") or {}
            if r.get("status") != 200 or not q.get("messages") or not ra.thinking(q) \
                    or any(m in json.dumps(q, ensure_ascii=False) for m in ra.MARK):
                continue
            h = hashlib.md5(json.dumps(q["messages"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            kind = "tools" if q.get("tools") else "json" if q.get("response_format") else "chat"
            reqs.append({"kind": kind, "q": q, "stream": bool(r.get("streamed"))})
    # spread over kinds
    by = collections.defaultdict(list)
    for r in reqs:
        by[r["kind"]].append(r)
    picked = []
    while len(picked) < n_traffic and any(by.values()):
        for k in list(by):
            if by[k] and len(picked) < n_traffic:
                picked.append(by[k].pop(len(by[k]) // 2))
    fr = [json.loads(l) for l in open("prompts/frozen-v1.jsonl")]
    fr = [x for k, x in enumerate(fr) if x["tag"] == "reasoning" and k % 2 == 1][:n_frozen]   # held-out
    for x in fr:
        picked.append({"kind": "reasoning", "stream": False,
                       "q": {"model": "qwen", "messages": [{"role": "user", "content": x["prompt"]}],
                             "max_tokens": 16384, "temperature": 1.0, "top_p": 0.95, "top_k": 20}})
    return picked


def valid(kind, q, o):
    if o["finish"] == "length":
        return False
    if kind == "json":
        try:
            json.loads(o["content"]); return True
        except Exception:
            return False
    if kind == "tools":
        if o["tools"]:
            try:
                return all(json.loads(t["args"] or "{}") is not None for t in o["tools"].values())
            except Exception:
                return False
        return bool(o["content"].strip())
    return bool(o["content"].strip())


def run(port, item, mode):
    q = copy.deepcopy(item["q"])
    q["max_tokens"] = min(q.get("max_tokens") or 16384, 16384)
    if mode == "greedy":
        q["temperature"] = 0
    try:
        o = ra.call(port, q, item["stream"])
    except Exception as e:
        return {"error": type(e).__name__}
    return {"reason": ra.ntok(port, o["reasoning"]), "content": ra.ntok(port, o["content"]),
            "length": o["finish"] == "length", "valid": valid(item["kind"], q, o)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--n-traffic", type=int, default=16); ap.add_argument("--n-frozen", type=int, default=8)
    ap.add_argument("--runs", type=int, default=2); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    (pa, na), (pb, nb) = a.a.split(":"), a.b.split(":")
    items = load(a.n_traffic, a.n_frozen)
    print(f"{len(items)} thinking-on requests {dict(collections.Counter(i['kind'] for i in items))}; "
          f"greedy x1 + logged sampling x{a.runs}; {na} on :{pa}, {nb} on :{pb}\n", flush=True)
    jobs = [(i, mode, rep) for i in range(len(items)) for mode, reps in (("greedy", 1), ("sampled", a.runs)) for rep in range(reps)]
    res = {}
    def worker(port):
        def f(j):
            i, mode, rep = j
            return (port, i, mode, rep), run(port, items[i], mode)
        return f
    with ThreadPoolExecutor(max_workers=2) as outer:
        futs = [outer.submit(lambda p=p: list(ThreadPoolExecutor(max_workers=a.workers).map(worker(p), jobs))) for p in (pa, pb)]
        for fu in futs:
            for k, v in fu.result():
                res[k] = v
    for mode in ("greedy", "sampled"):
        print(f"=== {mode} ===")
        print(f"{'kind':10s} {'n':>3s} | {'reason tok (median)':>20s} {'mean':>13s} | {'content (mean)':>14s} | {'truncated':>11s} | {'valid':>11s} | {nb + ' shorter':>10s}")
        for kind in sorted({i["kind"] for i in items}) + ["ALL"]:
            idx = [i for i, it in enumerate(items) if kind == "ALL" or it["kind"] == kind]
            reps = range(1 if mode == "greedy" else a.runs)
            A = [res[(pa, i, mode, r)] for i in idx for r in reps]; B = [res[(pb, i, mode, r)] for i in idx for r in reps]
            ok = [(x, y) for x, y in zip(A, B) if "error" not in x and "error" not in y]
            if not ok:
                continue
            ra_, rb_ = [x["reason"] for x, _ in ok], [y["reason"] for _, y in ok]
            shorter = sum(y["reason"] < x["reason"] for x, y in ok)
            print(f"{kind:10s} {len(ok):3d} | {st.median(ra_):8.0f} -> {st.median(rb_):8.0f}   {st.mean(ra_):6.0f}->{st.mean(rb_):6.0f} | "
                  f"{st.mean(x['content'] for x, _ in ok):5.0f}->{st.mean(y['content'] for _, y in ok):5.0f} | "
                  f"{sum(x['length'] for x, _ in ok):3d} -> {sum(y['length'] for _, y in ok):3d} | "
                  f"{sum(x['valid'] for x, _ in ok):3d} -> {sum(y['valid'] for _, y in ok):3d} | {shorter:4d}/{len(ok)}")
        errs = collections.Counter((p, v["error"]) for (p, i, m, r), v in res.items() if m == mode and "error" in v)
        if errs:
            print("  errors:", dict(errs))
        print()
    print(f"columns: {na} -> {nb}")


if __name__ == "__main__":
    main()
