#!/usr/bin/env python3
"""Decode speed on REAL text (random-token benchmarks mis-state MTP speculative decoding:
acceptance depends on how predictable the output is). Streams real prompts from the frozen
set, measures per-request decode tok/s = (tokens - 1) / (t_last - t_first), and the MTP
acceptance rate from the engine's /metrics delta.
Usage: python3 decode_real.py --port 9003 [--concurrency 1,8] [--n 16]
"""
import argparse, json, statistics as st, time, urllib.request, re
from concurrent.futures import ThreadPoolExecutor


def metric(port, name):
    t = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=10).read().decode()
    return sum(float(x) for x in re.findall(rf"^{re.escape(name)}\{{[^}}]*\}} ([0-9.e+]+)$", t, re.M))


def one(port, prompt, think, max_tokens):
    body = {"model": "qwen", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.7, "top_p": 0.8, "top_k": 20, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": think}}
    q = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; toks = 0
    for line in urllib.request.urlopen(q, timeout=600):
        line = line.decode().strip()
        if not line.startswith("data: {"):
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            toks = d["usage"]["completion_tokens"]
        ch = (d.get("choices") or [{}])[0].get("delta") or {}
        if ch.get("content") or ch.get("reasoning"):
            now = time.time(); first = first or now; last = now
    return {"ttft": first - t0, "tps": (toks - 1) / max(last - first, 1e-6), "toks": toks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="9003"); ap.add_argument("--concurrency", default="1,8")
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--set", default="frozen-v1.jsonl")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.set)]
    prompts = [(x["prompt"], x["tag"] == "reasoning") for x in items if x["tag"] in ("prose", "reasoning", "json_struct")]
    print(f"| workload | concurrency | decode per user (tok/s, median) | aggregate (tok/s) | MTP acceptance |")
    print(f"|---|---|---|---|---|")
    for c in map(int, a.concurrency.split(",")):
        for label, think in (("prose/JSON, thinking off", False), ("reasoning, thinking on", True)):
            ps = [p for p, t in prompts if t == think][:a.n]
            d0, a0 = metric(a.port, "vllm:spec_decode_num_draft_tokens_total"), metric(a.port, "vllm:spec_decode_num_accepted_tokens_total")
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=c) as ex:
                rs = list(ex.map(lambda p: one(a.port, p, think, a.max_tokens), ps))
            wall = time.time() - t0
            d1, a1 = metric(a.port, "vllm:spec_decode_num_draft_tokens_total"), metric(a.port, "vllm:spec_decode_num_accepted_tokens_total")
            print(f"| {label} | {c} | {st.median(r['tps'] for r in rs):.0f} | {sum(r['toks'] for r in rs) / wall:.0f} | {100 * (a1 - a0) / max(d1 - d0, 1):.1f}% |", flush=True)


if __name__ == "__main__":
    main()
