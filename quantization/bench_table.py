#!/usr/bin/env python3
"""Turn bench.sh results (vllm bench serve JSON) into the published markdown table.

PP (prefill) = input tokens / mean TTFT of single-stream requests.
Decode per user = 1000 / mean TPOT. Aggregate = total output tokens / wall time of the
concurrent wave (includes queued prefills, so it is lower than users x per-user decode).
Usage: python3 bench_table.py results/bench
"""
import glob, json, os, re, sys


def main(d):
    rows = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        m = re.match(r"(single|conc(\d+))-(\d+)\.json", os.path.basename(f))
        if not m:
            continue
        r = json.load(open(f))
        L = int(m.group(3))
        e = rows.setdefault(L, {})
        if m.group(1) == "single":
            e["pp"] = L / (r["mean_ttft_ms"] / 1000)
            e["ttft"] = r["mean_ttft_ms"] / 1000
            e["tps1"] = 1000 / r["mean_tpot_ms"]
        else:
            e["conc"] = int(m.group(2))
            e["tpsN"] = 1000 / r["mean_tpot_ms"]
            e["aggN"] = r["output_throughput"]
            e["ttftN"] = r["mean_ttft_ms"] / 1000
    print("| context | prefill (tok/s) | TTFT single | decode, 1 user (tok/s) | concurrency | decode per user (tok/s) | aggregate output (tok/s) | mean TTFT under load |")
    print("|---|---|---|---|---|---|---|---|")
    for L in sorted(rows):
        e = rows[L]
        name = f"{L // 1024}k" if L % 1024 == 0 else f"{L:,} (max)"
        c = (f"{e['conc']}", f"{e['tpsN']:.0f}", f"{e['aggN']:.0f}", f"{e['ttftN']:.1f} s") if "conc" in e else ("1", "—", "—", "—")
        print(f"| {name} | {e.get('pp', 0):,.0f} | {e.get('ttft', 0):.2f} s | {e.get('tps1', 0):.0f} | " + " | ".join(c) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/bench")
