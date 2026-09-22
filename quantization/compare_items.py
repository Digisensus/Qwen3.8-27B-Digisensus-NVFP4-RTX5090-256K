#!/usr/bin/env python3
"""Paired comparison of two Layer-A runs (score.py compare --out X.json -> X.items.json).

KL = position-weighted mean over items (lower = closer to bf16). Uncertainty comes from
item sampling (runs are bit-reproducible), so it is a paired bootstrap over items.
--held-out restricts to items the sensitivity scan never saw (odd lines of the reference).

Usage: python3 harness/compare_items.py results/cmp-r0.items.json results/cmp-r6.items.json --held-out
"""
import argparse, json, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--ref", default="results/reference-bf16.jsonl")
    ap.add_argument("--held-out", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    A, B = json.load(open(a.a)), json.load(open(a.b))
    order = [json.loads(l)["id"] for l in open(a.ref)]
    keep = {i for k, i in enumerate(order) if (k % 2 == 1 or not a.held_out)}
    na, nb = a.a.split("/")[-1].replace(".items.json", ""), a.b.split("/")[-1].replace(".items.json", "")
    print(f"{nb} vs {na} — {'HELD-OUT items only' if a.held_out else 'all items'}; ΔKL = {nb} - {na}, negative = {nb} closer to bf16\n")
    print(f"{'tag':12s} {'items':>5s} {'KL ' + na:>12s} {'KL ' + nb:>12s} {'ΔKL':>7s} {'95% CI':>17s} {'top1 ' + na:>10s} {'top1 ' + nb:>10s}")
    rng = random.Random(0)
    tags = sorted({v["tag"] for v in A.values()})
    for t in tags + ["ALL", "ALL-no-code"]:
        ids = [k for k in A if k in keep and A[k]["n"] > 0 and (t == "ALL" or (t == "ALL-no-code" and A[k]["tag"] != "code") or A[k]["tag"] == t)]
        if not ids:
            continue
        def agg(s):
            n = sum(A[k]["n"] for k in s)
            return sum(A[k]["kl"] for k in s) / n, sum(B[k]["kl"] for k in s) / n
        ka, kb = agg(ids)
        bs = sorted(100 * (y - x) / x for x, y in (agg([rng.choice(ids) for _ in ids]) for _ in range(a.boot)))
        lo, hi = bs[int(0.025 * a.boot)], bs[int(0.975 * a.boot) - 1]
        n = sum(A[k]["n"] for k in ids)
        ta, tb = sum(A[k]["top1"] for k in ids) / n, sum(B[k]["top1"] for k in ids) / n
        star = "*" if hi < 0 or lo > 0 else " "
        print(f"{t:12s} {len(ids):5d} {ka:12.5f} {kb:12.5f} {100 * (kb - ka) / ka:+6.1f}% [{lo:+6.1f}, {hi:+6.1f}]{star} {100 * ta:9.2f}% {100 * tb:9.2f}%")
    print("\n* = 95% interval excludes zero")


if __name__ == "__main__":
    main()
