#!/usr/bin/env python3
"""Bit reallocation: pick a format per unit to minimise total output KL under a byte budget.

Inputs: results/sensitivity.jsonl (cost of each unit at each format, alone, vs bf16).
Model:  total KL ~ sum of unit KLs (checked afterwards by measuring the chosen config
        as a whole). Multiple-choice knapsack solved exactly by DP over MiB.
Budget: by default the bytes of the R0/the reference quant scheme — same memory, better placement.

Bytes per parameter: bf16 2, fp8 1 (+ per-channel scale, negligible), nvfp4 0.5625
(4-bit values + one e4m3 scale per 16). Units not in the scan (norms, conv, GDN a/b,
vision, MTP) are fixed and excluded from both sides.

Usage: python3 harness/allocate.py [--budget-delta-mib 0] [--emit recipes/r5_alloc.yaml]
"""
import argparse, collections, json, re
import numpy as np

GROUP = 4
BPP = {"bf16": 2.0, "fp8": 1.0, "fp8w": 1.0, "nvfp4": 0.5625}


def r0_format(unit):
    """The the reference quant/R0 scheme for a unit."""
    if unit == "embed":
        return "bf16"
    if unit == "lm_head":
        return "fp8"
    kind = unit.split(".", 1)[1]
    layer = int(unit[1:3]) * (GROUP if unit.startswith("B") else 1)
    if kind.startswith("mlp."):
        return "fp8" if layer >= 56 else "nvfp4"
    return "fp8"          # attn.qkv/o, gdn.qkvz/out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="results/sensitivity.jsonl")
    ap.add_argument("--budget-delta-mib", type=float, default=0.0, help="+ = allow more memory than R0")
    ap.add_argument("--allow-bf16", action="store_true", help="bf16 as an option for scanned units")
    ap.add_argument("--emit", default="")
    ap.add_argument("--forbid", default="", help="regex unit:fmt pairs to forbid, e.g. 'gdn\\..*:nvfp4'")
    a = ap.parse_args()

    raw, params, floor = collections.defaultdict(dict), {}, {}
    for l in open(a.scan):
        r = json.loads(l)
        if r["fmt"] == "noise":
            floor[r["unit"].split(".")[0]] = r["kl"]; continue
        raw[r["unit"]][r["fmt"]] = r["kl"]
        params[r["unit"]] = r["params"]
    # signal = measured KL minus the band's bf16-noise floor (any perturbation costs that)
    bands = sorted(floor)
    def fl(u):
        if u.startswith("B"):
            return floor.get(u.split(".")[0], 0.0)
        return floor[bands[0]] if u == "embed" else floor[bands[-1]]   # embed ~ input, lm_head ~ output
    cost = {u: {f: max(k - fl(u), 0.0) for f, k in d.items()} for u, d in raw.items()}
    units = sorted(cost)
    opts = {}
    for u in units:
        o = {f: c for f, c in cost[u].items()}
        if u == "embed":
            o = {"bf16": 0.0, "fp8w": o.get("fp8w", 0.0)}
        elif a.allow_bf16:
            o["bf16"] = 0.0
        if a.forbid:
            o = {f: c for f, c in o.items() if not re.search(a.forbid, f"{u}:{f}")} or o
        opts[u] = o

    mib = lambda u, f: params[u] * BPP[f] / 2**20
    r0 = {u: r0_format(u) for u in units}
    r0_bytes = sum(mib(u, r0[u]) for u in units)
    r0_cost = sum(opts[u].get(r0[u], cost[u].get(r0[u], 0.0)) for u in units)
    budget = r0_bytes + a.budget_delta_mib

    # DP over integer MiB: best[b] = min cost using at most b MiB
    B = int(budget) + 1
    INF = float("inf")
    best = np.full(B, INF); best[0] = 0.0
    choice = []
    for u in units:
        new = np.full(B, INF); pick = np.full(B, -1, dtype=np.int16)
        fl = list(opts[u].items())
        for j, (f, c) in enumerate(fl):
            w = int(round(mib(u, f)))
            if w >= B:
                continue
            cand = np.full(B, INF); cand[w:] = best[:B - w] + c
            better = cand < new
            new[better] = cand[better]; pick[better] = j
        choice.append((fl, pick)); best = new
    b = int(np.argmin(best)); total = best[b]
    alloc = {}
    for u, (fl, pick) in zip(reversed(units), reversed(choice)):
        j = pick[b]; f = fl[j][0]; alloc[u] = f; b -= int(round(mib(u, f)))
    used = sum(mib(u, alloc[u]) for u in units)

    print(f"units {len(units)}  R0 bytes {r0_bytes:,.0f} MiB  sum-KL {r0_cost:.4e}")
    print(f"budget {budget:,.0f} MiB -> used {used:,.0f} MiB  sum-KL {total:.4e}  "
          f"({100 * (total - r0_cost) / r0_cost:+.1f}% vs R0)\n")
    moves = collections.Counter()
    for u in units:
        if alloc[u] != r0[u]:
            kind = re.sub(r"^[LB]\d+\.", "", u)
            moves[(kind, r0[u], alloc[u])] += 1
    print("changes vs R0 (unit kind: from -> to, count):")
    for (k, f, t), n in sorted(moves.items()):
        ls = sorted(int(u[1:3]) for u in units if re.sub(r"^[LB]\d+\.", "", u) == k and r0[u] == f and alloc[u] == t and u[0] in "LB")
        print(f"  {k:14s} {f:5s} -> {t:5s} x{n:3d}  {'bands' if units[0].startswith('B') else 'layers'} {ls}")
    if a.emit:
        json.dump({"budget_mib": budget, "used_mib": used, "sum_kl": total, "r0_sum_kl": r0_cost,
                   "alloc": alloc}, open(a.emit, "w"), indent=1)
        print(f"\nwrote {a.emit}")


if __name__ == "__main__":
    main()
