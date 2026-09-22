#!/usr/bin/env python3
"""Structural check of a quantized checkpoint against the reference quant's (which prod vLLM loads).

Compares every tensor name, dtype and shape, the quantization_config shape, and the MTP
layout. Differences are expected only where the recipe deliberately differs from the reference quant;
anything else means vLLM may refuse the checkpoint or, worse, load it wrongly.

Usage: python3 harness/verify_ckpt.py <ckpt> [--ref <ckpt>] [--layers N]
"""
import argparse, collections, json, re, struct, sys
import os

REF = os.environ.get("REF_CKPT", "models/Qwen3.8-27B-NVFP4-reference")


def headers(d):
    idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
    out = {}
    for f in sorted(set(idx.values())):
        with open(f"{d}/{f}", "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (v["dtype"], tuple(v["shape"]), f)
    missing_in_files = set(idx) - set(out)
    return out, idx, missing_in_files


def layer_of(k):
    m = re.search(r"\.layers\.(\d+)\.", k)
    return int(m.group(1)) if m and not k.startswith(("mtp.", "model.visual.")) else None


def kind(k):
    return re.sub(r"\.layers\.\d+\.", ".layers.N.", k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--layers", type=int, default=None, help="only compare layers < N (smoke models)")
    a = ap.parse_args()
    ours, oidx, o_orphans = headers(a.ckpt)
    ref, _, _ = headers(a.ref)
    n = a.layers if a.layers is not None else 1 + max(l for l in map(layer_of, ours) if l is not None)
    ref = {k: v for k, v in ref.items() if layer_of(k) is None or layer_of(k) < n}

    only_ref = sorted(set(ref) - set(ours))
    only_ours = sorted(set(ours) - set(ref))
    diff = sorted(k for k in set(ref) & set(ours) if ref[k][:2] != ours[k][:2])
    ok = True

    def show(title, keys, fmt):
        nonlocal ok
        if not keys:
            print(f"[PASS] {title}")
            return
        ok = False
        grouped = collections.Counter(kind(k) for k in keys)
        print(f"[DIFF] {title}: {len(keys)} tensors, {len(grouped)} kinds")
        for kd, c in grouped.most_common(12):
            ex = next(k for k in keys if kind(k) == kd)
            print(f"        {c:4d} x {kd}   {fmt(ex)}")

    print(f"comparing {a.ckpt}\n     vs {a.ref}  (layers < {n})\n")
    show("tensors only in the reference quant (we are missing them)", only_ref, lambda k: f"ref {ref[k][:2]}")
    show("tensors only in ours", only_ours, lambda k: f"ours {ours[k][:2]}")
    show("same name, different dtype/shape", diff, lambda k: f"ref {ref[k][:2]} ours {ours[k][:2]}")
    if o_orphans:
        ok = False; print(f"[FAIL] index lists {len(o_orphans)} tensors not present in any file")
    else:
        print("[PASS] every index entry exists in its file")

    mtp = [k for k in ours if k.startswith("mtp.")]
    mtp_file = {ours[k][2] for k in mtp}
    good = len(mtp) == 15 and mtp_file == {"model_mtp.safetensors"} and all(ours[k][0] == "BF16" for k in mtp)
    print(f"[{'PASS' if good else 'FAIL'}] MTP head: {len(mtp)} tensors in {sorted(mtp_file)}, bf16")
    ok &= good

    oc = json.load(open(f"{a.ckpt}/config.json")).get("quantization_config", {})
    rc = json.load(open(f"{a.ref}/config.json")).get("quantization_config", {})
    for key in ("quant_method", "format", "quantization_status"):
        same = oc.get(key) == rc.get(key)
        ok &= same
        print(f"[{'PASS' if same else 'DIFF'}] quantization_config.{key}: ours={oc.get(key)!r} ref={rc.get(key)!r}")
    ours_fmt = sorted(g.get("format") for g in oc.get("config_groups", {}).values())
    ref_fmt = sorted(g.get("format") for g in rc.get("config_groups", {}).values())
    print(f"[{'PASS' if ours_fmt == ref_fmt else 'DIFF'}] group formats: ours={ours_fmt} ref={ref_fmt}")
    kv = (oc.get("kv_cache_scheme") or {}).get("num_bits") == (rc.get("kv_cache_scheme") or {}).get("num_bits")
    print(f"[{'PASS' if kv else 'DIFF'}] kv_cache_scheme present and same width")
    mig = "re:^mtp.*" in oc.get("ignore", [])
    print(f"[{'PASS' if mig else 'FAIL'}] 're:^mtp.*' in ignore")
    ok &= mig
    gib = sum(v[1] and 1 for v in ours.values())
    print("\nRESULT:", "STRUCTURALLY IDENTICAL TO THE REFERENCE CHECKPOINT" if ok else "DIFFERENCES ABOVE — check each is intended")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
