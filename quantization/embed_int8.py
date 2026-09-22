#!/usr/bin/env python3
"""Replace embed_tokens in a quantized checkpoint with an INT8 per-row embedding, written
in compressed-tensors pack-quantized format — what vLLM's CompressedTensorsEmbeddingWNA16Int
(dequant-on-lookup) serves. Needs the qwen3_5.py / qwen3_5_mtp.py patches in ~/vllm-patches.

Why not llm-compressor: its QuantizationModifier on the Embedding (0.13) wrote broken scales
(54% zero, some NaN, values saturated at 0/255) -> vLLM looked up all-zero rows.

Format (matches the vLLM kernel): q = round(w / s) in [-127, 127], s = amax(row) / 127
(bf16); stored as q + 128 in uint8, four per int32, little-endian (byte k = bits 8k..8k+7).

Usage: venv/bin/python harness/embed_int8.py <quantized ckpt dir> [--source <bf16 dir>]
"""
import argparse, json, os, struct
import torch
from safetensors import safe_open
from safetensors.torch import save_file

KEY = "model.language_model.embed_tokens"


def quantize_rows(w):
    w = w.float()
    s = (w.abs().amax(dim=1, keepdim=True) / 127.0)
    s = torch.where(s > 0, s, torch.ones_like(s))            # all-zero rows: any scale works
    s = s.to(torch.bfloat16)                                   # the kernel multiplies by the bf16 scale
    q = torch.clamp(torch.round(w / s.float()), -127, 127).to(torch.int64) + 128
    q = q.reshape(q.shape[0], -1, 4)
    packed = (q[..., 0] | (q[..., 1] << 8) | (q[..., 2] << 16) | (q[..., 3] << 24))
    packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
    return packed, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--source", default=os.environ.get("BF16_MODEL", "models/Qwen3.8-27B"))
    a = ap.parse_args()

    sidx = json.load(open(f"{a.source}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{a.source}/{sidx[KEY + '.weight']}", "pt") as f:
        w = f.get_tensor(KEY + ".weight")
    packed, scale = quantize_rows(w)
    deq = ((((packed.to(torch.int64) & 0xFFFFFFFF)[..., None] >> torch.tensor([0, 8, 16, 24])) & 255) - 128)
    deq = deq.reshape(w.shape).float() * scale.float()
    live = w.float().abs().amax(1) > 0
    rel = ((deq - w.float())[live].norm() / w.float()[live].norm()).item()
    print(f"embed {tuple(w.shape)} -> packed {tuple(packed.shape)} int32 + scale {tuple(scale.shape)}; "
          f"round-trip rel err {rel:.4f}")

    idx_path = f"{a.ckpt}/model.safetensors.index.json"
    idx = json.load(open(idx_path))
    main = idx["weight_map"][KEY + ".weight"] if KEY + ".weight" in idx["weight_map"] else idx["weight_map"][KEY + ".weight_packed"]
    tensors = {}
    with safe_open(f"{a.ckpt}/{main}", "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            if not k.startswith(KEY + "."):
                tensors[k] = f.get_tensor(k)
    tensors[KEY + ".weight_packed"] = packed
    tensors[KEY + ".weight_scale"] = scale
    tensors[KEY + ".weight_shape"] = torch.tensor(list(w.shape), dtype=torch.int64)
    tmp = f"{a.ckpt}/{main}.tmp"
    save_file(tensors, tmp, metadata=meta or {"format": "pt"})
    os.replace(tmp, f"{a.ckpt}/{main}")
    for k in list(idx["weight_map"]):
        if k.startswith(KEY + "."):
            del idx["weight_map"][k]
    for k in ("weight_packed", "weight_scale", "weight_shape"):
        idx["weight_map"][f"{KEY}.{k}"] = main
    idx.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(f"{a.ckpt}/{f}") for f in set(idx["weight_map"].values()))
    json.dump(idx, open(idx_path, "w"), indent=2)

    cfg_path = f"{a.ckpt}/config.json"
    c = json.load(open(cfg_path))
    q = c["quantization_config"]
    q["config_groups"] = {n: g for n, g in q["config_groups"].items()
                          if not any("embed_tokens" in t for t in g.get("targets", []))}
    q["config_groups"]["embed_int8"] = {
        "targets": ["re:.*embed_tokens$"], "format": "pack-quantized",
        "weights": {"num_bits": 8, "type": "int", "strategy": "channel", "symmetric": True,
                    "dynamic": False, "group_size": None, "observer": "minmax",
                    "observer_kwargs": {}, "actorder": None, "block_structure": None,
                    "scale_dtype": None, "zp_dtype": None},
        "input_activations": None, "output_activations": None}
    q["format"] = "mixed-precision"
    json.dump(c, open(cfg_path, "w"), indent=2)
    print(f"rewrote {main} and config.json (embed_int8 group, pack-quantized)")


if __name__ == "__main__":
    main()
