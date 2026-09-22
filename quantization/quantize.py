#!/usr/bin/env python3
"""Quantize the official Qwen3.8-27B bf16 checkpoint with an llm-compressor recipe.

Memory plan (host has 76 GB RAM shared with prod; bf16 weights are 52 GiB):
  - weights: `--cpu-mem` GiB in RAM, the rest offloaded to disk (compressed-tensors offload)
  - one decoder layer at a time on the GPU (sequential pipeline)
  - cached layer-boundary activations on CPU: ~tokens x 5120 x 2 B each

After saving:
  - the MTP head (transformers drops `mtp.*` on load) is copied verbatim from the bf16
    shards into model_mtp.safetensors and added to the index — same layout as the reference quant's
  - tokenizer / chat template / processor files are copied from the bf16 release
  - provenance.json records recipe, calibration mix, versions, timings

Usage:
  CUDA_VISIBLE_DEVICES=<uuid> venv/bin/python harness/quantize.py \
      --recipe recipes/r0_reference_equiv.yaml --calib data/calib-r0.jsonl --out ~/models/qwen38-r0
Smoke test (CPU, truncated model): add --model <small ckpt> --fp8-mlp-layers 3 --max-samples 8
"""
import argparse, collections, json, os, re, shutil, struct, sys, time

BF16 = os.environ.get("BF16_MODEL", "models/Qwen3.8-27B")
AUX = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
       "preprocessor_config.json", "video_preprocessor_config.json", "merges.txt", "vocab.json", "LICENSE"]


def layer_regex(layers):
    return "|".join(str(i) for i in sorted(layers))


def parse_range(s, n):
    out = set()
    for part in filter(None, s.split(",")):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return {i for i in out if i < n}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def copy_mtp(src_dir, out_dir):
    """Copy mtp.* tensors byte-for-byte into model_mtp.safetensors and register them."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    idx = json.load(open(f"{src_dir}/model.safetensors.index.json"))["weight_map"]
    keys = sorted(k for k in idx if k.startswith("mtp."))
    tensors = {}
    for k in keys:
        with safe_open(f"{src_dir}/{idx[k]}", "pt") as f:
            tensors[k] = f.get_tensor(k)
    save_file(tensors, f"{out_dir}/model_mtp.safetensors", metadata={"format": "pt"})
    ip = f"{out_dir}/model.safetensors.index.json"
    if os.path.exists(ip):
        index = json.load(open(ip))
    else:   # single-file save: create an index so vLLM finds both files
        h = read_header(f"{out_dir}/model.safetensors")
        index = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in h if k != "__metadata__"}}
    for k in keys:
        index["weight_map"][k] = "model_mtp.safetensors"
    index["metadata"]["total_size"] = sum(
        os.path.getsize(f"{out_dir}/{f}") for f in set(index["weight_map"].values()))
    json.dump(index, open(ip, "w"), indent=2)
    return len(keys), sum(t.numel() * t.element_size() for t in tensors.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=BF16)
    ap.add_argument("--fp8-mlp-layers", default="56-63", help="MLP layers kept at FP8 (rest NVFP4)")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--cpu-mem", type=float, default=24, help="GiB of weights kept in RAM; rest on disk")
    ap.add_argument("--offload-dir", default="offload")
    a = ap.parse_args()
    t0 = time.time()

    import torch
    from datasets import Dataset
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
    from compressed_tensors.offload import load_offloaded_model
    from llmcompressor import oneshot
    import llmcompressor, compressed_tensors, transformers

    cfg = AutoConfig.from_pretrained(a.model)
    n_layers = cfg.text_config.num_hidden_layers
    fp8 = parse_range(a.fp8_mlp_layers, n_layers)
    nvfp4 = set(range(n_layers)) - fp8
    recipe = open(a.recipe).read()
    recipe = recipe.replace("{FP8_MLP_LAYERS}", layer_regex(fp8) or "NONE_MATCHES")
    recipe = recipe.replace("{NVFP4_MLP_LAYERS}", layer_regex(nvfp4) or "NONE_MATCHES")
    if not fp8:   # drop the now-empty target line rather than leave a never-matching regex
        recipe = "\n".join(l for l in recipe.splitlines() if "NONE_MATCHES" not in l)

    rows = [json.loads(l) for l in open(a.calib)]
    if a.max_samples:
        rows = rows[:a.max_samples]
    rows = [{"input_ids": r["input_ids"][:a.max_len], "attention_mask": [1] * min(len(r["input_ids"]), a.max_len)}
            for r in rows]
    mix = collections.Counter()
    for r in [json.loads(l) for l in open(a.calib)][:len(rows)]:
        mix[r["src"]] += min(len(r["input_ids"]), a.max_len)
    ds = Dataset.from_list(rows)
    print(f"[{time.time()-t0:6.0f}s] layers={n_layers} nvfp4_mlp={len(nvfp4)} fp8_mlp={sorted(fp8)} "
          f"calib={len(rows)} samples {sum(mix.values()):,} tok {dict(mix)}", flush=True)

    os.makedirs(a.offload_dir, exist_ok=True)
    with load_offloaded_model(Qwen3_5ForConditionalGeneration):
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            a.model, dtype=torch.bfloat16, device_map="auto_offload",
            max_memory={"cpu": int(a.cpu_mem * 2**30)}, offload_folder=a.offload_dir)
    print(f"[{time.time()-t0:6.0f}s] loaded", flush=True)

    t1 = time.time()
    # text-only calibration: pass the tokenizer so llm-compressor does not build the
    # vision/video processor (needs torchvision, irrelevant here)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    oneshot(model=model, processor=tok, dataset=ds, recipe=recipe, num_calibration_samples=len(rows),
            max_seq_length=a.max_len, pad_to_max_length=False, batch_size=1,
            shuffle_calibration_samples=False, pipeline="sequential",
            sequential_targets=["Qwen3_5DecoderLayer"])
    t_cal = time.time() - t1
    print(f"[{time.time()-t0:6.0f}s] calibrated+quantized in {t_cal:.0f}s", flush=True)

    os.makedirs(a.out, exist_ok=True)
    # ONE shard, like the reference quant's checkpoint. transformers 5.14 save_pretrained has a bug on the
    # sharded + offloaded path (modeling_utils.py:3675 passes dict.update a generator of
    # 1-key dicts -> always raises); on 2026-09-21 it destroyed a finished 65-minute run.
    try:
        model.save_pretrained(a.out, save_compressed=True, max_shard_size="200GB")
    except Exception as e:
        print(f"save_pretrained failed ({type(e).__name__}: {e}); retrying without format revert", flush=True)
        model.save_pretrained(a.out, save_compressed=True, max_shard_size="200GB", save_original_format=False)
    for f in AUX:
        if os.path.exists(f"{a.model}/{f}"):
            shutil.copy2(f"{a.model}/{f}", f"{a.out}/{f}")
    n_mtp, b_mtp = copy_mtp(a.model, a.out)
    # vLLM must not treat the bf16 MTP head as quantized
    c = json.load(open(f"{a.out}/config.json"))
    q = c.get("quantization_config") or {}
    if "re:^mtp.*" not in q.get("ignore", []):
        q.setdefault("ignore", []).append("re:^mtp.*")
        c["quantization_config"] = q
        json.dump(c, open(f"{a.out}/config.json", "w"), indent=2)
    shutil.copy2(a.recipe, f"{a.out}/recipe.template.yaml")
    open(f"{a.out}/recipe.yaml", "w").write(recipe)
    json.dump({"source": a.model, "recipe": a.recipe, "fp8_mlp_layers": sorted(fp8),
               "calibration_datasets": [
                   {"id": "private", "use": "real test-traffic conversations and the model's own responses "
                    "(contact-centre QA/audit/PII/scoring, tool-calling agents), collected before 2026-09-21",
                    "published": False, "note": "used only as calibration activations; the data itself is not released"},
                   {"id": "HuggingFaceH4/ultrachat_200k", "files": "data/test_sft-*.parquet", "use": "prose", "license": "MIT"},
                   {"id": "NousResearch/hermes-function-calling-v1", "files": "func-calling-singleturn.json",
                    "use": "tool calling (converted to the model's native tool-call format)", "license": "Apache-2.0"},
                   {"id": "NousResearch/hermes-function-calling-v1", "files": "json-mode-singleturn.json, json-mode-agentic.json",
                    "use": "structured JSON output", "license": "Apache-2.0"}],
               "calibration": {"file": os.path.basename(a.calib), "samples": len(rows),
                               "tokens_by_source": dict(mix), "max_len": a.max_len},
               "mtp": {"tensors": n_mtp, "bytes": b_mtp, "dtype": "bf16 (copied verbatim)"},
               "versions": {"llmcompressor": llmcompressor.__version__, "compressed_tensors": compressed_tensors.__version__,
                            "transformers": transformers.__version__, "torch": torch.__version__},
               "seconds": {"total": round(time.time() - t0), "calibration": round(t_cal)}},
              open(f"{a.out}/provenance.json", "w"), indent=2)
    sizes = collections.Counter()
    for f in os.listdir(a.out):
        if f.endswith(".safetensors"):
            sizes[f] = os.path.getsize(f"{a.out}/{f}")
    print(f"[{time.time()-t0:6.0f}s] saved {a.out}: {sum(sizes.values())/2**30:.2f} GiB "
          f"in {len(sizes)} files, mtp {n_mtp} tensors", flush=True)


if __name__ == "__main__":
    main()
