# Reproducing the quantization

Requirements: one RTX 5090 (quantization), two for the bf16 reference and the sensitivity
scan; ~80 GB RAM; Python venv with `llmcompressor==0.13.0`, `compressed-tensors==0.18.0`,
`transformers==5.14.1`, `torch==2.13`, `flash-linear-attention` + `fla-core` 0.5.2, `einops`.
Paths are set by env vars: `BF16_MODEL` (default `models/Qwen3.8-27B`), `REF_CKPT`,
`TRAFFIC_LOGS` (optional private request logs, JSONL).

```bash
# 1. calibration set (chat-templated: native tool-call XML, <think> blocks)
python build_calibration.py --out data/calib.jsonl            # add --no-traffic for public-only
# 2. quantize with the Digisensus recipe (GPTQ; weights 28 GiB in RAM, rest offloaded to disk)
CUDA_VISIBLE_DEVICES=<gpu> python quantize.py --recipe recipes/digisensus_r6.yaml \
    --calib data/calib.jsonl --out models/Qwen3.8-27B-Digisensus
# 3. INT8 embedding (post-step; llm-compressor 0.13's embedding quantization is broken)
python embed_int8.py models/Qwen3.8-27B-Digisensus
# 4. structure check against a checkpoint known to load in vLLM
python verify_ckpt.py models/Qwen3.8-27B-Digisensus --ref <another compressed-tensors Qwen3.8-27B checkpoint>
```

Measuring (see `../docs/METHOD.md`):

| script | what |
|---|---|
| `build_promptset.py` | frozen capability-tagged prompt set (`frozen-v1.jsonl`, sha256 alongside) |
| `score.py reference / compare` | Layer A: bf16 continuations + top-20 logprobs; candidate teacher-forced KL / top-1 |
| `compare_items.py` | paired bootstrap between two runs, `--held-out` for items the scan never saw |
| `sensitivity.py` | per-unit / per-band KL cost of FP8 and NVFP4 in the bf16 model (two GPUs) |
| `allocate.py` | multiple-choice knapsack at a byte budget -> `emit_recipe.py` -> recipe |
| `think_ab.py` | Layer B: reasoning length, truncation, validity on real thinking-on requests |
| `replay_audit.py` | replays logged requests verbatim, audits responses (prints counts only) |
| `bench.sh` | serving benchmark: PP, single and concurrent decode, 1k .. 256k context |

Calibration used for the published weights: ~641k tokens — 50% private [Digisensus.com](https://digisensus.com) test traffic
(call-centre QA/audit/PII/scoring and tool-calling agents; **not released**), 18%
NousResearch/hermes-function-calling-v1 tool calls (Apache-2.0), 12% hermes json-mode
(Apache-2.0), 20% HuggingFaceH4/ultrachat_200k (MIT).
