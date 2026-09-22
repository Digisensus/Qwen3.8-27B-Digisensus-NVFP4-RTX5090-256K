# How Qwen3.8-27B-Digisensus was made

**Goal:** use the same GPU memory as another RTX 5090-adjusted NVFP4 quant of Qwen3.8-27B,
but stay closer to the original bf16 model ("stock"). The way: find out which parts of the
network lose the most quality when quantized, and give those parts more bits, paid for by
parts that lose almost nothing.

Everything below was measured, not guessed. The scripts are in `quantization/`.

## 1. The measuring stick

- A fixed prompt set of 251 items, tagged by task (`quantization/frozen-v1.jsonl`, checksum in
  `frozen-v1.sha256`): prose 60, tool calls 56 (with real tool schemas), JSON 50, reasoning 45
  (thinking on), code 40.
- Stock bf16 Qwen3.8-27B (two GPUs, stock vLLM, MTP off — MTP corrupts `prompt_logprobs`,
  vllm#53488) writes an answer to each prompt once. We keep the answer and, for every
  token of it, the 20 most likely next tokens with their probabilities (106,813 positions).
- A quantized model is then fed exactly the same tokens and we measure, per position, how
  far its probabilities are from stock (KL divergence on those 20 tokens plus a bucket for
  the rest) and whether it picks the same top token.
- Runs are exactly repeatable, so the only uncertainty is which prompts are in the set. We
  report 95% intervals from resampling the prompts.
- Check: scoring the stock engine against itself gives KL 0.00001 and 99.91% top-1.

## 2. The control (R0)

We took the other quant's scheme from its `config.json` (FP8 on attention, the Gated-DeltaNet
projections, `lm_head` and the MLPs of layers 56–63; NVFP4 on the other MLPs; FP8 KV scales)
and re-quantized stock with it, using our own calibration data. Result: within about 3%
KL of the other quant, same top-1. So our pipeline reproduces its quality, and changing
only the calibration data changes nothing. What matters is **where the bits go**.

## 3. Where does quality get lost? (`quantization/sensitivity.py`)

vLLM fuses some weights into one kernel, so they must share a format. Our units are these
fused groups: GDN `in_proj_qkv+in_proj_z`, GDN `out_proj`, attention `q/k/v`, attention
`o_proj`, MLP `gate+up`, MLP `down`, `lm_head`, `embed_tokens`. For each unit we quantize
**only that unit** (simulated in the bf16 model, bit-exact against compressed-tensors) to
FP8 and to NVFP4 and measure the KL of the output vs stock.

Three things we had to learn to make this measurement mean anything:

1. **Score only text the model wrote itself.** On user-written call transcripts one FP8
   unit "cost" more than the whole quantized model: some positions are so undecided that
   two numerically equivalent kernels differ by KL 0.17 there. We score the stock model's
   own answers (even-numbered items; the odd ones are held out for the final result) and
   assistant turns only.
2. **bf16 has a noise floor.** Any change to any weight reshuffles rounding downstream:
   0.1% random noise costs about 4e-4 KL — as much as one FP8 unit. So we measure bands of
   4 layers at a time and subtract each band's noise floor.
3. The vocabulary is 248k tokens; the logits do not fit in memory at once. Process 512
   positions at a time.

What we found (signal above the noise floor, ×1e-4; full map in
`results/layer-a/sensitivity-map-2026-09-21.txt`):

- FP8 costs little anywhere (0.2–4.4 per band).
- NVFP4 on MLPs is cheap in early layers (0–23: 2–6) and expensive later (10–17), worst in
  layers 60–63 (`gate+up`: 31.6).
- **`lm_head` at FP8 costs 10.3 — more than any band. The embeddings at 8 bit cost 0.3.**

## 4. Moving the bits (`quantization/allocate.py`)

An exact knapsack picks a format per unit to minimise the summed KL at the other quant's
byte budget. We kept the Gated-DeltaNet projections at FP8 or better (published results
show quantizing linear attention harder hurts long reasoning); letting them go to NVFP4 would
have gained only 1.3 points more.

The result (`quantization/recipes/digisensus_r6.yaml`):

- `embed_tokens`: bf16 → **INT8 per row** (frees 1.18 GiB; this pays for everything else)
- MLP layers 28–35 and 48–55: NVFP4 → **FP8** (the 4-bit MLPs that lost the most)
- MLP layers 56–59: FP8 → NVFP4 (the other quant kept them at FP8; they lose less than 48–55)
- layer 63 attention `o_proj`: FP8 → bf16
- everything else as in the other quant

Without the embedding move, the whole reallocation was worth only 3%.

## 5. Serving an INT8 embedding

- vLLM 0.28 can serve an INT8 embedding (dequantize on lookup), but the Qwen3.5 model code
  never passes it the quantization config. Fixed in `patches/files/qwen3_5.py` and
  `qwen3_5_mtp.py`.
- llm-compressor 0.13 wrote broken embedding scales (54% zero, some NaN), which made every
  lookup return zeros. `quantization/embed_int8.py` writes the INT8 embedding directly
  (per-row rounding, packed the way the vLLM kernel reads it; 0.8% round-trip error).

## 6. Results

KL vs stock on the **held-out** half of the prompt set (125 items the scan never saw):

| | KL vs stock | top-1 |
|---|---|---|
| another 5090-adjusted NVFP4 quant | 0.01520 | 95.27% |
| R0 (its scheme, our calibration) | 0.01569 | 95.51% |
| **Qwen3.8-27B-Digisensus** | **0.01148** | **96.07%** |

Digisensus vs the other quant: 24.5% lower KL [95% CI 21.8–26.9]: prose −26%, reasoning
−26%, code −22%, tool calls −19%. Same weight bytes (+9 MiB), same 266,197-token KV pool on
one RTX 5090.

Thinking on, 24 real requests (JSON scoring, tool agents, reasoning), same sampling as the
apps use, MTP on: Digisensus wrote 8% fewer reasoning tokens, hit the token limit 2 times
instead of 5 (of 48 runs), and gave 46 valid answers instead of 43. MTP accepted 67.9% of
draft tokens vs 65.9% — a model closer to stock matches the stock-trained MTP head better.
