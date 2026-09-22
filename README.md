# Qwen3.8-27B-Digisensus — 256K context on one RTX 5090

This is the Qwen3.8-27B model that [Digisensus.com](https://digisensus.com) uses for testing on its RTX 5090 GPU
cluster. We are sharing the weights, the serving setup and the way it was made.

**Made for:** agentic use (tool calling, multi-step agents), structured output (JSON with a
schema) and reasoning with thinking on.

**On a single RTX 5090 (32 GB):** the full 262,144-token (256K) context, NVFP4 KV cache,
MTP speculative decoding, about 115 tokens/s for one user (92 at the full 256K).

Weights: [Digisensus/Qwen3.8-27B-Digisensus-NVFP4-RTX5090-256K](https://huggingface.co/Digisensus/Qwen3.8-27B-Digisensus-NVFP4-RTX5090-256K) (Hugging Face).

## Quick start

```bash
./image/build.sh                                   # vLLM v0.28.0 image for the RTX 5090 (sm_120)
hf download Digisensus/Qwen3.8-27B-Digisensus-NVFP4-RTX5090-256K \
    --local-dir models/Qwen3.8-27B-Digisensus-NVFP4-RTX5090-256K
cd compose && cp .env.example .env                 # set GPU_UUID (from `nvidia-smi -L`) and PORT
docker compose up -d                               # first start takes 3–5 minutes
curl localhost:9001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-digisensus","messages":[{"role":"user","content":"Hi"}]}'
```

A good start prints these lines in the log: `NVFP4KV-SM120: linear-V-scale store overlay ACTIVE`,
`decode_backend=xqa`, `GPU KV cache size: 266,197 tokens`,
`Sharing target model embedding weights with the draft model`.

You need the patched vLLM in this repo. Stock vLLM 0.28 cannot load these weights
(the embeddings are INT8) and cannot run the NVFP4 KV cache on an RTX 5090.

## How close is it to the original model?

We measure how far the quantized model's next-token predictions are from the original bf16
Qwen3.8-27B ("stock"): the KL divergence, averaged over every token of 125 held-out prompts
(prose, tool calls, JSON, reasoning, code). **Lower is closer to stock.** "Top-1" is how often
the quantized model picks the same next token as stock.

For comparison we include another NVFP4 quant of the same model that was also adjusted for
the RTX 5090. Both use the same amount of GPU memory.

| model | KL vs stock (lower is better) | top-1 agreement with stock |
|---|---|---|
| another 5090-adjusted NVFP4 quant | 0.01520 | 95.27% |
| **Qwen3.8-27B-Digisensus** | **0.01148** | **96.07%** |

Digisensus is 24.5% closer to stock (95% confidence interval 21.8%–26.9%). By task: prose 26%
closer, reasoning 26%, code 22%, tool calls 19%.

Thinking on, 24 real requests (JSON scoring, tool agents, reasoning): Digisensus wrote 8% fewer
reasoning tokens, hit the token limit 2 times instead of 5 (of 48 runs), and gave 46 valid
answers instead of 43. MTP accepted 67.9% of draft tokens instead of 65.9%.

How we got there, in short: we measured for every part of the network how much quality
it loses at FP8 and at NVFP4, then moved bits from the parts that do not need them to
the parts that do. The embeddings went to INT8 (almost no loss, frees 1.18 GiB); that
paid for FP8 instead of NVFP4 in the MLP layers that lose the most. Details:
[docs/METHOD.md](docs/METHOD.md).

## Speed (one RTX 5090, 480 W and 600 W, the compose config in this repo)

Numbers measured with `--max-model-len 262144`, NVFP4 KV cache, MTP k=3, prefix caching on,
up to 8 requests at once. Raw data and scripts: `results/bench/`, `quantization/bench.sh`,
`quantization/decode_ctx.py`.

**Prefill** (reading the prompt), one request at a time:

| context | prefill tokens/s | time to first token |
|---|---|---|
| 1k | 4,475 | 0.23 s |
| 8k | 8,517 | 0.96 s |
| 16k | 10,703 | 1.5 s |
| 32k | 6,494 | 5.1 s |
| 64k | 4,986 | 13 s |
| 128k | 3,347 | 39 s |
| 261,600 (max) | 1,661 | 158 s |

**Decode** (writing the answer) on real text: a long document of real conversations plus
"summarise", 384 output tokens, thinking off, temperature 0.7 (`quantization/decode_ctx.py`).
Random-token prompts are not used for decode because MTP speed depends on how predictable the
text is. **Prefix caching was switched off** for these runs so nothing is reused between
requests. Same engine, same test, only the GPU power limit changed. Single-user numbers are
the engine's own counters (tokens / decode-step time; two runs each agree within noise).

| context | 1 user: tokens/s, 480 W | 1 user: tokens/s, 600 W | step time 480 / 600 W | users at once | **all users together: tokens/s, 480 W** | **600 W** | 600 W vs 480 W |
|---|---|---|---|---|---|---|---|
| 1k | 113 | 118 | 23.0 / 23.1 ms | 8 | **692** | **807** | +17% |
| 8k | 119 | 120 | 23.3 / 23.2 ms | 8 | **339** | **351** | +4% |
| 16k | 122 | 118 | 23.3 / 23.3 ms | 8 | **195** | **206** | +6% |
| 32k | 119 | 117 | 23.9 / 24.1 ms | 8 | **87** | **93** | +7% |
| 64k | 132 | 116 | 24.7 / 24.7 ms | 4 | **40** | **42** | +5% |
| 128k | 113 | 114 | 26.4 / 26.3 ms | 2 | **22** | **24** | +9% |
| 258,000 | 98 | 92 | 30.0 / 29.8 ms | 1 | — | — | — |

Peak power draw: 495 W at the 480 W limit, 610 W at the 600 W limit.

What the table says:

- **One user needs no power.** At 480 W a single request runs at the same speed as at 600 W
  at every context (the differences are run-to-run noise). A decode step for one user is
  limited by reading the 21 GB of weights, not by compute, so the extra 120 W buys nothing.
- **Many users at short context are where 600 W helps:** 8 users at 1k get 807 tokens/s at
  600 W vs 692 at 480 W (+17%). At 8k and beyond the gain is 4–9%.
- **Decode barely depends on context** because 48 of the 64 layers are Gated-DeltaNet
  (linear attention), whose cost per token does not grow with context; only the 16
  full-attention layers read the (NVFP4) KV cache. The step time grows from 23.0 ms at 1k to
  30.0 ms at 258k (+30%); MTP then turns each step into 2.6–3.3 tokens.
- "All users together" = all output tokens divided by the time during which at least one user
  was receiving tokens. Per-user speed drops at long context because the engine mixes other
  users' prompt reading into the decode steps. The number of users at once is limited by KV
  memory: min(max-num-seqs, KV pool / (context + output)).

### More than 8 users (480 W, high-concurrency profile)

The config above allows 8 requests at once and the full 256K context. To see whether more
users add throughput we ran a second profile at 480 W: `--max-num-seqs 32`,
`--max-model-len 139264` (136K), KV pool sized by vLLM (198,767 tokens),
`--gpu-memory-utilization 0.90` (`compose/docker-compose.high-concurrency.example.yaml`).
Why not 64 users or 256K here: every sequence reserves a Gated-DeltaNet recurrent state
(about 75 MB across the 48 GDN layers) up front, so 64 sequences cost 4.8 GB and 32 sequences
no longer leave room for one 256K request.

| context | users at once | per user: tokens/s | **all users together: tokens/s** |
|---|---|---|---|
| 1k | 8 | 111 | **777** |
| 1k | 16 | 93 | **752** |
| 1k | 32 | 85 | **633** |
| 8k | 8 | 73 | **358** |
| 8k | 16 | 51 | **316** |
| 8k | 23 | 48 | **307** |
| 16k | 11 | 34 | **113** |
| 32k | 5 | 53 | **83** |
| 64k | 3 | 41 | **51** |

More users do not add throughput: with MTP every step checks 4 draft tokens per user, so the
batch grows fast and past about 8 users the total goes down, not up. On this card and this
quant, 8 users is the sweet spot; more users only spread the same throughput thinner.

## What is in this repo

| folder | |
|---|---|
| `image/` | Dockerfile and patch stack for the RTX 5090 (NVFP4 KV cache on sm_120) — vendored from adrienbrault/qwen3.8-27b-rtx5090, see NOTICE |
| `patches/` | fixes mounted over vLLM v0.28.0: stop strings inside thinking, leading newlines in answers, JSON quote truncation, INT8 embeddings, backport of vllm#48200 — [patches/README.md](patches/README.md) |
| `compose/` | docker compose for one GPU, health check, `.env.example`, high-concurrency example |
| `quantization/` | the whole pipeline: calibration set, GPTQ recipe, sensitivity scan, bit allocation, INT8 embedding, evaluation, benchmarks — [quantization/README.md](quantization/README.md) |
| `results/` | evaluation scores, sensitivity map, thinking test, benchmark data |

## Settings we learned the hard way

- `--gpu-memory-utilization 0.94` with a fixed KV budget passed our stress tests but ran out
  of memory under real traffic with mixed prompt lengths. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  fixes it without losing KV memory.
- Do not set `"disable_any_whitespace": true` for llguidance. It puts JSON delimiters inside
  string values.
- `presence_penalty=1.5` (the Qwen instruct recommendation) breaks repeated tool calls: the
  model leaves arguments empty. Use 0.
- `prompt_logprobs` with top-k is wrong while MTP is on (vllm#53488). Turn MTP off when scoring.

## License

Apache-2.0. See LICENSE and NOTICE for the third-party parts.
