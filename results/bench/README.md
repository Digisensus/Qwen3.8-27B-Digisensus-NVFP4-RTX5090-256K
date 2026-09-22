# Speed benchmarks — Qwen3.8-27B-Digisensus on one RTX 5090

Hardware: one RTX 5090 32 GB. Prefill was measured at a 600 W power limit; decode at both
480 W and 600 W (`telemetry-480W.csv`, `telemetry-600W.csv`). Software: vLLM v0.28.0 with the image and patches in this repo, using
exactly `compose/docker-compose.yaml`: `--max-model-len 262144`, NVFP4 KV cache (266,197-token
pool), MTP k=3, prefix caching on, `--max-num-seqs 8`, `--max-num-batched-tokens 8192`.
Nothing else was running on the GPU. Measured 2026-09-21.

## Prefill and time to first token (one request at a time)

Random-token prompts (each one unique, so the prefix cache never helps), 512 output tokens,
measured with `vllm bench serve` (`quantization/bench.sh`; the raw JSON is in this folder).

| context | prefill tokens/s | time to first token |
|---|---|---|
| 1k | 4,475 | 0.23 s |
| 8k | 8,517 | 0.96 s |
| 16k | 10,703 | 1.53 s |
| 32k | 6,494 | 5.05 s |
| 64k | 4,986 | 13.14 s |
| 128k | 3,347 | 39.16 s |
| 261,600 (max) | 1,661 | 157.54 s |

The full 262,144-token context works on one card: a 261,600-token prompt is read in about
2.6 minutes and then answered at about 102 tokens/s (table below).

## Decode speed on real text — 480 W vs 600 W

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

## More than 8 users (480 W, high-concurrency profile)

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
