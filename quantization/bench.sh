#!/bin/bash
# Publishable serving benchmark: Qwen3.8-27B-Digisensus on ONE RTX 5090, exact prod config
# (nvfp4 KV, MTP k=3, patches, prefix caching on), engine off the router on :${PORT}.
# Per context length: single stream (PP = input/TTFT, decode = 1/TPOT) and one concurrent
# wave (concurrency = min(8, KV pool / (ctx+out))). Random-token prompts: unique per request,
# so the prefix cache never hits. GPU power/clocks logged alongside.
set -u
PORT=${PORT:-9003}; GPU_IDX=${GPU_IDX:-2}; OUT=${OUT:-512}; POOL=${POOL:-266197}
IMG=vllm-qwen38:v0280-nvfp4kv; MODEL=${MODEL:-$PWD/models/Qwen3.8-27B-Digisensus-NVFP4-RTX5090-256K}
RES=${RES:-$PWD/results/bench}; mkdir -p "$RES"
nvidia-smi -i "$GPU_IDX" --query-gpu=timestamp,power.draw,power.limit,clocks.sm,temperature.gpu,memory.used \
  --format=csv,noheader -lms 1000 > "$RES/gpu-telemetry.csv" & TEL=$!
trap 'kill $TEL 2>/dev/null' EXIT
run() {  # name input_len concurrency num_prompts
  docker run --rm --network host -v "$MODEL":/models/model:ro -v "$RES":/res --entrypoint vllm "$IMG" bench serve \
    --base-url "http://localhost:$PORT" --model qwen --tokenizer /models/model \
    --dataset-name random --random-input-len "$2" --random-output-len "$OUT" --random-range-ratio 0 \
    --ignore-eos --max-concurrency "$3" --num-prompts "$4" --seed 7 \
    --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir /res --result-filename "$1.json" 2>&1 | grep -E "Successful|Mean TTFT|Mean TPOT|Output token throughput|Total token throughput" | sed "s/^/  [$1] /"
}
echo "warmup"; run warmup 1024 1 2 >/dev/null
for L in 1024 8192 16384 32768 65536 131072 261600; do
  C=$(( POOL / (L + OUT) )); [ $C -gt 8 ] && C=8; [ $C -lt 1 ] && C=1
  echo "== ctx $L: single"; run "single-$L" "$L" 1 2
  [ $C -gt 1 ] && { echo "== ctx $L: concurrent x$C"; run "conc$C-$L" "$L" "$C" "$C"; }
done
echo DONE
