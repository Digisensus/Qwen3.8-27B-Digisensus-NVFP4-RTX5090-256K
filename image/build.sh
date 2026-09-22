#!/bin/sh
# Builds the sm_120 (RTX 5090) vLLM v0.28.0 image with the NVFP4 KV-cache stack.
# Vendored from adrienbrault/qwen3.8-27b-rtx5090 @c694dbc (patches-v0280) — see ../NOTICE.
set -e
cd "$(dirname "$0")/patches-v0280"
docker build -f Dockerfile.v0280-nvfp4kv -t "${IMAGE:-digisensus/vllm-qwen38:v0280-sm120-nvfp4kv}" .
