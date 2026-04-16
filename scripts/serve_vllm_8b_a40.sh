#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8125}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
LOG_FILE="${LOG_FILE:-/tmp/vllm_8b_a40.log}"

export HF_HOME="${HF_HOME:-/models/huggingface}"
mkdir -p "$HF_HOME"

echo "Starting vLLM for $MODEL on GPU $GPU, port $PORT"
echo "  dtype: bfloat16"
echo "  max_model_len: $MAX_MODEL_LEN"
echo "  gpu_memory_utilization: $GPU_MEM_UTIL"
echo "  HF_HOME: $HF_HOME"
echo "  log: $LOG_FILE"
echo "  endpoint: http://localhost:$PORT/v1"

cd /tmp  # Avoid IPC socket path length bug (see known_bugs_and_fixes.md #3)

CUDA_VISIBLE_DEVICES="$GPU" python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --dtype bfloat16 \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --served-model-name "$MODEL" \
    2>&1 | tee -a "$LOG_FILE"
