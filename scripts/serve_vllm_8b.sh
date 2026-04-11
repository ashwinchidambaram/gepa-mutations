#!/bin/bash
#SBATCH --job-name=vllm-8b
#SBATCH --partition=ray-cluster
#SBATCH --nodelist=archimedes
#SBATCH --gres=gpu:1
#SBATCH --time=90-00:00:00
#SBATCH --output=logs/vllm_8b_%j.log

# Serve Qwen3-8B on archimedes (RTX 4090, 24GB) at port 8125.
# Qwen3-8B in bf16: ~16GB weights + ~4GB KV cache — fits at 0.90 utilization.

source /users/achidamb/projects/gepa-mutations/vllm-venv/bin/activate

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8125 \
  --no-enable-log-requests
