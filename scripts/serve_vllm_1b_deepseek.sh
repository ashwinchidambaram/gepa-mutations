#!/bin/bash
#SBATCH --job-name=vllm-1b
#SBATCH --partition=ray-cluster
#SBATCH --nodelist=deepseek
#SBATCH --gres=gpu:1
#SBATCH --time=90-00:00:00
#SBATCH --output=logs/vllm_1b_deepseek_%j.log

# Serve Qwen3-1.7B on deepseek (ray-cluster) at port 8127.

source /users/achidamb/projects/gepa-mutations/vllm-venv/bin/activate

cd /tmp

HF_HOME=/models/huggingface python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --dtype auto \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8127 \
  --no-enable-log-requests
