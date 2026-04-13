#!/bin/bash
#SBATCH --job-name=vllm-14b-ansatz
#SBATCH --partition=ray-cluster
#SBATCH --nodelist=ansatz
#SBATCH --gres=gpu:1
#SBATCH --time=90-00:00:00
#SBATCH --output=logs/vllm_14b_ansatz_%j.log

# Serve Qwen3-14B on ansatz (ray-cluster) at port 8129.
# Uses fp8 quantization (~14GB weights) to fit within 24GB VRAM.

source /users/achidamb/projects/gepa-mutations/vllm-venv/bin/activate

cd /tmp

HF_HOME=/models/huggingface python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B \
  --dtype auto \
  --quantization fp8 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8129 \
  --no-enable-log-requests
