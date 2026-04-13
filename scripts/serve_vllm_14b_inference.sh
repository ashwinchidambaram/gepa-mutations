#!/bin/bash
#SBATCH --job-name=vllm-14b-inference
#SBATCH --partition=inference
#SBATCH --nodelist=inference
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/vllm_14b_inference_%j.log

# Serve Qwen3-14B on inference node at port 8130.
# Uses fp8 quantization (~14GB weights) to fit within 24GB VRAM.

source /users/achidamb/projects/gepa-mutations/vllm-venv/bin/activate

cd /tmp

HF_HOME=/models/huggingface python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B \
  --dtype auto \
  --quantization fp8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8130 \
  --no-enable-log-requests
