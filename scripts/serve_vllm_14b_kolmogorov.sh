#!/bin/bash
#SBATCH --job-name=vllm-14b-kolmogorov
#SBATCH --partition=student-gpu
#SBATCH --nodelist=kolmogorov
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/vllm_14b_kolmogorov_%j.log

# Serve Qwen3-14B on kolmogorov (RTX 4090, 24GB) at port 8128.
# Uses fp8 quantization (~14GB weights) to fit within 24GB VRAM.
# student-gpu partition has 2h limit — chain jobs via sbatch --dependency.

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
  --port 8128 \
  --no-enable-log-requests
