#!/bin/bash
#SBATCH --job-name=vllm-4b-mandelbrot
#SBATCH --partition=student-gpu
#SBATCH --nodelist=mandelbrot
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/vllm_4b_mandelbrot_%j.log

# Serve Qwen3-4B on mandelbrot (RTX 4060 Ti, 16GB) at port 8129.
# Second 4B worker — runs experiments in parallel with sapphire (port 8126).
# student-gpu partition has 2h limit — chain jobs via sbatch --dependency.

source /users/achidamb/projects/gepa-mutations/vllm-venv/bin/activate

cd /tmp

HF_HOME=/models/huggingface python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --dtype auto \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8129 \
  --no-enable-log-requests
