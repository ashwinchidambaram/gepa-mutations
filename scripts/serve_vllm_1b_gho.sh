#!/bin/bash
#SBATCH --job-name=vllm-1b-gho
#SBATCH --partition=student-gpu
#SBATCH --nodelist=gho-gpu-vm
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/vllm_1b_gho_%j.log

# Serve Qwen3-1.7B on gho-gpu-vm (RTX 4060 Ti, 16GB) at port 8130.
# Second 1.7B worker — runs experiments in parallel with sar-gpu-vm (port 8127).
# student-gpu partition has 2h limit — chain jobs via sbatch --dependency.

source /users/achidamb/projects/gepa-mutations/vllm-venv/bin/activate

cd /tmp

HF_HOME=/models/huggingface python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --dtype auto \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8130 \
  --no-enable-log-requests
