#!/bin/bash
#SBATCH --job-name=vllm-1b-sar
#SBATCH --partition=capstone
#SBATCH --nodelist=sar-gpu-vm
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/vllm_1b_sar_%j.log

# Serve Qwen3-1.7B on sar-gpu-vm (RTX 3090, 24GB) at port 8127.
# Substitute for deepseek (down). Same GPU spec (RTX 3090, 24GB).
# NOTE: capstone partition has 8h time limit.

source /users/achidamb/projects/gepa-mutations/vllm-venv/bin/activate

# vLLM v1 creates ZMQ IPC sockets named by UUID in the CWD.
# The project path is too long (>107 chars). cd /tmp to avoid the limit.
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
