#!/bin/bash
#SBATCH --job-name=vllm-4b-sapphire
#SBATCH --partition=capstone
#SBATCH --nodelist=sapphire
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/vllm_4b_sapphire_%j.log

# Serve Qwen3-4B on sapphire (RTX 4090, 24GB) at port 8126.
# Substitute for ansatz (down). Clean GPU — no foreign process occupying memory,
# so using 0.85 utilization (vs 0.65 on ansatz which had ~7.3GB occupied).
# NOTE: capstone partition has 8h time limit.

source /users/achidamb/projects/gepa-mutations/vllm-venv/bin/activate

# vLLM v1 creates ZMQ IPC sockets named by UUID in the CWD.
# The project path is too long (>107 chars). cd /tmp to avoid the limit.
cd /tmp

HF_HOME=/models/huggingface python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8126 \
  --no-enable-log-requests
