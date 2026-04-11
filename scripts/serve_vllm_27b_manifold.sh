#!/bin/bash
#SBATCH --job-name=vllm-27b-manifold
#SBATCH --partition=capstone
#SBATCH --nodelist=manifold
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/vllm_27b_manifold_%j.log

# Serve Qwen3.5-27B-AWQ on manifold (RTX 5090, 32GB) at port 8124.
# Substitute for bourbaki (down). Same GPU class (RTX 5090, 32GB).
# AWQ 4-bit + enforce-eager + max-model-len 4096 required to fit on 32GB.
# NOTE: capstone partition has 8h time limit.

source /users/achidamb/projects/gepa-mutations/vllm-venv/bin/activate

# vLLM v1 creates ZMQ IPC sockets named by UUID in the CWD.
# The project path is too long (>107 chars). cd /tmp to avoid the limit.
cd /tmp

HF_HOME=/models/huggingface python -m vllm.entrypoints.openai.api_server \
  --model QuantTrio/Qwen3.5-27B-AWQ \
  --quantization awq \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8124 \
  --no-enable-log-requests
