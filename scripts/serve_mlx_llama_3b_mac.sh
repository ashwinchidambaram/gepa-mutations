#!/bin/bash
# Serve Llama 3.2 3B on Mac via MLX-LM at port 8138.
# Memory: ~2GB. Mid-small Llama — unique scale between 1B and 8B.
#
# Usage: bash scripts/serve_mlx_llama_3b_mac.sh

.venv/bin/python -m mlx_lm.server \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --port 8138 \
  --host 127.0.0.1
