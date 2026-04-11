#!/bin/bash
# Serve Llama 3.2 1B on Mac via MLX-LM at port 8137.
# Memory: ~0.6GB. Third architecture at the 1B scale alongside Qwen3-0.6B and Gemma 3 1B.
#
# Usage: bash scripts/serve_mlx_llama_1b_mac.sh

.venv/bin/python -m mlx_lm.server \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --port 8137 \
  --host 127.0.0.1
