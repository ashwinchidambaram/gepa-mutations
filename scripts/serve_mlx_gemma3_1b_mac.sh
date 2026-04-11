#!/bin/bash
# Serve Gemma 3 1B on Mac via MLX-LM at port 8133.
# Memory: ~0.6GB. Smallest Gemma 3 — good for cross-arch baseline at tiny scale.
#
# Usage: bash scripts/serve_mlx_gemma3_1b_mac.sh

.venv/bin/python -m mlx_lm.server \
  --model mlx-community/gemma-3-1b-it-4bit \
  --port 8133 \
  --host 127.0.0.1
