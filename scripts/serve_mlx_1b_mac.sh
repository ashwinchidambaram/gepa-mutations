#!/bin/bash
# Serve Qwen3-1.7B on Mac via MLX-LM at port 8125.
# Run directly — no SLURM needed.
# Memory: ~1GB. Fast model, supports 6-8 concurrent workers.
#
# Usage: bash scripts/serve_mlx_1b_mac.sh

python -m mlx_lm.server \
  --model mlx-community/Qwen3-1.7B-4bit \
  --port 8125 \
  --host 127.0.0.1
