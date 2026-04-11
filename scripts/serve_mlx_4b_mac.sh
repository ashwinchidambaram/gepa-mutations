#!/bin/bash
# Serve Qwen3-4B on Mac via MLX-LM at port 8126.
# Run directly — no SLURM needed.
# Memory: ~3GB. Supports 5-7 concurrent workers.
#
# Usage: bash scripts/serve_mlx_4b_mac.sh

python -m mlx_lm.server \
  --model mlx-community/Qwen3-4B-4bit \
  --port 8126 \
  --host 127.0.0.1
