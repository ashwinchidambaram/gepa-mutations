#!/bin/bash
# Serve Qwen3-32B on Mac via MLX-LM at port 8131.
# Run directly — no SLURM needed.
# Memory: ~20GB (4-bit). Takes ~30s to load. Keep workers <= 3.
#
# Usage: bash scripts/serve_mlx_32b_mac.sh

.venv/bin/python -m mlx_lm.server \
  --model mlx-community/Qwen3-32B-4bit \
  --port 8131 \
  --host 127.0.0.1
