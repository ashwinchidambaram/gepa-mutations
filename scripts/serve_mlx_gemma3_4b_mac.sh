#!/bin/bash
# Serve Gemma 3 4B on Mac via MLX-LM at port 8134.
# Memory: ~2.5GB. Cross-architecture comparison at the 4B scale.
#
# Usage: bash scripts/serve_mlx_gemma3_4b_mac.sh

python -m mlx_lm.server \
  --model mlx-community/gemma-3-4b-it-4bit \
  --port 8134 \
  --host 127.0.0.1
