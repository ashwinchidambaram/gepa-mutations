#!/bin/bash
# Serve Gemma 3 27B on Mac via MLX-LM at port 8136.
# Memory: ~15GB. Direct cross-architecture comparison with cluster's Qwen3-27B-AWQ.
# Run alone or alongside small models only (combined budget ~42GB with Qwen3-32B).
#
# Usage: bash scripts/serve_mlx_gemma3_27b_mac.sh

python -m mlx_lm.server \
  --model mlx-community/gemma-3-27b-it-4bit \
  --port 8136 \
  --host 127.0.0.1
