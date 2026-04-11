#!/bin/bash
# Serve Gemma 3 12B on Mac via MLX-LM at port 8135.
# Memory: ~7GB. Mid-size Gemma 3 — unique scale not covered by cluster.
#
# Usage: bash scripts/serve_mlx_gemma3_12b_mac.sh

python -m mlx_lm.server \
  --model mlx-community/gemma-3-12b-it-4bit \
  --port 8135 \
  --host 127.0.0.1
