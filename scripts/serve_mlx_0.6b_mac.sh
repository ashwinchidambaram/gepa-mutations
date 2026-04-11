#!/bin/bash
# Serve Qwen3-0.6B on Mac via MLX-LM at port 8132.
# Memory: ~0.4GB. Extends the Qwen3 scaling curve to sub-1B.
#
# Usage: bash scripts/serve_mlx_0.6b_mac.sh

.venv/bin/python -m mlx_lm.server \
  --model mlx-community/Qwen3-0.6B-4bit \
  --port 8132 \
  --host 127.0.0.1
