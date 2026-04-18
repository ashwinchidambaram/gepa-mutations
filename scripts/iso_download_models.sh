#!/bin/bash
# iso_download_models.sh — Download and cache model weights on Network Volume
# Downloads Qwen3-8B (bf16) and Qwen3-32B-AWQ to the HuggingFace cache.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

# Default cache dir (RunPod persistent volume)
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
echo "=== ISO Model Download ==="
echo "HuggingFace cache: $HF_CACHE"
echo ""

# Models to download
TASK_MODEL="Qwen/Qwen3-8B"
REFLECTION_MODEL="Qwen/Qwen3-32B-AWQ"

echo "[1/2] Downloading task model: $TASK_MODEL"
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('$TASK_MODEL')
print(f'  Downloaded to: {path}')
"

echo ""
echo "[2/2] Downloading reflection model: $REFLECTION_MODEL"
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('$REFLECTION_MODEL')
print(f'  Downloaded to: {path}')
"

echo ""
echo "=== Model Download Complete ==="

# Print model commit SHAs for config pinning
echo ""
echo "Model commit SHAs (pin these in configs/*.yaml):"
python -c "
from huggingface_hub import model_info
for model_id in ['$TASK_MODEL', '$REFLECTION_MODEL']:
    info = model_info(model_id)
    print(f'  {model_id}: {info.sha}')
"
