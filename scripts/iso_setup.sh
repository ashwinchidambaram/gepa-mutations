#!/bin/bash
# iso_setup.sh — One-time RunPod pod setup for ISO experiments
# Installs uv, creates venv with Python 3.12, installs pinned deps,
# verifies critical imports and GPU visibility.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== ISO Experiment Setup ==="
echo "Project dir: $PROJECT_DIR"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Step 1: Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[1/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env" 2>/dev/null || true
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
else
    echo "[1/5] uv already installed: $(uv --version)"
fi

# Step 2: Create virtual environment
echo "[2/5] Creating virtual environment with Python 3.12..."
if [ ! -d ".venv" ]; then
    uv venv --python 3.12 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Step 3: Install dependencies
echo "[3/5] Installing dependencies..."
if [ -f "requirements.txt" ]; then
    echo "  Using pinned requirements.txt"
    uv pip install -r requirements.txt
else
    echo "  No requirements.txt found; using pyproject.toml"
    uv sync
fi

# Step 4: Verify critical imports
echo "[4/5] Verifying Python imports..."
python -c "
import torch
import vllm
import dspy
import mlflow
import pyarrow
import pydantic
import yaml
import duckdb
print(f'  torch={torch.__version__} (CUDA={torch.cuda.is_available()})')
print(f'  vllm={vllm.__version__}')
print(f'  dspy={dspy.__version__}')
print(f'  mlflow={mlflow.__version__}')
print(f'  pyarrow={pyarrow.__version__}')
print(f'  duckdb={duckdb.__version__}')
"

# Step 5: Verify GPU visibility
echo "[5/5] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available to PyTorch!'
print(f'  PyTorch sees {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}')
"
else
    echo "  WARNING: nvidia-smi not found (expected on RunPod GPU pods)"
fi

# Step 6: Generate requirements.txt for portability (if not present)
if [ ! -f "requirements.txt" ]; then
    echo ""
    echo "Generating requirements.txt for portability..."
    uv pip compile pyproject.toml -o requirements.txt 2>/dev/null || echo "  (skipped: uv pip compile not available)"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
