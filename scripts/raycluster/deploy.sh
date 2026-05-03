#!/usr/bin/env bash
# Deploy only the necessary parts of gepa-mutations to gho-vm-2.
#
# Syncs a minimal subset: benchmark loaders, evaluators, raycluster scripts,
# and a lightweight pyproject.toml. Does NOT clone the full repo.
#
# Prerequisites:
#   - WireGuard VPN active
#   - SSH key at ~/.ssh/id_ed25519
#
# Usage:
#   bash scripts/raycluster/deploy.sh          # Full deploy
#   bash scripts/raycluster/deploy.sh --sync   # Re-sync code only (skip env setup)

set -euo pipefail

VM_HOST="10.0.50.65"
VM_USER="achidamb"
REMOTE_DIR="local-projects/gepa-mutations-cluster"
SSH_CMD="ssh -o StrictHostKeyChecking=accept-new ${VM_USER}@${VM_HOST}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Raycluster Deploy (lightweight) ==="
echo "Local:  ${REPO_ROOT}"
echo "Remote: ${VM_USER}@${VM_HOST}:~/${REMOTE_DIR}"
echo ""

# --- Check connectivity ---
echo "[1/4] Checking connectivity..."
if ! ping -c 1 -W 3 ${VM_HOST} > /dev/null 2>&1; then
    echo "FAIL: Cannot reach ${VM_HOST}. Is WireGuard VPN active?"
    exit 1
fi
echo "  OK — VM reachable"

# --- Create remote directory structure ---
echo "[2/4] Creating remote directory structure..."
${SSH_CMD} "mkdir -p ~/${REMOTE_DIR}/{src,scripts/raycluster,runs/qwen3.5-27b}"

# --- Sync only what we need ---
echo "[3/4] Syncing code..."

RSYNC_OPTS="-avz --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv'"

# Core benchmark infrastructure (loaders + evaluators)
echo "  syncing src/gepa_mutations/benchmarks/..."
rsync ${RSYNC_OPTS} \
    "${REPO_ROOT}/src/gepa_mutations/benchmarks/" \
    "${VM_USER}@${VM_HOST}:~/${REMOTE_DIR}/src/gepa_mutations/benchmarks/"

# Package init files (needed for imports)
rsync -avz \
    "${REPO_ROOT}/src/gepa_mutations/__init__.py" \
    "${VM_USER}@${VM_HOST}:~/${REMOTE_DIR}/src/gepa_mutations/__init__.py"

# Metrics (needed by evaluators)
echo "  syncing src/gepa_mutations/metrics/..."
rsync ${RSYNC_OPTS} \
    "${REPO_ROOT}/src/gepa_mutations/metrics/" \
    "${VM_USER}@${VM_HOST}:~/${REMOTE_DIR}/src/gepa_mutations/metrics/"

# Raycluster scripts (includes stubs/)
echo "  syncing scripts/raycluster/..."
rsync ${RSYNC_OPTS} \
    "${REPO_ROOT}/scripts/raycluster/" \
    "${VM_USER}@${VM_HOST}:~/${REMOTE_DIR}/scripts/raycluster/"

# Gepa stub (minimal types needed by evaluators)
echo "  syncing gepa stub..."
rsync ${RSYNC_OPTS} \
    "${REPO_ROOT}/scripts/raycluster/stubs/gepa/" \
    "${VM_USER}@${VM_HOST}:~/${REMOTE_DIR}/src/gepa/"

# Runs README
rsync -avz \
    "${REPO_ROOT}/runs/README.md" \
    "${VM_USER}@${VM_HOST}:~/${REMOTE_DIR}/runs/README.md"

# Lightweight pyproject.toml for the cluster (minimal deps)
echo "  deploying cluster pyproject.toml..."
${SSH_CMD} "cat > ~/${REMOTE_DIR}/pyproject.toml" << 'PYPROJECT'
[project]
name = "gepa-mutations-cluster"
version = "0.1.0"
description = "Lightweight deployment for raycluster experiments"
requires-python = ">=3.11"
dependencies = [
    "dspy",
    "datasets",
    "requests",
    "numpy",
    "rich",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/gepa_mutations"]
PYPROJECT

echo "  Sync complete."

# --- Environment setup (skip with --sync) ---
if [[ "${1:-}" == "--sync" ]]; then
    echo "[4/4] Skipped (--sync mode)"
    echo ""
    echo "=== Sync Complete ==="
    exit 0
fi

echo "[4/4] Setting up Python environment on cluster..."
${SSH_CMD} << SETUP_SCRIPT
set -euo pipefail
cd ~/${REMOTE_DIR}

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="\$HOME/.local/bin:\$PATH"
fi

# Create venv and install deps
echo "  Running uv sync..."
uv sync 2>&1 | tail -10

echo ""
echo "  Python env ready:"
uv run python --version

# Quick import test
echo "  Testing imports..."
uv run python -c "from gepa_mutations.benchmarks.loader import load_benchmark; print('  OK — benchmarks importable')"
SETUP_SCRIPT

echo ""
echo "=== Deploy Complete ==="
echo ""
echo "Next steps:"
echo "  1. Test connectivity:  ${SSH_CMD} 'cd ~/${REMOTE_DIR} && uv run python scripts/raycluster/test_connectivity.py'"
echo "  2. Run baseline:       ${SSH_CMD} 'cd ~/${REMOTE_DIR} && nohup uv run python scripts/raycluster/run_baseline.py > baseline.log 2>&1 &'"
echo "  3. Check progress:     ${SSH_CMD} 'tail -f ~/${REMOTE_DIR}/baseline.log'"
echo "  4. Copy results back:  scp -r ${VM_USER}@${VM_HOST}:~/${REMOTE_DIR}/runs/ ${REPO_ROOT}/runs/"
