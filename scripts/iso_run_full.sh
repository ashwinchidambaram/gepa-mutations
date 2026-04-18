#!/bin/bash
# iso_run_full.sh — Phase 1 full experiment driver
# Usage: ./scripts/iso_run_full.sh [--config PATH] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

CONFIG="configs/full.yaml"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=== ISO Full Experiment ==="
echo "Config: $CONFIG"
echo "Dry run: $DRY_RUN"
echo ""

# Strict git check: refuse dirty tree
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "ERROR: Git working tree is dirty."
    echo "Commit all changes before starting full experiment runs."
    echo ""
    git status --short
    exit 1
fi

GIT_SHA=$(git rev-parse HEAD)
echo "Git SHA: $GIT_SHA"
echo ""

# Health check
echo "Running health check..."
if ! bash "$SCRIPT_DIR/iso_health_check.sh"; then
    echo ""
    echo "ERROR: Health check failed. Fix issues before running."
    exit 1
fi
echo ""

# Build Python command
CMD="python -m iso_harness.experiment.orchestrator"
CMD="$CMD --config $CONFIG"
CMD="$CMD --strict-git"

if $DRY_RUN; then
    CMD="$CMD --dry-run"
fi

echo "Running: $CMD"
echo ""
exec $CMD
