#!/bin/bash
# iso_run_pilot.sh — Phase 0 pilot run driver
# Usage: ./scripts/iso_run_pilot.sh [--smoke-test] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

CONFIG="configs/pilot.yaml"
SMOKE_TEST=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --smoke-test) SMOKE_TEST=true ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "=== ISO Pilot Run ==="
echo "Config: $CONFIG"
echo "Smoke test: $SMOKE_TEST"
echo "Dry run: $DRY_RUN"
echo ""

# Health check first
echo "Running health check..."
if ! bash "$SCRIPT_DIR/iso_health_check.sh"; then
    echo ""
    echo "ERROR: Health check failed. Fix issues before running pilot."
    exit 1
fi
echo ""

# Build Python command
CMD="python -m iso_harness.experiment.orchestrator"
CMD="$CMD --config $CONFIG"

if $SMOKE_TEST; then
    CMD="$CMD --smoke-test"
fi

if $DRY_RUN; then
    CMD="$CMD --dry-run"
fi

echo "Running: $CMD"
echo ""
exec $CMD
