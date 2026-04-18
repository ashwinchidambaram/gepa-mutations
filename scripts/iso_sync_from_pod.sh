#!/bin/bash
# iso_sync_from_pod.sh — LOCAL-machine script: rsync from RunPod + git commit
# Usage: ./scripts/iso_sync_from_pod.sh [--dry-run]
#
# Configure via .env (not committed):
#   POD_SSH_TARGET=root@<pod-ip>
#   POD_DATA_DIR=/workspace/iso-experiment
#   LOCAL_DATA_DIR=~/iso-experiment-data
#   SSH_KEY=~/.ssh/runpod_key
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Load config from .env
if [ -f ".env" ]; then
    # shellcheck disable=SC1091
    set -a; source .env; set +a
fi

POD_SSH_TARGET="${POD_SSH_TARGET:-}"
POD_DATA_DIR="${POD_DATA_DIR:-/workspace/iso-experiment}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-$HOME/iso-experiment-data}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/runpod_key}"

if [ -z "$POD_SSH_TARGET" ]; then
    echo "ERROR: POD_SSH_TARGET not set. Configure in .env or environment."
    echo "Example: POD_SSH_TARGET=root@203.0.113.5"
    exit 1
fi

echo "=== ISO Data Sync ==="
echo "Pod: $POD_SSH_TARGET:$POD_DATA_DIR"
echo "Local: $LOCAL_DATA_DIR"
echo "Dry run: $DRY_RUN"
echo ""

SSH_CMD="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"

# Step 1: Snapshot MLflow database on pod (safe copy)
echo "[1/4] Snapshotting MLflow database on pod..."
if ! $DRY_RUN; then
    $SSH_CMD "$POD_SSH_TARGET" \
        "sqlite3 $POD_DATA_DIR/mlflow/mlflow.db '.backup $POD_DATA_DIR/mlflow/mlflow_snapshot.db'" \
        2>/dev/null || echo "  (MLflow snapshot skipped — database may not exist yet)"
fi

# Step 2: rsync
echo "[2/4] Syncing data..."
mkdir -p "$LOCAL_DATA_DIR"

RSYNC_ARGS=(
    -avz --partial --progress
    --exclude '*.tmp'
    --exclude 'mlflow.db-journal'
    --exclude 'mlflow.db-wal'
    --exclude 'mlflow.db'  # Use snapshot instead
    -e "$SSH_CMD"
    "$POD_SSH_TARGET:$POD_DATA_DIR/"
    "$LOCAL_DATA_DIR/"
)

if $DRY_RUN; then
    RSYNC_ARGS=(--dry-run "${RSYNC_ARGS[@]}")
fi

rsync "${RSYNC_ARGS[@]}"

# Step 3: Detect newly completed runs
echo ""
echo "[3/4] Checking for newly completed runs..."
NEW_RUNS=0
RESULTS_DIR="$PROJECT_DIR/results"
mkdir -p "$RESULTS_DIR"

if [ -d "$LOCAL_DATA_DIR/runs" ]; then
    for run_dir in "$LOCAL_DATA_DIR"/runs/*/; do
        [ -d "$run_dir" ] || continue
        if [ -f "$run_dir/COMPLETE" ] && [ -f "$run_dir/summary.json" ]; then
            run_id=$(basename "$run_dir")
            dest="$RESULTS_DIR/$run_id"
            if [ ! -d "$dest" ]; then
                echo "  New completed run: $run_id"
                mkdir -p "$dest"
                # Copy result subset
                cp "$run_dir/summary.json" "$dest/" 2>/dev/null || true
                cp "$run_dir/report.md" "$dest/" 2>/dev/null || true
                cp "$run_dir/COMPLETE" "$dest/" 2>/dev/null || true
                NEW_RUNS=$((NEW_RUNS + 1))
            fi
        fi
    done
fi

echo "  $NEW_RUNS new completed runs found"

# Step 4: Git commit if new results
echo ""
echo "[4/4] Git commit..."
if [ "$NEW_RUNS" -gt 0 ] && ! $DRY_RUN; then
    cd "$PROJECT_DIR"
    git add results/
    if git diff --cached --quiet; then
        echo "  No new changes to commit"
    else
        git commit -m "results: sync $NEW_RUNS completed run(s) from pod"
        echo "  Committed $NEW_RUNS new run(s)"
    fi
else
    echo "  No new results to commit"
fi

echo ""
echo "=== Sync complete ==="
