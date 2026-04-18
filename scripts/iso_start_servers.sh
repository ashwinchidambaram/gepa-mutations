#!/bin/bash
# iso_start_servers.sh — Launch dual vLLM servers + MLflow UI
# Usage: ./scripts/iso_start_servers.sh [--phase pilot|full]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

PHASE="${1:---phase}"
PHASE_VALUE="${2:-pilot}"

# Parse --phase argument
if [[ "$PHASE" == "--phase" ]]; then
    PHASE_VALUE="${PHASE_VALUE}"
elif [[ "$1" == "pilot" || "$1" == "full" ]]; then
    PHASE_VALUE="$1"
fi

echo "=== Starting ISO Servers (phase=$PHASE_VALUE) ==="

# GPU memory splits per phase (from spec Sections 5.1 and 5.2)
if [[ "$PHASE_VALUE" == "pilot" ]]; then
    TASK_GPU_UTIL=0.42
    TASK_MAX_SEQ=4
    TASK_MAX_LEN=32768
    REFL_GPU_UTIL=0.50
    REFL_MAX_SEQ=1
    REFL_MAX_LEN=32768
elif [[ "$PHASE_VALUE" == "full" ]]; then
    TASK_GPU_UTIL=0.35
    TASK_MAX_SEQ=6
    TASK_MAX_LEN=32768
    REFL_GPU_UTIL=0.55
    REFL_MAX_SEQ=2
    REFL_MAX_LEN=49152
else
    echo "ERROR: Unknown phase '$PHASE_VALUE'. Use 'pilot' or 'full'."
    exit 1
fi

TASK_MODEL="Qwen/Qwen3-8B"
REFL_MODEL="Qwen/Qwen3-32B-AWQ"
TASK_PORT=8000
REFL_PORT=8001
MLFLOW_PORT=5000
MLFLOW_URI="file:///workspace/mlflow"

# Kill existing servers if running
echo "Stopping any existing servers..."
tmux kill-session -t iso-task 2>/dev/null || true
tmux kill-session -t iso-reflection 2>/dev/null || true
tmux kill-session -t iso-mlflow 2>/dev/null || true
sleep 1

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Start task model server
echo "[1/3] Starting task model ($TASK_MODEL) on port $TASK_PORT..."
tmux new-session -d -s iso-task "
    source $PROJECT_DIR/.venv/bin/activate 2>/dev/null;
    vllm serve $TASK_MODEL \
        --port $TASK_PORT \
        --dtype bfloat16 \
        --gpu-memory-utilization $TASK_GPU_UTIL \
        --max-model-len $TASK_MAX_LEN \
        --max-num-seqs $TASK_MAX_SEQ \
        --enable-prefix-caching \
        --disable-log-requests \
        --trust-remote-code \
        2>&1 | tee $PROJECT_DIR/logs/vllm_task.log
"

# Start reflection model server
echo "[2/3] Starting reflection model ($REFL_MODEL) on port $REFL_PORT..."
tmux new-session -d -s iso-reflection "
    source $PROJECT_DIR/.venv/bin/activate 2>/dev/null;
    vllm serve $REFL_MODEL \
        --port $REFL_PORT \
        --quantization awq \
        --gpu-memory-utilization $REFL_GPU_UTIL \
        --max-model-len $REFL_MAX_LEN \
        --max-num-seqs $REFL_MAX_SEQ \
        --enable-prefix-caching \
        --disable-log-requests \
        --trust-remote-code \
        2>&1 | tee $PROJECT_DIR/logs/vllm_reflection.log
"

# Start MLflow UI
echo "[3/3] Starting MLflow UI on port $MLFLOW_PORT..."
mkdir -p /workspace/mlflow
tmux new-session -d -s iso-mlflow "
    source $PROJECT_DIR/.venv/bin/activate 2>/dev/null;
    mlflow ui --backend-store-uri $MLFLOW_URI --port $MLFLOW_PORT --host 0.0.0.0 \
        2>&1 | tee $PROJECT_DIR/logs/mlflow.log
"

# Wait for health
echo ""
echo "Waiting for servers to be ready (timeout 5 min)..."
TIMEOUT=300
ELAPSED=0

while true; do
    TASK_OK=false
    REFL_OK=false

    if curl -sf "http://localhost:$TASK_PORT/health" > /dev/null 2>&1; then
        TASK_OK=true
    fi
    if curl -sf "http://localhost:$REFL_PORT/health" > /dev/null 2>&1; then
        REFL_OK=true
    fi

    if $TASK_OK && $REFL_OK; then
        echo ""
        echo "=== All servers ready ==="
        echo "  Task model:      http://localhost:$TASK_PORT"
        echo "  Reflection model: http://localhost:$REFL_PORT"
        echo "  MLflow UI:       http://localhost:$MLFLOW_PORT"
        echo ""
        echo "Inspect with:"
        echo "  tmux attach -t iso-task"
        echo "  tmux attach -t iso-reflection"
        echo "  tmux attach -t iso-mlflow"
        exit 0
    fi

    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo ""
        echo "ERROR: Servers not ready after ${TIMEOUT}s"
        $TASK_OK || echo "  Task server (port $TASK_PORT): NOT READY"
        $REFL_OK || echo "  Reflection server (port $REFL_PORT): NOT READY"
        echo "Check logs: tail -50 logs/vllm_task.log"
        exit 1
    fi

    sleep 5
    ELAPSED=$((ELAPSED + 5))
    printf "."
done
