#!/bin/bash
# iso_health_check.sh — Verify all ISO experiment endpoints and resources
set -euo pipefail

TASK_PORT=${TASK_PORT:-8000}
REFL_PORT=${REFL_PORT:-8001}
MLFLOW_PORT=${MLFLOW_PORT:-5000}
DISK_PATH=${DISK_PATH:-/workspace}
MIN_FREE_GB=${MIN_FREE_GB:-50}

PASS=0
FAIL=0

check() {
    local name="$1"
    local result="$2"
    if [ "$result" -eq 0 ]; then
        echo "  ✓ $name"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $name"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== ISO Health Check ==="
echo ""

# 1. Task server health
echo "[Servers]"
curl -sf "http://localhost:$TASK_PORT/health" > /dev/null 2>&1
check "Task server (port $TASK_PORT)" $?

curl -sf "http://localhost:$REFL_PORT/health" > /dev/null 2>&1
check "Reflection server (port $REFL_PORT)" $?

# 2. Model verification
TASK_MODELS=$(curl -sf "http://localhost:$TASK_PORT/v1/models" 2>/dev/null || echo "")
if echo "$TASK_MODELS" | grep -q "Qwen3-8B"; then
    check "Task model ID (Qwen3-8B)" 0
else
    check "Task model ID (Qwen3-8B)" 1
fi

REFL_MODELS=$(curl -sf "http://localhost:$REFL_PORT/v1/models" 2>/dev/null || echo "")
if echo "$REFL_MODELS" | grep -q "Qwen3-32B"; then
    check "Reflection model ID (Qwen3-32B)" 0
else
    check "Reflection model ID (Qwen3-32B)" 1
fi

# 3. MLflow UI
echo ""
echo "[MLflow]"
curl -sf "http://localhost:$MLFLOW_PORT" > /dev/null 2>&1
check "MLflow UI (port $MLFLOW_PORT)" $?

# 4. GPU
echo ""
echo "[GPU]"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader > /dev/null 2>&1
    check "nvidia-smi" $?
else
    check "nvidia-smi" 1
fi

# 5. Disk space
echo ""
echo "[Disk]"
if [ -d "$DISK_PATH" ]; then
    FREE_KB=$(df -k "$DISK_PATH" 2>/dev/null | tail -1 | awk '{print $4}')
    FREE_GB=$((FREE_KB / 1024 / 1024))
    if [ "$FREE_GB" -ge "$MIN_FREE_GB" ]; then
        check "Disk space (${FREE_GB}GB free >= ${MIN_FREE_GB}GB min)" 0
    else
        check "Disk space (${FREE_GB}GB free < ${MIN_FREE_GB}GB min)" 1
    fi
else
    echo "  - Disk path $DISK_PATH not found (expected on RunPod)"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
