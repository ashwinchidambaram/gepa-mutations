#!/bin/bash
# Pod B: RTX 5000 Ada (32GB) — runs Qwen3-8B
# Usage:
#   bash scripts/runpod_pod_b.sh          # Full sweep
#   bash scripts/runpod_pod_b.sh smoke    # Smoke test only

set -euo pipefail

PHASE="${1:-1}"
BENCHMARKS="hotpotqa hover pupa ifbench livebench aime"
METHODS="gepa contrastive_reflection synaptic_pruning slime_mold tournament"
BRANCH="runpod/pod-b"

echo "=== RunPod Pod B — Qwen3-8B ==="

# --- GPU Info ---
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "GPU: single RTX 5000 Ada (CUDA_VISIBLE_DEVICES=0)"

# --- Ensure we're on the correct branch ---
cd /workspace/gepa-mutations
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
echo "Branch: $(git branch --show-current)"

# --- Check existing results ---
echo ""
echo "--- Results so far ---"
python3 -c "
import glob, collections
c = collections.Counter()
for f in glob.glob('runs/qwen3-*/*/*/*/result.json'):
    c[f.split('/')[1]] += 1
for k, v in sorted(c.items()):
    print(f'  {k}: {v}')
print(f'  Total: {sum(c.values())}')
" 2>/dev/null || echo "  (no results yet)"

# --- Helper: git checkpoint ---
git_checkpoint() {
    cd /workspace/gepa-mutations
    git add runs/ logs/ 2>/dev/null || true
    git commit -m "data: checkpoint pod-b $(date '+%Y-%m-%d %H:%M')" 2>/dev/null || true
    git push origin "$BRANCH" 2>/dev/null || true
}

# --- Helper: start vLLM if not already running ---
start_vllm() {
    local gpu=$1 model=$2 port=$3 max_len=$4 label=$5

    if curl -sf http://localhost:$port/v1/models > /dev/null 2>&1; then
        echo "  $label: already running on :$port"
        return
    fi

    echo "  $label: starting on GPU $gpu, port $port, max-model-len $max_len..."
    mkdir -p /workspace/gepa-mutations/logs
    cd /tmp  # avoid vLLM IPC socket path length bug
    CUDA_VISIBLE_DEVICES=$gpu nohup python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --dtype auto \
        --max-model-len "$max_len" \
        --gpu-memory-utilization 0.90 \
        --enforce-eager \
        --host 0.0.0.0 \
        --port "$port" \
        --no-enable-log-requests \
        > /workspace/gepa-mutations/logs/vllm_${label}.log 2>&1 &
    cd /workspace/gepa-mutations
    echo "    PID: $!"
}

# --- Helper: wait for vLLM server ---
wait_for_server() {
    local port=$1 timeout_secs=${2:-300}
    local elapsed=0
    while ! curl -sf http://localhost:$port/v1/models > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $timeout_secs ]; then
            echo "  TIMEOUT: port $port not ready after ${timeout_secs}s"
            echo "  Check: tail -50 logs/vllm_*.log"
            exit 1
        fi
    done
    local model
    model=$(curl -s http://localhost:$port/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    echo "  Port $port: ready ($model) [${elapsed}s]"
}

# --- Helper: launch orchestrator if not already running ---
launch_orchestrator() {
    local model=$1 url=$2 tag=$3 workers=$4
    shift 4
    local benchmarks="$*"

    if pgrep -f "orchestrator_${tag}_runpod" > /dev/null 2>&1; then
        echo "  $tag: orchestrator already running"
        return
    fi

    echo "  $tag: launching ($workers workers, benchmarks: $benchmarks)..."
    GEPA_MODEL="$model" GEPA_BASE_URL="$url" API_BASE_URL="$url" \
        nohup python scripts/run_all_local.py \
        --workers "$workers" \
        --benchmark $benchmarks \
        --method $METHODS \
        > logs/orchestrator_${tag}_runpod.log 2>&1 &
    echo "    PID: $!"
}

# =========================================================================
# Smoke test: start server, run smoke test, report, stop
# =========================================================================
if [ "$PHASE" = "smoke" ]; then
    echo ""
    echo "--- Smoke test mode: Qwen3-8B ---"
    start_vllm 0 "Qwen/Qwen3-8B" 8125 16384 "8b"

    echo ""
    echo "--- Waiting for server ---"
    wait_for_server 8125

    echo ""
    echo "--- Running smoke test ---"
    GEPA_MODEL="Qwen/Qwen3-8B" GEPA_BASE_URL="http://localhost:8125/v1" \
        python scripts/run_all_local.py --smoke-test --workers 4 \
        --benchmark $BENCHMARKS --method $METHODS

    echo ""
    echo "Smoke test complete. Check results."
    exit 0
fi

# =========================================================================
# Full sweep: 8B on GPU 0
# =========================================================================
if [ "$PHASE" = "1" ]; then
    echo ""
    echo "--- Starting vLLM server (Qwen3-8B) ---"
    start_vllm 0 "Qwen/Qwen3-8B" 8125 16384 "8b"

    echo ""
    echo "--- Waiting for server ---"
    wait_for_server 8125

    echo ""
    echo "--- Running baselines ---"
    GEPA_MODEL="Qwen/Qwen3-8B" GEPA_BASE_URL="http://localhost:8125/v1" \
        python scripts/run_baseline.py --benchmark $BENCHMARKS 2>&1 | tail -5

    echo ""
    echo "--- Health check ---"
    curl -sf http://localhost:8125/v1/models > /dev/null && echo "  Server healthy" || { echo "  ERROR: server not responding"; exit 1; }

    echo ""
    echo "--- Launching orchestrator (Qwen3-8B) ---"
    launch_orchestrator "Qwen/Qwen3-8B" "http://localhost:8125/v1" "8b" 6 $BENCHMARKS

    echo ""
    echo "=== Startup complete ==="
    echo ""
    echo "Monitor:"
    echo "  tail -f logs/orchestrator_8b_runpod.log"
    echo ""
    echo "Progress:"
    echo "  bash scripts/runpod_progress.sh"

    echo ""
    echo "--- Git checkpoint ---"
    git_checkpoint

else
    echo "Unknown argument: $PHASE (use 1 or smoke)"
    exit 1
fi
