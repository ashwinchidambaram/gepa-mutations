#!/bin/bash
# RunPod startup/recovery script
# Run after provisioning a pod OR after a preemption restart.
# Idempotent: skips already-running servers and orchestrators.
#
# Usage: bash scripts/runpod_start.sh [phase]
#   phase 1 (default): 1.7B + 8B
#   phase 2: swap 1.7B → 4B (run after 1.7B completes)

set -euo pipefail

PHASE="${1:-1}"
BENCHMARKS="hotpotqa hover pupa ifbench livebench aime"
METHODS="gepa best_of_k_K3 contrastive_reflection synaptic_pruning slime_mold tournament"

echo "=== RunPod Sweep — Phase $PHASE ==="

# --- GPU Info ---
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "GPUs detected: $GPU_COUNT"

# Validate GPU count for Phase 1
if [ "$PHASE" = "1" ] && [ "$GPU_COUNT" -lt 2 ]; then
    echo "WARNING: Phase 1 requires 2 GPUs, found $GPU_COUNT"
fi

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

# --- Helper: start vLLM if not already running ---
start_vllm() {
    local gpu=$1 model=$2 port=$3 max_len=$4 label=$5

    if curl -sf http://localhost:$port/v1/models > /dev/null 2>&1; then
        echo "  $label: already running on :$port"
        return
    fi

    echo "  $label: starting on GPU $gpu, port $port, max-model-len $max_len..."
    mkdir -p logs
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
# Phase 1: 1.7B (GPU 0) + 8B (GPU 1)
# =========================================================================
if [ "$PHASE" = "1" ]; then
    echo ""
    echo "--- Starting vLLM servers (Phase 1) ---"
    start_vllm 0 "Qwen/Qwen3-1.7B" 8127 16384 "1.7b"
    start_vllm 1 "Qwen/Qwen3-8B"   8125 16384 "8b"

    echo ""
    echo "--- Waiting for servers ---"
    wait_for_server 8127
    wait_for_server 8125

    echo ""
    echo "--- Running baselines ---"
    GEPA_MODEL="Qwen/Qwen3-1.7B" GEPA_BASE_URL="http://localhost:8127/v1" \
        python scripts/run_baseline.py --benchmark $BENCHMARKS 2>&1 | tail -5 &
    GEPA_MODEL="Qwen/Qwen3-8B" GEPA_BASE_URL="http://localhost:8125/v1" \
        python scripts/run_baseline.py --benchmark $BENCHMARKS 2>&1 | tail -5 &
    wait

    echo ""
    echo "--- Launching orchestrators (Phase 1) ---"
    launch_orchestrator "Qwen/Qwen3-1.7B" "http://localhost:8127/v1" "1.7b" 8 $BENCHMARKS
    launch_orchestrator "Qwen/Qwen3-8B"   "http://localhost:8125/v1" "8b"   6 $BENCHMARKS

# =========================================================================
# Phase 2: swap 1.7B → 4B (GPU 0), 8B continues on GPU 2
# =========================================================================
elif [ "$PHASE" = "2" ]; then
    echo ""
    echo "--- Phase 2: swapping 1.7B → 4B on GPU 0 ---"

    # Kill 1.7B vLLM server
    if curl -sf http://localhost:8127/v1/models > /dev/null 2>&1; then
        echo "  Stopping 1.7B server on :8127..."
        pkill -f "port 8127" || true
        sleep 5
    fi

    # Download 4B if needed (cached on persistent volume)
    echo "  Downloading Qwen/Qwen3-4B (if not cached)..."
    huggingface-cli download Qwen/Qwen3-4B --quiet

    # Start 4B
    start_vllm 0 "Qwen/Qwen3-4B" 8127 16384 "4b"

    echo ""
    echo "--- Waiting for 4B server ---"
    wait_for_server 8127

    echo ""
    echo "--- Running 4B baselines ---"
    GEPA_MODEL="Qwen/Qwen3-4B" GEPA_BASE_URL="http://localhost:8127/v1" \
        python scripts/run_baseline.py --benchmark $BENCHMARKS 2>&1 | tail -5

    echo ""
    echo "--- Launching 4B orchestrator ---"
    launch_orchestrator "Qwen/Qwen3-4B" "http://localhost:8127/v1" "4b" 8 $BENCHMARKS

else
    echo "Unknown phase: $PHASE (use 1 or 2)"
    exit 1
fi

echo ""
echo "=== Startup complete ==="
echo ""
echo "Monitor:"
echo "  tail -f logs/orchestrator_*_runpod.log"
echo ""
echo "Progress:"
echo "  bash scripts/runpod_progress.sh"
