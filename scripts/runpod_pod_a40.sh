#!/usr/bin/env bash
set -euo pipefail

# Configuration
PORT="${PORT:-8125}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
EXPERIMENT_DIR="experiments/03-inductive-discovery-2026-04"
LOG_DIR="$EXPERIMENT_DIR/logs"
mkdir -p "$LOG_DIR"

VLLM_URL="http://localhost:$PORT/v1"
export GEPA_MODEL="$MODEL"
export GEPA_BASE_URL="$VLLM_URL"
export API_BASE_URL="$VLLM_URL"  # some code paths use this instead
export EXPERIMENT_LOGS_DIR="$LOG_DIR"

_usage() {
    cat <<EOF
Usage: $0 [smoke|preflight|tier1|help]

  smoke      Run a 5-example smoke test across all 5 Tier 1 methods on HotpotQA
  preflight  Run A40 throughput + adaptive K probe + discovery quality check
  tier1      Launch full Tier 1 experiment (HotpotQA first, then HOVER/PUPA/IFBench)
  help       Show this message

Environment:
  PORT=$PORT
  MODEL=$MODEL
  GEPA_BASE_URL=$VLLM_URL
  EXPERIMENT_LOGS_DIR=$LOG_DIR
EOF
}

_wait_vllm() {
    echo "Waiting for vLLM at $VLLM_URL..."
    for i in $(seq 1 60); do
        if curl -sf "$VLLM_URL/models" > /dev/null; then
            echo "vLLM is up"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: vLLM did not come up within 5 minutes" >&2
    return 1
}

_ensure_vllm() {
    if ! curl -sf "$VLLM_URL/models" > /dev/null; then
        echo "vLLM not running — starting serve_vllm_8b_a40.sh in background"
        bash "$(dirname "$0")/serve_vllm_8b_a40.sh" > "$LOG_DIR/vllm.log" 2>&1 &
        _wait_vllm
    else
        echo "vLLM already running at $VLLM_URL"
    fi
}

_smoke() {
    _ensure_vllm
    echo "Running smoke test (5 examples, 1 seed, all 5 Tier 1 methods)"
    .venv/bin/python scripts/run_all_local.py \
        --smoke-test --workers 4 \
        --seeds 42 \
        --method \
            iso \
            iso_prescribed8 \
            iso_inductive_k5 \
            iso_inductive_k5_crosspollin \
            iso_inductive_k5_refresh_expand \
        --benchmark hotpotqa 2>&1 | tee "$LOG_DIR/smoke.log"
}

_preflight() {
    _ensure_vllm
    echo "Running A40 throughput preflight..."
    bash scripts/preflight_a40_throughput.sh --url "$VLLM_URL" 2>&1 | tee "$LOG_DIR/preflight_throughput.log"
    echo ""
    echo "Running adaptive K probe..."
    .venv/bin/python scripts/preflight_adaptive_k_probe.py \
        --benchmarks hotpotqa hover pupa ifbench \
        --seeds 42 123 456 \
        --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/preflight_adaptive_k.log"
    echo ""
    echo "Running discovery quality probe..."
    .venv/bin/python scripts/preflight_discovery_quality.py \
        --benchmarks hotpotqa hover pupa \
        --seeds 42 123 456 \
        --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/preflight_discovery_quality.log"
    echo ""
    echo "Preflight outputs written to $LOG_DIR/"
    echo "MANUAL REVIEW REQUIRED: inspect $LOG_DIR/discovery_preflight.md"
}

_tier1_benchmark() {
    local benchmark="$1"
    local seeds="$2"
    echo "Running Tier 1 on $benchmark with seeds: $seeds"
    .venv/bin/python scripts/run_all_local.py \
        --workers 8 \
        --seeds $seeds \
        --method \
            iso \
            iso_prescribed8 \
            iso_inductive_k5 \
            iso_inductive_k5_crosspollin \
            iso_inductive_k5_refresh_expand \
        --benchmark "$benchmark" 2>&1 | tee -a "$LOG_DIR/tier1_$benchmark.log"
}

_tier1() {
    _ensure_vllm
    echo "=== Tier 1 experiment ==="
    echo "Starting with HotpotQA (10 seeds) — the variance benchmark"
    _tier1_benchmark hotpotqa "42 123 456 789 1024 2048 4096 8192 16384 32768"
    echo "HotpotQA complete — checkpointing"
    git add "$EXPERIMENT_DIR"/runs/ "$EXPERIMENT_DIR"/plots/ 2>/dev/null || true
    git commit -m "data: HotpotQA Tier 1 complete" || echo "(nothing to commit)"

    for bench in hover pupa ifbench; do
        _tier1_benchmark "$bench" "42 123 456 789 1024"
        echo "$bench complete — checkpointing"
        git add "$EXPERIMENT_DIR"/runs/ "$EXPERIMENT_DIR"/plots/ 2>/dev/null || true
        git commit -m "data: $bench Tier 1 complete" || echo "(nothing to commit)"
    done

    echo ""
    echo "=== Tier 1 complete ==="
    echo "Total runs: $(find $EXPERIMENT_DIR/runs -name result.json | wc -l)"
}

case "${1:-help}" in
    smoke)      _smoke ;;
    preflight)  _preflight ;;
    tier1)      _tier1 ;;
    help|-h|--help) _usage ;;
    *)          echo "Unknown subcommand: $1"; _usage; exit 1 ;;
esac
