#!/usr/bin/env bash
# Pilot driver: GEPA vs Slime Mold on HotpotQA, Qwen3-8B, paper budget.
#
# Run on the A40 pod after vLLM is up at $GEPA_BASE_URL.
# See README.md in this directory for the pre-registered decision criteria.
set -euo pipefail

PORT="${PORT:-8125}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
EXPERIMENT_DIR="experiments/04-pilot-gepa-comparison-2026-04"
LOG_DIR="$EXPERIMENT_DIR/logs"
RUNS_DIR="$EXPERIMENT_DIR/runs"
mkdir -p "$LOG_DIR" "$RUNS_DIR"

VLLM_URL="http://localhost:$PORT/v1"
export GEPA_MODEL="$MODEL"
export GEPA_BASE_URL="$VLLM_URL"
export API_BASE_URL="$VLLM_URL"
# MODEL_PREFIX=openai routes litellm through the local vLLM endpoint
# (using OpenAI-compatible API). Without this, model_id becomes
# "openrouter/Qwen/Qwen3-8B" and litellm tries to authenticate against
# OpenRouter regardless of api_base. See scripts/launch_mac_sweep.sh for
# the same convention.
export MODEL_PREFIX="openai"
# vLLM accepts any non-empty api_key, but litellm requires the field to be set
# when MODEL_PREFIX=openai (it expects OpenAI auth headers).
export API_KEY="${API_KEY:-EMPTY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Reduce test_eval parallelism from default 10 → 5. The original setting
# overwhelmed vLLM during the smoke test (GPU memory dropped to 0 mid-run).
# 5 parallel workers per method × 1 sequential method = 5 max concurrent
# requests on vLLM, well within A40's batch capacity.
export TEST_EVAL_WORKERS="${TEST_EVAL_WORKERS:-5}"
export EXPERIMENT_LOGS_DIR="$LOG_DIR"

_usage() {
    cat <<EOF
Usage: ${0##*/} [smoke|pilot|analyze|help]

  smoke    Tiny smoke test (subset 5, seed 555 only) — verifies both methods
           run end-to-end. ~5 min.
  pilot    Full pilot: 6 runs (gepa + slime_mold × seeds 555,999,1337) at paper
           budget 6871. ~3 hrs per run, ~\$10-15 total cost.
  analyze  Run analyze.py on the completed runs.
  help     Show this message.

Environment:
  PORT=$PORT
  MODEL=$MODEL
  GEPA_BASE_URL=$VLLM_URL
EOF
}

_wait_vllm() {
    echo "Waiting for vLLM at $VLLM_URL..."
    for _ in $(seq 1 60); do
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
        bash "$(dirname "$0")/../../scripts/serve_vllm_8b_a40.sh" > "$LOG_DIR/vllm.log" 2>&1 &
        _wait_vllm
    else
        echo "vLLM already running at $VLLM_URL"
    fi
}

_smoke() {
    _ensure_vllm
    echo "Running pilot smoke test (subset 5, seed 555, both methods)"
    .venv/bin/python scripts/run_all_local.py \
        --smoke-test \
        --workers 2 \
        --seeds 555 \
        --method gepa slime_mold \
        --benchmark hotpotqa \
        --runs-dir "$RUNS_DIR" 2>&1 | tee "$LOG_DIR/smoke.log"
}

_pilot() {
    _ensure_vllm
    echo "=== Pilot: GEPA vs Slime Mold on HotpotQA ==="
    echo "Methods: gepa, slime_mold"
    echo "Seeds:   555, 999, 1337"
    echo "Budget:  6871 rollouts each (paper default for HotpotQA)"
    echo "Runs:    6 total"
    echo "Logs:    $LOG_DIR/pilot.log"
    echo ""
    # --workers 1: run methods strictly sequentially. Avoids the parallelism
    # crash we hit during smoke (vLLM dropped its GPU allocation under
    # gepa+slime_mold concurrent load). Doubles wall-clock to ~18 hr but
    # eliminates that risk entirely.
    .venv/bin/python scripts/run_all_local.py \
        --workers 1 \
        --seeds 555,999,1337 \
        --method gepa slime_mold \
        --benchmark hotpotqa \
        --runs-dir "$RUNS_DIR" 2>&1 | tee "$LOG_DIR/pilot.log"
    echo ""
    echo "=== Pilot complete ==="
    echo "Total result.json files: $(find $RUNS_DIR -name result.json | wc -l)"
    echo ""
    echo "Next: bash $0 analyze"
}

_analyze() {
    local run_path
    run_path="$RUNS_DIR/$(ls $RUNS_DIR | head -1)/hotpotqa"
    if [ ! -d "$run_path" ]; then
        echo "ERROR: expected runs at $run_path but it doesn't exist" >&2
        exit 1
    fi
    .venv/bin/python "$EXPERIMENT_DIR/analyze.py" --runs-dir "$run_path"
}

case "${1:-help}" in
    smoke)   _smoke ;;
    pilot)   _pilot ;;
    analyze) _analyze ;;
    help|-h|--help) _usage ;;
    *)       echo "Unknown subcommand: $1"; _usage; exit 1 ;;
esac
