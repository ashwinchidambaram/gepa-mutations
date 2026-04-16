#!/bin/bash

# preflight_a40_throughput.sh
# Smoke test vLLM server throughput on A40 GPU
# Measures concurrent LLM calls/sec; PASS if >= 0.4 calls/sec

set -euo pipefail

# Default values
URL="${GEPA_BASE_URL:-http://localhost:8000}/v1"
N_PARALLEL=50
PROMPT_FILE=""
DEFAULT_PROMPT="Explain the concept of transformer attention in 3 sentences."

# Functions
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Measure vLLM server throughput with concurrent LLM calls.

OPTIONS:
  --url URL              Completions endpoint URL (default: \$GEPA_BASE_URL/v1)
  --n N                  Number of parallel calls (default: 50)
  --prompt-file FILE     Read prompt from file (default: hardcoded prompt)
  --help                 Show this help message

EXIT CODES:
  0  - Throughput >= 0.4 calls/sec (PASS)
  1  - Throughput < 0.4 calls/sec (FAIL)

ENVIRONMENT:
  GEPA_BASE_URL         Base URL for vLLM endpoint (e.g., http://localhost:8000)
  GEPA_MODEL            Model name (default: qwen/qwen3-8b)

EXAMPLE:
  export GEPA_BASE_URL=http://localhost:8000
  export GEPA_MODEL=qwen/qwen3-8b
  $0 --n 50

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            usage
            ;;
        --url)
            URL="$2"
            shift 2
            ;;
        --n)
            N_PARALLEL="$2"
            shift 2
            ;;
        --prompt-file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

# Determine prompt
if [[ -n "$PROMPT_FILE" && -f "$PROMPT_FILE" ]]; then
    PROMPT=$(cat "$PROMPT_FILE")
else
    PROMPT="$DEFAULT_PROMPT"
fi

# Validate environment
if [[ -z "${GEPA_MODEL:-}" ]]; then
    echo "Error: GEPA_MODEL not set" >&2
    echo "Set: export GEPA_MODEL=qwen/qwen3-8b" >&2
    exit 1
fi

MODEL="${GEPA_MODEL}"

# Build JSON payload
PAYLOAD=$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "$PROMPT"}],
  "max_tokens": 100
}
EOF
)

# Escape special characters in JSON payload for shell expansion
PAYLOAD_ESCAPED=$(printf '%s\n' "$PAYLOAD" | jq -c .)

echo "=== vLLM Throughput Preflight ==="
echo "URL: $URL/chat/completions"
echo "Model: $MODEL"
echo "Parallel calls: $N_PARALLEL"
echo "Prompt: $(echo "$PROMPT" | head -c 50)..."
echo ""

# Start timing
START_NS=$(date +%s%N)
START_SECONDS=$(date +%s)

# Launch parallel calls using GNU parallel or xargs
# Each call: curl to the vLLM endpoint
call_lm() {
    curl -sf -X POST "$URL/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$1" > /dev/null 2>&1
}

export -f call_lm
export URL PAYLOAD_ESCAPED

# Run N parallel calls
if command -v parallel &> /dev/null; then
    # GNU parallel (preferred)
    seq 1 "$N_PARALLEL" | parallel -j "$N_PARALLEL" "call_lm '$PAYLOAD_ESCAPED'" || true
else
    # Fallback to xargs
    seq 1 "$N_PARALLEL" | xargs -P "$N_PARALLEL" -I {} bash -c "call_lm '$PAYLOAD_ESCAPED'" || true
fi

# Stop timing
END_NS=$(date +%s%N)
END_SECONDS=$(date +%s)

# Calculate elapsed time in seconds
ELAPSED_NS=$((END_NS - START_NS))
ELAPSED_SEC=$(echo "scale=3; $ELAPSED_NS / 1000000000" | bc)

# Calculate throughput
THROUGHPUT=$(echo "scale=4; $N_PARALLEL / $ELAPSED_SEC" | bc)

# Results
echo "Total calls: $N_PARALLEL"
echo "Wall time: ${ELAPSED_SEC}s"
echo "Throughput: $THROUGHPUT calls/sec"
echo ""

# Pass/fail
THRESHOLD="0.4"
if (( $(echo "$THROUGHPUT >= $THRESHOLD" | bc -l) )); then
    echo "✓ PASS: throughput >= $THRESHOLD calls/sec"
    exit 0
else
    echo "✗ FAIL: throughput $THROUGHPUT < $THRESHOLD calls/sec"
    exit 1
fi
