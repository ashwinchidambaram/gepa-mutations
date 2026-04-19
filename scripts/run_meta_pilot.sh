#!/bin/bash
set -euo pipefail

# Track 2: Meta-optimizer pilot
META_VARIANT="scout"
INNER_VARIANT="iso_tide"
BENCHMARK="ifbench"
N_EPISODES=50
SURROGATE_SIZE=20

while [[ $# -gt 0 ]]; do
    case $1 in
        --meta-variant) META_VARIANT="$2"; shift 2 ;;
        --inner-variant) INNER_VARIANT="$2"; shift 2 ;;
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --n-episodes) N_EPISODES="$2"; shift 2 ;;
        --surrogate-size) SURROGATE_SIZE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  --meta-variant    scout|cartographer|atlas (default: scout)"
            echo "  --inner-variant   ISO variant for inner loop (default: iso_tide)"
            echo "  --benchmark       Benchmark name (default: ifbench)"
            echo "  --n-episodes      Number of meta episodes (default: 50)"
            echo "  --surrogate-size  Surrogate subset size (default: 20)"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== META-Optimizer Pilot ==="
echo "Meta variant: $META_VARIANT"
echo "Inner variant: $INNER_VARIANT"
echo "Benchmark: $BENCHMARK"
echo "Episodes: $N_EPISODES"
echo "Surrogate size: $SURROGATE_SIZE"
echo ""

python -m iso_harness.meta.cli \
    --meta-variant "$META_VARIANT" \
    --inner-variant "$INNER_VARIANT" \
    --benchmark "$BENCHMARK" \
    --n-episodes "$N_EPISODES" \
    --surrogate-size "$SURROGATE_SIZE"
