#!/bin/bash
set -e
cd /Users/ashwinchidambaram/dev/projects/gepa-mutations

echo "=== Starting all benchmark smoke tests ==="
echo "Start time: $(date)"
echo ""

for bm in hotpotqa ifbench hover pupa livebench; do
    echo "=========================================="
    echo "Starting smoke test: $bm"
    echo "Time: $(date)"
    echo "=========================================="
    uv run gepa-mutations run "$bm" --subset 5 --seed 42 --max-metric-calls 50 2>&1
    echo ""
    echo "$bm completed at $(date)"
    echo ""
done

echo "=== All smoke tests complete ==="
echo "End time: $(date)"
