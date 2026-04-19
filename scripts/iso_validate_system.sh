#!/bin/bash
# iso_validate_system.sh — Run V1-V10 pre-pilot validation sequence
# Halts on first failure. Produces validation_report.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

echo "=== ISO System Validation ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Git SHA: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo ""

REPORT="$PROJECT_DIR/validation_report.md"
PASS=0
FAIL=0
TOTAL_START=$(date +%s)

# Initialize report
cat > "$REPORT" << EOF
# ISO System Validation Report

**Timestamp:** $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Git SHA:** $(git rev-parse HEAD 2>/dev/null || echo 'unknown')

## Environment
- **Python:** $(python --version 2>&1)
- **PyTorch:** $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")
- **vLLM:** $(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "N/A")
- **DSPy:** $(python -c "import dspy; print(dspy.__version__)" 2>/dev/null || echo "N/A")
- **MLflow:** $(python -c "import mlflow; print(mlflow.__version__)" 2>/dev/null || echo "N/A")

## Results

| # | Validation | Status | Duration |
|---|-----------|--------|----------|
EOF

# Validation sequence — runs in order, halts on first failure
VALIDATIONS=(
    "V1:Environment:tests/validation/test_environment.py"
    "V2:Models:tests/validation/test_models.py"
    "V3:Servers:tests/validation/test_servers.py"
    "V4:Inference:tests/validation/test_inference.py"
    "V5:DSPy:tests/validation/test_dspy.py"
    "V6:Benchmarks:tests/validation/test_benchmarks.py"
    "V7:Logging:tests/validation/test_logging.py"
    "V8:Checkpoint:tests/validation/test_checkpoint.py"
    "V9:Smoke:tests/validation/test_smoke.py"
    "V10:Rsync:tests/validation/test_rsync.py"
)

for entry in "${VALIDATIONS[@]}"; do
    IFS=':' read -r num name path <<< "$entry"

    echo -n "[$num] $name... "
    START=$(date +%s)

    if [ -f "$path" ]; then
        if python -m pytest "$path" -v --tb=short 2>&1 | tee -a "$PROJECT_DIR/logs/validation_${num}.log" | tail -1 | grep -q "passed"; then
            END=$(date +%s)
            DURATION=$((END - START))
            echo "PASS (${DURATION}s)"
            echo "| $num | $name | ✓ PASS | ${DURATION}s |" >> "$REPORT"
            PASS=$((PASS + 1))
        else
            END=$(date +%s)
            DURATION=$((END - START))
            echo "FAIL (${DURATION}s)"
            echo "| $num | $name | ✗ FAIL | ${DURATION}s |" >> "$REPORT"
            FAIL=$((FAIL + 1))

            echo ""
            echo "=== VALIDATION FAILED at $num: $name ==="
            echo "See: logs/validation_${num}.log"
            echo "See: TROUBLESHOOTING.md for diagnostic guidance"

            # Write remaining as skipped
            for remaining in "${VALIDATIONS[@]}"; do
                IFS=':' read -r rnum rname rpath <<< "$remaining"
                if [[ "$rnum" > "$num" ]]; then
                    echo "| $rnum | $rname | — SKIP | — |" >> "$REPORT"
                fi
            done
            break
        fi
    else
        echo "SKIP (test file not found)"
        echo "| $num | $name | — SKIP | — |" >> "$REPORT"
    fi
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

# ---------------------------------------------------------------------------
# V11-V16a: ISO Optimizer Validation
# ---------------------------------------------------------------------------
echo ""
echo "=== V11-V16a: ISO Optimizer Validation ==="
echo ""

ISO_VALIDATIONS=(
    "V11:SkillDiscovery:tests/validation/test_skill_discovery.py"
    "V12:VariantIsolation:tests/validation/test_variants.py"
    "V13:ReflectionScopes:tests/validation/test_reflection.py"
    "V14:PruningRules:tests/validation/test_pruning.py"
    "V15:MergeOperator:tests/validation/test_merge.py"
    "V16a:E2ESmokeTest:tests/validation/test_iso_smoke.py"
)

for entry in "${ISO_VALIDATIONS[@]}"; do
    IFS=':' read -r num name path <<< "$entry"

    echo -n "[$num] $name... "
    START=$(date +%s)

    if [ -f "$path" ]; then
        # V16a excludes live_server tests (mock-LM only)
        PYTEST_EXTRA_ARGS=""
        if [ "$num" = "V16a" ]; then
            PYTEST_EXTRA_ARGS="-k not live_server"
        fi

        if python -m pytest "$path" -v --tb=short $PYTEST_EXTRA_ARGS 2>&1 \
                | tee -a "$PROJECT_DIR/logs/validation_${num}.log" \
                | tail -1 | grep -q "passed"; then
            END=$(date +%s)
            DURATION=$((END - START))
            echo "PASS (${DURATION}s)"
            echo "| $num | $name | ✓ PASS | ${DURATION}s |" >> "$REPORT"
            PASS=$((PASS + 1))
        else
            END=$(date +%s)
            DURATION=$((END - START))
            echo "FAIL (${DURATION}s)"
            echo "| $num | $name | ✗ FAIL | ${DURATION}s |" >> "$REPORT"
            FAIL=$((FAIL + 1))

            echo ""
            echo "=== VALIDATION FAILED at $num: $name ==="
            echo "See: logs/validation_${num}.log"

            # Mark remaining ISO validations as skipped
            SKIP=false
            for remaining in "${ISO_VALIDATIONS[@]}"; do
                IFS=':' read -r rnum rname rpath <<< "$remaining"
                if [ "$SKIP" = true ]; then
                    echo "| $rnum | $rname | — SKIP | — |" >> "$REPORT"
                fi
                if [ "$rnum" = "$num" ]; then
                    SKIP=true
                fi
            done
            break
        fi
    else
        echo "SKIP (test file not found)"
        echo "| $num | $name | — SKIP | — |" >> "$REPORT"
    fi
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

# Write summary to report
cat >> "$REPORT" << EOF

## Summary

- **Passed:** $PASS
- **Failed:** $FAIL
- **Total duration:** ${TOTAL_DURATION}s
EOF

echo ""
echo "=== Validation Complete: $PASS passed, $FAIL failed (${TOTAL_DURATION}s) ==="
echo "Report: $REPORT"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
