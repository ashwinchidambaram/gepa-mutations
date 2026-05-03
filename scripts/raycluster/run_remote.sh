#!/usr/bin/env bash
# Kick off experiment on gho-vm-2 via SSH (persists after disconnect).
#
# Usage:
#   bash scripts/raycluster/run_remote.sh baseline                    # All benchmarks, all seeds
#   bash scripts/raycluster/run_remote.sh baseline --benchmark hotpotqa hover
#   bash scripts/raycluster/run_remote.sh baseline --seeds 42 123
#   bash scripts/raycluster/run_remote.sh status                      # Check running jobs
#   bash scripts/raycluster/run_remote.sh logs                        # Tail latest log
#   bash scripts/raycluster/run_remote.sh results                     # Copy results back

set -euo pipefail

VM_HOST="10.0.50.65"
VM_USER="achidamb"
REMOTE_DIR="local-projects/gepa-mutations-cluster"
SSH_CMD="ssh -o StrictHostKeyChecking=accept-new ${VM_USER}@${VM_HOST}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

ACTION="${1:-help}"
shift 2>/dev/null || true

case "$ACTION" in
    baseline)
        echo "Starting baseline run on gho-vm-2..."
        ARGS="${*:-}"
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        LOG_FILE="baseline_${TIMESTAMP}.log"

        ${SSH_CMD} "export PATH=\$HOME/.local/bin:\$PATH && cd ~/${REMOTE_DIR} && nohup uv run python scripts/raycluster/run_baseline.py ${ARGS} > ${LOG_FILE} 2>&1 & echo \"PID: \$!\""
        echo ""
        echo "Job started. Log: ~/${REMOTE_DIR}/${LOG_FILE}"
        echo ""
        echo "Monitor:  bash scripts/raycluster/run_remote.sh logs"
        echo "Status:   bash scripts/raycluster/run_remote.sh status"
        echo "Results:  bash scripts/raycluster/run_remote.sh results"
        ;;

    gepa)
        echo "Starting GEPA optimization on gho-vm-2..."
        ARGS="${*:-}"
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        LOG_FILE="gepa_${TIMESTAMP}.log"

        ${SSH_CMD} "export PATH=\$HOME/.local/bin:\$PATH && cd ~/${REMOTE_DIR} && nohup uv run python scripts/raycluster/run_gepa.py ${ARGS} > ${LOG_FILE} 2>&1 & echo \"PID: \$!\""
        echo ""
        echo "Job started. Log: ~/${REMOTE_DIR}/${LOG_FILE}"
        echo ""
        echo "Monitor:  bash scripts/raycluster/run_remote.sh logs"
        echo "Status:   bash scripts/raycluster/run_remote.sh status"
        echo "Results:  bash scripts/raycluster/run_remote.sh results"
        ;;

    status)
        echo "Checking running jobs on gho-vm-2..."
        ${SSH_CMD} "export PATH=\$HOME/.local/bin:\$PATH && ps aux | grep 'run_baseline\|run_gepa\|run_iso' | grep -v grep || echo 'No jobs running.'"
        echo ""
        echo "Recent log files:"
        ${SSH_CMD} "cd ~/${REMOTE_DIR} && ls -lt *.log 2>/dev/null | head -5 || echo 'No logs yet.'"
        ;;

    logs)
        LOG="${1:-}"
        if [ -z "$LOG" ]; then
            echo "Tailing most recent log..."
            ${SSH_CMD} "cd ~/${REMOTE_DIR} && tail -f \$(ls -t *.log 2>/dev/null | head -1)"
        else
            ${SSH_CMD} "cd ~/${REMOTE_DIR} && tail -f ${LOG}"
        fi
        ;;

    results)
        echo "Copying results from gho-vm-2..."
        LOCAL_RUNS="${REPO_ROOT}/runs"
        mkdir -p "${LOCAL_RUNS}"
        rsync -avz --exclude='__pycache__' \
            "${VM_USER}@${VM_HOST}:~/${REMOTE_DIR}/runs/" \
            "${LOCAL_RUNS}/"
        echo ""
        echo "Results synced to ${LOCAL_RUNS}/"
        find "${LOCAL_RUNS}" -name "result.json" -newer "${LOCAL_RUNS}/README.md" 2>/dev/null | while read f; do
            score=$(python3 -c "import json; print(f'{json.load(open(\"$f\"))[\"test_score\"]:.4f}')" 2>/dev/null || echo "?")
            echo "  ${f#${LOCAL_RUNS}/}: score=${score}"
        done
        ;;

    iso)
        echo "Starting ISO optimization on gho-vm-2..."
        ARGS="${*:-}"
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        LOG_FILE="iso_${TIMESTAMP}.log"

        ${SSH_CMD} "export PATH=\$HOME/.local/bin:\$PATH && cd ~/${REMOTE_DIR} && nohup uv run python scripts/raycluster/run_iso.py ${ARGS} > ${LOG_FILE} 2>&1 & echo \"PID: \$!\""
        echo ""
        echo "Job started. Log: ~/${REMOTE_DIR}/${LOG_FILE}"
        echo ""
        echo "Monitor:  bash scripts/raycluster/run_remote.sh logs"
        echo "Status:   bash scripts/raycluster/run_remote.sh status"
        echo "Results:  bash scripts/raycluster/run_remote.sh results"
        ;;

    help|*)
        echo "Usage: bash scripts/raycluster/run_remote.sh <action> [args]"
        echo ""
        echo "Actions:"
        echo "  baseline [--benchmark X] [--seeds N]     Start baseline evaluation"
        echo "  gepa [--benchmark X] [--seeds N]         Start GEPA optimization"
        echo "  iso [--variant V] [--benchmark X] [...]  Start ISO optimization"
        echo "  status                                    Check running jobs"
        echo "  logs [filename]                           Tail experiment logs"
        echo "  results                                   Copy results back to local"
        echo ""
        echo "Examples:"
        echo "  bash scripts/raycluster/run_remote.sh baseline"
        echo "  bash scripts/raycluster/run_remote.sh gepa --benchmark hotpotqa --seeds 42"
        echo "  bash scripts/raycluster/run_remote.sh iso --variant sprint grove --benchmark hotpotqa"
        echo "  bash scripts/raycluster/run_remote.sh status"
        echo "  bash scripts/raycluster/run_remote.sh results"
        ;;
esac
