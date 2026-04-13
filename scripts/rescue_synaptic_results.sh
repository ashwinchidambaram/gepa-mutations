#!/usr/bin/env bash
# Rescue in-flight synaptic_pruning/ifbench results that wrote to wrong path
# due to missing model_tag in runner.py (now fixed for future tasks).
#
# In-flight tasks that started before the fix must be caught here.
# Completion order for seed=1024: 27B (~19:27) → 8B (~19:51) → 14B (~20:09)
# Each overwrites runs/ifbench/synaptic_pruning/1024/result.json — must copy fast.

RUNS=/users/achidamb/projects/gepa-mutations/runs
WRONG=$RUNS/ifbench/synaptic_pruning
LOG27=$RUNS/../logs/orchestrator_27b.log
LOG8=$RUNS/../logs/orchestrator_8b.log
LOG14=$RUNS/../logs/orchestrator_14b.log

echo "[$(date '+%T')] Starting rescue watcher..."

# Already rescued manually: 8B seeds 123/789/42/456, 14B seed=123
# Still pending:
#   27B: seed=1024 (started 18:54, ~33min tasks → ETA ~19:27)
#   8B:  seed=1024 (started 19:06, ~45min tasks → ETA ~19:51)
#   14B: seeds 789, 42, 456 (started 18:21 → ETA ~19:15-19:40), seed=1024 (started 19:15 → ETA ~20:09)

declare -A DONE_27B=()
declare -A DONE_8B=()
declare -A DONE_14B=()

SEEDS_27B=(1024)
SEEDS_8B=(1024)
SEEDS_14B=(789 42 456 1024)

copy_result() {
    local model=$1 seed=$2
    local src="$WRONG/${seed}/result.json"
    local dst_dir="$RUNS/${model}/ifbench/synaptic_pruning/${seed}"
    if [[ -f "$src" ]]; then
        mkdir -p "$dst_dir"
        cp "$src" "$dst_dir/result.json"
        echo "[$(date '+%T')] ✓ Saved ${model} seed=${seed}"
    else
        echo "[$(date '+%T')] ✗ Missing: $src (seed=${seed} done but no file?)"
    fi
}

while true; do
    sleep 20

    # Check 27B
    for seed in "${SEEDS_27B[@]}"; do
        [[ -v DONE_27B[$seed] ]] && continue
        if grep -q "DONE.*ifbench/synaptic_pruning/seed=${seed}" "$LOG27" 2>/dev/null; then
            copy_result qwen3-27b-awq "$seed"
            DONE_27B[$seed]=1
        fi
    done

    # Check 8B
    for seed in "${SEEDS_8B[@]}"; do
        [[ -v DONE_8B[$seed] ]] && continue
        if grep -q "DONE.*ifbench/synaptic_pruning/seed=${seed}" "$LOG8" 2>/dev/null; then
            copy_result qwen3-8b "$seed"
            DONE_8B[$seed]=1
        fi
    done

    # Check 14B
    for seed in "${SEEDS_14B[@]}"; do
        [[ -v DONE_14B[$seed] ]] && continue
        if grep -q "DONE.*ifbench/synaptic_pruning/seed=${seed}" "$LOG14" 2>/dev/null; then
            copy_result qwen3-14b "$seed"
            DONE_14B[$seed]=1
        fi
    done

    remaining_27b=$(( ${#SEEDS_27B[@]} - ${#DONE_27B[@]} ))
    remaining_8b=$(( ${#SEEDS_8B[@]} - ${#DONE_8B[@]} ))
    remaining_14b=$(( ${#SEEDS_14B[@]} - ${#DONE_14B[@]} ))

    if [[ $remaining_27b -eq 0 && $remaining_8b -eq 0 && $remaining_14b -eq 0 ]]; then
        echo "[$(date '+%T')] All in-flight rescues complete!"
        break
    fi

    echo "[$(date '+%T')] Waiting: 27B=${remaining_27b} 8B=${remaining_8b} 14B=${remaining_14b} remaining"
done

echo "Final counts:"
echo "  qwen3-27b-awq/ifbench/synaptic_pruning: $(find $RUNS/qwen3-27b-awq/ifbench/synaptic_pruning -name result.json 2>/dev/null | wc -l)"
echo "  qwen3-8b/ifbench/synaptic_pruning:      $(find $RUNS/qwen3-8b/ifbench/synaptic_pruning -name result.json 2>/dev/null | wc -l)"
echo "  qwen3-14b/ifbench/synaptic_pruning:     $(find $RUNS/qwen3-14b/ifbench/synaptic_pruning -name result.json 2>/dev/null | wc -l)"
echo ""
echo "NOTE: 27B seeds 42/789/123/456 were overwritten before recovery. They'll rerun"
echo "      automatically on next 27B orchestrator restart (result paths missing)."
