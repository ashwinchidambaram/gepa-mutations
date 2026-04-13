#!/bin/bash
# Quick progress check for RunPod sweep
# Usage: bash scripts/runpod_progress.sh

echo "=== Sweep Progress ==="
echo ""

# Result counts by model
python3 -c "
import glob, json, collections

c = collections.Counter()
for f in glob.glob('runs/qwen3-*/*/*/*/result.json'):
    parts = f.split('/')
    model, bench, method = parts[1], parts[2], parts[3]
    c[(model,)] += 1
    c[(model, bench)] += 1

models = sorted(set(k[0] for k in c if len(k) == 1))
benchmarks = ['ifbench','pupa','livebench','hotpotqa','hover','aime']
methods_target = 6  # gepa + 5 mutations
seeds_target = 5

for model in models:
    total = c[(model,)]
    target = len(benchmarks) * methods_target * seeds_target
    pct = total / target * 100 if target else 0
    print(f'{model}: {total}/{target} ({pct:.0f}%)')
    for bench in benchmarks:
        count = c.get((model, bench), 0)
        bt = methods_target * seeds_target
        print(f'  {bench:<12} {count}/{bt}')
    print()

grand = sum(c[k] for k in c if len(k) == 1)
print(f'Grand total: {grand}')
"

# Check running processes
echo ""
echo "--- Running Processes ---"
echo "vLLM servers:"
for port in 8125 8126 8127; do
    if curl -sf http://localhost:$port/v1/models > /dev/null 2>&1; then
        model=$(curl -s http://localhost:$port/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
        echo "  :$port $model"
    fi
done

echo "Orchestrators:"
ps aux | grep "run_all_local" | grep -v grep | awk '{print "  PID " $2 " — " $0}' | cut -c1-100 || echo "  (none running)"

# Check for failures in recent logs
echo ""
echo "--- Recent Failures (last 10) ---"
grep -h "FAILED\|TIMEOUT\|ERROR" logs/orchestrator_*_runpod.log 2>/dev/null | tail -10 || echo "  (none)"
