#!/bin/bash
# Master launch script for Mac Group A sweep.
# Starts 7 MLX-LM inference servers with a 15-minute thermal stagger,
# launches one orchestrator per model, and runs automated monitoring.
#
# Usage: bash scripts/launch_mac_sweep.sh

set -e
cd "$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p logs
> logs/sweep_pids.txt

# ---------------------------------------------------------------------------
# Prevent sleep for the duration of the sweep
# ---------------------------------------------------------------------------
caffeinate -d -i -m -s &
echo $! >> logs/sweep_pids.txt
echo "caffeinate started (PID $(tail -1 logs/sweep_pids.txt))"

# ---------------------------------------------------------------------------
# Initial Telegram notification
# ---------------------------------------------------------------------------
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from gepa_mutations.config import Settings
import asyncio, telegram
s = Settings()
bot = telegram.Bot(token=s.telegram_bot_token)
asyncio.run(bot.send_message(
    chat_id=s.telegram_chat_id,
    text='🚀 <b>Mac Group A sweep launching</b>\n<code>7 models · 2100 experiments · 15-min stagger</code>\nMonitoring active every 15/30 min.',
    parse_mode='HTML'
))
" 2>/dev/null || echo "Telegram notify failed (non-fatal)"

# ---------------------------------------------------------------------------
# Helper: wait for MLX-LM server to be ready
# ---------------------------------------------------------------------------
wait_for_server() {
    local port=$1
    local name=$2
    echo "  Waiting for ${name} on port ${port}..."
    for i in $(seq 1 60); do
        if curl -sf --max-time 3 "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "  ${name} ready!"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: ${name} failed to start on port ${port} after 5 minutes"
    exit 1
}

# ---------------------------------------------------------------------------
# Monitoring loop (single loop — sequential to avoid snapshot conflicts)
# ---------------------------------------------------------------------------
monitor_loop() {
    local tick=0
    while true; do
        sleep 900
        tick=$((tick + 1))
        .venv/bin/python scripts/monitor_multi_model.py --mode 15min 2>/dev/null || true
        if [ $((tick % 2)) -eq 0 ]; then
            sleep 10
            .venv/bin/python scripts/monitor_multi_model.py --mode 30min 2>/dev/null || true
        fi
    done
}

# ---------------------------------------------------------------------------
# Model 1: Qwen3-0.6B (port 8132)
# ---------------------------------------------------------------------------
echo ""
echo "Starting Qwen3-0.6B server..."
bash scripts/serve_mlx_0.6b_mac.sh &> logs/mlx_0.6b.log &
echo $! >> logs/sweep_pids.txt
wait_for_server 8132 "Qwen3-0.6B"

echo "Launching Qwen3-0.6B orchestrator (8 workers)..."
export GEPA_MODEL="mlx-community/Qwen3-0.6B-4bit" GEPA_BASE_URL="http://localhost:8132/v1" MODEL_PREFIX="openai" DSPY_CACHEDIR="$HOME/.dspy_cache/qwen3-0.6b"
nohup .venv/bin/python scripts/run_all_local.py \
    --workers 8 \
    --benchmark hotpotqa hover pupa ifbench livebench \
    &> logs/orchestrator_0.6b.log &
echo $! >> logs/sweep_pids.txt
echo "  Qwen3-0.6B orchestrator running. Waiting 15 min before next server..."
sleep 900

# ---------------------------------------------------------------------------
# Model 2: Qwen3-1.7B (port 8125)
# ---------------------------------------------------------------------------
echo ""
echo "Starting Qwen3-1.7B server..."
bash scripts/serve_mlx_1b_mac.sh &> logs/mlx_1b.log &
echo $! >> logs/sweep_pids.txt
wait_for_server 8125 "Qwen3-1.7B"

echo "Launching Qwen3-1.7B orchestrator (6 workers)..."
export GEPA_MODEL="mlx-community/Qwen3-1.7B-4bit" GEPA_BASE_URL="http://localhost:8125/v1" MODEL_PREFIX="openai" DSPY_CACHEDIR="$HOME/.dspy_cache/qwen3-1.7b"
nohup .venv/bin/python scripts/run_all_local.py \
    --workers 6 \
    --benchmark hotpotqa hover pupa ifbench livebench \
    &> logs/orchestrator_1b.log &
echo $! >> logs/sweep_pids.txt
echo "  Qwen3-1.7B orchestrator running. Waiting 15 min before next server..."
sleep 900

# ---------------------------------------------------------------------------
# Model 3: Qwen3-4B (port 8126)
# ---------------------------------------------------------------------------
echo ""
echo "Starting Qwen3-4B server..."
bash scripts/serve_mlx_4b_mac.sh &> logs/mlx_4b.log &
echo $! >> logs/sweep_pids.txt
wait_for_server 8126 "Qwen3-4B"

echo "Launching Qwen3-4B orchestrator (6 workers)..."
export GEPA_MODEL="mlx-community/Qwen3-4B-4bit" GEPA_BASE_URL="http://localhost:8126/v1" MODEL_PREFIX="openai" DSPY_CACHEDIR="$HOME/.dspy_cache/qwen3-4b"
nohup .venv/bin/python scripts/run_all_local.py \
    --workers 6 \
    --benchmark hotpotqa hover pupa ifbench livebench \
    &> logs/orchestrator_4b.log &
echo $! >> logs/sweep_pids.txt
echo "  Qwen3-4B orchestrator running. Waiting 15 min before next server..."
sleep 900

# ---------------------------------------------------------------------------
# Model 4: Gemma3-1B (port 8133)
# ---------------------------------------------------------------------------
echo ""
echo "Starting Gemma3-1B server..."
bash scripts/serve_mlx_gemma3_1b_mac.sh &> logs/mlx_gemma3_1b.log &
echo $! >> logs/sweep_pids.txt
wait_for_server 8133 "Gemma3-1B"

echo "Launching Gemma3-1B orchestrator (8 workers)..."
export GEPA_MODEL="mlx-community/gemma-3-1b-it-4bit" GEPA_BASE_URL="http://localhost:8133/v1" MODEL_PREFIX="openai" DSPY_CACHEDIR="$HOME/.dspy_cache/gemma3-1b"
nohup .venv/bin/python scripts/run_all_local.py \
    --workers 8 \
    --benchmark hotpotqa hover pupa ifbench livebench \
    &> logs/orchestrator_gemma3_1b.log &
echo $! >> logs/sweep_pids.txt
echo "  Gemma3-1B orchestrator running. Waiting 15 min before next server..."
sleep 900

# ---------------------------------------------------------------------------
# Model 5: Gemma3-4B (port 8134)
# ---------------------------------------------------------------------------
echo ""
echo "Starting Gemma3-4B server..."
bash scripts/serve_mlx_gemma3_4b_mac.sh &> logs/mlx_gemma3_4b.log &
echo $! >> logs/sweep_pids.txt
wait_for_server 8134 "Gemma3-4B"

echo "Launching Gemma3-4B orchestrator (6 workers)..."
export GEPA_MODEL="mlx-community/gemma-3-4b-it-4bit" GEPA_BASE_URL="http://localhost:8134/v1" MODEL_PREFIX="openai" DSPY_CACHEDIR="$HOME/.dspy_cache/gemma3-4b"
nohup .venv/bin/python scripts/run_all_local.py \
    --workers 6 \
    --benchmark hotpotqa hover pupa ifbench livebench \
    &> logs/orchestrator_gemma3_4b.log &
echo $! >> logs/sweep_pids.txt
echo "  Gemma3-4B orchestrator running. Waiting 15 min before next server..."
sleep 900

# ---------------------------------------------------------------------------
# Model 6: Llama3.2-1B (port 8137)
# ---------------------------------------------------------------------------
echo ""
echo "Starting Llama3.2-1B server..."
bash scripts/serve_mlx_llama_1b_mac.sh &> logs/mlx_llama_1b.log &
echo $! >> logs/sweep_pids.txt
wait_for_server 8137 "Llama3.2-1B"

echo "Launching Llama3.2-1B orchestrator (8 workers)..."
export GEPA_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit" GEPA_BASE_URL="http://localhost:8137/v1" MODEL_PREFIX="openai" DSPY_CACHEDIR="$HOME/.dspy_cache/llama3-1b"
nohup .venv/bin/python scripts/run_all_local.py \
    --workers 8 \
    --benchmark hotpotqa hover pupa ifbench livebench \
    &> logs/orchestrator_llama_1b.log &
echo $! >> logs/sweep_pids.txt
echo "  Llama3.2-1B orchestrator running. Waiting 15 min before next server..."
sleep 900

# ---------------------------------------------------------------------------
# Model 7: Llama3.2-3B (port 8138) — no sleep after this one
# ---------------------------------------------------------------------------
echo ""
echo "Starting Llama3.2-3B server..."
bash scripts/serve_mlx_llama_3b_mac.sh &> logs/mlx_llama_3b.log &
echo $! >> logs/sweep_pids.txt
wait_for_server 8138 "Llama3.2-3B"

echo "Launching Llama3.2-3B orchestrator (6 workers)..."
export GEPA_MODEL="mlx-community/Llama-3.2-3B-Instruct-4bit" GEPA_BASE_URL="http://localhost:8138/v1" MODEL_PREFIX="openai" DSPY_CACHEDIR="$HOME/.dspy_cache/llama3-3b"
nohup .venv/bin/python scripts/run_all_local.py \
    --workers 6 \
    --benchmark hotpotqa hover pupa ifbench livebench \
    &> logs/orchestrator_llama_3b.log &
echo $! >> logs/sweep_pids.txt

# ---------------------------------------------------------------------------
# Start monitoring loop
# ---------------------------------------------------------------------------
echo ""
echo "All 7 models running. Starting monitoring loop..."
monitor_loop &
echo $! >> logs/sweep_pids.txt

echo ""
echo "All systems go. Monitoring active every 15/30 min."
echo "PIDs saved to logs/sweep_pids.txt"
echo "To stop: bash scripts/stop_mac_sweep.sh"
