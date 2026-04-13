# gepa-mutations — Mac Instance

This is the **`sweep/mac` branch**. This Claude Code instance runs on a MacBook Pro M5 Max
(48GB unified memory) and handles all **small and cross-architecture models** via **MLX-LM**.

Large Qwen3 models (8B, 14B, 27B-AWQ) run independently on the SLURM cluster (`sweep/cluster`).
This instance operates completely autonomously — no coordination needed.

---

## Models (10 total)

### Qwen3 — scaling curve extension
| Model | Port | Memory | MLX model ID |
|-------|------|--------|-------------|
| Qwen3-0.6B | 8132 | ~0.4 GB | `mlx-community/Qwen3-0.6B-4bit` |
| Qwen3-1.7B | 8125 | ~1 GB | `mlx-community/Qwen3-1.7B-4bit` |
| Qwen3-4B | 8126 | ~3 GB | `mlx-community/Qwen3-4B-4bit` |
| Qwen3-32B | 8131 | ~20 GB | `mlx-community/Qwen3-32B-4bit` |

### Gemma 3 — cross-architecture (Google)
| Model | Port | Memory | MLX model ID |
|-------|------|--------|-------------|
| Gemma 3 1B | 8133 | ~0.6 GB | `mlx-community/gemma-3-1b-it-4bit` |
| Gemma 3 4B | 8134 | ~2.5 GB | `mlx-community/gemma-3-4b-it-4bit` |
| Gemma 3 12B | 8135 | ~7 GB | `mlx-community/gemma-3-12b-it-4bit` |
| Gemma 3 27B | 8136 | ~15 GB | `mlx-community/gemma-3-27b-it-4bit` |

### Llama 3.2 — cross-architecture (Meta)
| Model | Port | Memory | MLX model ID |
|-------|------|--------|-------------|
| Llama 3.2 1B | 8137 | ~0.6 GB | `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| Llama 3.2 3B | 8138 | ~2 GB | `mlx-community/Llama-3.2-3B-Instruct-4bit` |

**Combined scaling curve (Mac + cluster):** 0.6B → 1.7B → 4B → 8B → 12B → 14B → 27B → 32B
plus Gemma 3 and Llama 3.2 for cross-architecture validation.

---

## Prerequisites (first-time setup)

```bash
# 1. Clone repo and check out this branch
git clone https://github.com/ashwinchidambaram/gepa-mutations.git
cd gepa-mutations
git checkout sweep/mac

# 2. Install Python dependencies
pip install uv
uv sync

# 3. Install MLX-LM (Apple Silicon only)
pip install mlx-lm

# 4. Create .env with Telegram credentials
cp .env.example .env
# edit .env: add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID

# 5. Pre-download all models (do once on a good connection)
python -c "
from mlx_lm import load
models = [
    'mlx-community/Qwen3-0.6B-4bit',
    'mlx-community/Qwen3-1.7B-4bit',
    'mlx-community/Qwen3-4B-4bit',
    'mlx-community/Qwen3-32B-4bit',
    'mlx-community/gemma-3-1b-it-4bit',
    'mlx-community/gemma-3-4b-it-4bit',
    'mlx-community/gemma-3-12b-it-4bit',
    'mlx-community/gemma-3-27b-it-4bit',
    'mlx-community/Llama-3.2-1B-Instruct-4bit',
    'mlx-community/Llama-3.2-3B-Instruct-4bit',
]
for m in models:
    print(f'Downloading {m}...')
    load(m)
    print('  done')
"
```

---

## Step 1: Start Inference Servers

48GB fits all small models simultaneously (~14GB) or the large group together (~42GB).
See [CLAUDE/mac_setup.md](CLAUDE/mac_setup.md) for recommended memory groupings.

```bash
mkdir -p logs

# Small models — run all at once (~14GB total)
bash scripts/serve_mlx_0.6b_mac.sh   &> logs/mlx_0.6b.log   &
bash scripts/serve_mlx_1b_mac.sh     &> logs/mlx_1b.log     &
bash scripts/serve_mlx_4b_mac.sh     &> logs/mlx_4b.log     &
bash scripts/serve_mlx_gemma3_1b_mac.sh  &> logs/mlx_gemma3_1b.log  &
bash scripts/serve_mlx_gemma3_4b_mac.sh  &> logs/mlx_gemma3_4b.log  &
bash scripts/serve_mlx_llama_1b_mac.sh   &> logs/mlx_llama_1b.log   &
bash scripts/serve_mlx_llama_3b_mac.sh   &> logs/mlx_llama_3b.log   &

# Large models — run together or individually (~42GB if all three at once)
bash scripts/serve_mlx_32b_mac.sh        &> logs/mlx_32b.log        &
bash scripts/serve_mlx_gemma3_12b_mac.sh &> logs/mlx_gemma3_12b.log &
bash scripts/serve_mlx_gemma3_27b_mac.sh &> logs/mlx_gemma3_27b.log &
```

Verify:
```bash
for port in 8125 8126 8131 8132 8133 8134 8135 8136 8137 8138; do
  curl -sf http://localhost:$port/v1/models > /dev/null && echo "port $port OK" || echo "port $port DOWN"
done
```

---

## Step 2: Smoke Test

```bash
declare -A MODELS
MODELS["mlx-community/Qwen3-0.6B-4bit"]=8132
MODELS["mlx-community/Qwen3-1.7B-4bit"]=8125
MODELS["mlx-community/Qwen3-4B-4bit"]=8126
MODELS["mlx-community/Qwen3-32B-4bit"]=8131
MODELS["mlx-community/gemma-3-1b-it-4bit"]=8133
MODELS["mlx-community/gemma-3-4b-it-4bit"]=8134
MODELS["mlx-community/gemma-3-12b-it-4bit"]=8135
MODELS["mlx-community/gemma-3-27b-it-4bit"]=8136
MODELS["mlx-community/Llama-3.2-1B-Instruct-4bit"]=8137
MODELS["mlx-community/Llama-3.2-3B-Instruct-4bit"]=8138

for model in "${!MODELS[@]}"; do
  port="${MODELS[$model]}"
  export GEPA_MODEL="$model" GEPA_BASE_URL="http://localhost:$port/v1"
  .venv/bin/python scripts/run_all_local.py --smoke-test --workers 3 \
    --benchmark hotpotqa pupa ifbench
done
```

---

## Step 3: Full Sweep

300 experiments per model (5 benchmarks × 12 methods × 5 seeds).

```bash
# Helper — launch one orchestrator in the background
launch() {
  local model=$1 port=$2 tag=$3 workers=$4
  export GEPA_MODEL="$model" GEPA_BASE_URL="http://localhost:$port/v1"
  nohup .venv/bin/python scripts/run_all_local.py --workers $workers \
    --benchmark hotpotqa hover pupa ifbench livebench \
    &> logs/orchestrator_${tag}.log &
  echo "Launched $tag (PID $!)"
}

# Qwen3
launch mlx-community/Qwen3-0.6B-4bit  8132 0.6b  8
launch mlx-community/Qwen3-1.7B-4bit  8125 1b    6
launch mlx-community/Qwen3-4B-4bit    8126 4b    6
launch mlx-community/Qwen3-32B-4bit   8131 32b   3

# Gemma 3
launch mlx-community/gemma-3-1b-it-4bit   8133 gemma3_1b   8
launch mlx-community/gemma-3-4b-it-4bit   8134 gemma3_4b   6
launch mlx-community/gemma-3-12b-it-4bit  8135 gemma3_12b  4
launch mlx-community/gemma-3-27b-it-4bit  8136 gemma3_27b  3

# Llama 3.2
launch mlx-community/Llama-3.2-1B-Instruct-4bit 8137 llama_1b 8
launch mlx-community/Llama-3.2-3B-Instruct-4bit 8138 llama_3b 6
```

---

## Monitoring

```bash
# Progress counts
for tag in qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-32b \
           gemma3-1b gemma3-4b gemma3-12b gemma3-27b \
           llama3-1b llama3-3b; do
  count=$(find runs/$tag -name result.json 2>/dev/null | wc -l | tr -d ' ')
  echo "$tag: $count / 300"
done

# Telegram update
.venv/bin/python scripts/monitor_multi_model.py --mode 30min

# Live log
tail -f logs/orchestrator_gemma3_27b.log
```

---

## Reference Docs

- [CLAUDE/mac_setup.md](CLAUDE/mac_setup.md) — model IDs, memory groupings, serve script flags
- [CLAUDE/sweep_execution.md](CLAUDE/sweep_execution.md) — orchestrator reference, run directory layout
- [CLAUDE/known_bugs_and_fixes.md](CLAUDE/known_bugs_and_fixes.md) — gepa state fix, etc.

## Conventions

- Paper hyperparameters: `temp=0.6`, `top_p=0.95`, `top_k=20`, `minibatch=3`
- AIME excluded from active sweep
- Always smoke test before full sweeps
- gepa submodule is patched — do NOT pull upstream without re-applying the makedirs fix
