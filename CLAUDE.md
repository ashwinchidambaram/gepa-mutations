# gepa-mutations — Mac Instance

This is the **`sweep/mac` branch**. This Claude Code instance runs on a MacBook Pro M5 Max
(48GB unified memory) and handles the **small models + a new large model**: Qwen3-1.7B,
Qwen3-4B, and Qwen3-32B, all served locally via **MLX-LM**.

Large models (8B, 14B, 27B-AWQ) run independently on the SLURM cluster (`sweep/cluster`
branch). This instance operates completely autonomously — no coordination needed.

**Scaling curve:** **1.7B, 4B, 32B → this Mac** | 8B, 14B, 27B-AWQ → cluster

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

# 4. Create .env with Telegram credentials (optional but recommended)
cp .env.example .env
# edit .env: add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID

# 5. Pre-download models (do this while you have a good connection)
python -c "from mlx_lm import load; load('mlx-community/Qwen3-1.7B-4bit')"
python -c "from mlx_lm import load; load('mlx-community/Qwen3-4B-4bit')"
python -c "from mlx_lm import load; load('mlx-community/Qwen3-32B-4bit')"
```

---

## Active Models

| Model | MLX model ID | Port | Memory (~) |
|-------|-------------|------|------------|
| Qwen3-1.7B | `mlx-community/Qwen3-1.7B-4bit` | 8125 | ~1 GB |
| Qwen3-4B | `mlx-community/Qwen3-4B-4bit` | 8126 | ~3 GB |
| Qwen3-32B | `mlx-community/Qwen3-32B-4bit` | 8131 | ~20 GB |

All three fit simultaneously in 48GB with ~24GB to spare for the OS and other processes.

---

## Step 1: Start Inference Servers

Open three terminal tabs (or use tmux). Each server runs in the foreground:

```bash
# Terminal 1 — Qwen3-1.7B
bash scripts/serve_mlx_1b_mac.sh

# Terminal 2 — Qwen3-4B
bash scripts/serve_mlx_4b_mac.sh

# Terminal 3 — Qwen3-32B (loads ~20GB; takes ~30s to be ready)
bash scripts/serve_mlx_32b_mac.sh
```

Or run all three in the background:
```bash
bash scripts/serve_mlx_1b_mac.sh  &> logs/mlx_1b.log &
bash scripts/serve_mlx_4b_mac.sh  &> logs/mlx_4b.log &
bash scripts/serve_mlx_32b_mac.sh &> logs/mlx_32b.log &
```

Verify servers are up:
```bash
curl -sf http://localhost:8125/v1/models && echo "1.7B OK"
curl -sf http://localhost:8126/v1/models && echo "4B OK"
curl -sf http://localhost:8131/v1/models && echo "32B OK"
```

---

## Step 2: Smoke Test

Always smoke-test before a full sweep:

```bash
export GEPA_MODEL="mlx-community/Qwen3-1.7B-4bit" GEPA_BASE_URL="http://localhost:8125/v1"
.venv/bin/python scripts/run_all_local.py --smoke-test --workers 3 \
  --benchmark hotpotqa pupa ifbench

export GEPA_MODEL="mlx-community/Qwen3-4B-4bit" GEPA_BASE_URL="http://localhost:8126/v1"
.venv/bin/python scripts/run_all_local.py --smoke-test --workers 3 \
  --benchmark hotpotqa pupa ifbench

export GEPA_MODEL="mlx-community/Qwen3-32B-4bit" GEPA_BASE_URL="http://localhost:8131/v1"
.venv/bin/python scripts/run_all_local.py --smoke-test --workers 2 \
  --benchmark hotpotqa pupa ifbench
```

---

## Step 3: Full Sweep

300 experiments per model (5 benchmarks × 12 methods × 5 seeds). Run all three in parallel:

```bash
mkdir -p logs

# Qwen3-1.7B
export GEPA_MODEL="mlx-community/Qwen3-1.7B-4bit" GEPA_BASE_URL="http://localhost:8125/v1"
nohup .venv/bin/python scripts/run_all_local.py --workers 6 \
  --benchmark hotpotqa hover pupa ifbench livebench \
  &> logs/orchestrator_1b.log &

# Qwen3-4B
export GEPA_MODEL="mlx-community/Qwen3-4B-4bit" GEPA_BASE_URL="http://localhost:8126/v1"
nohup .venv/bin/python scripts/run_all_local.py --workers 6 \
  --benchmark hotpotqa hover pupa ifbench livebench \
  &> logs/orchestrator_4b.log &

# Qwen3-32B (fewer workers — 32B is slower per call)
export GEPA_MODEL="mlx-community/Qwen3-32B-4bit" GEPA_BASE_URL="http://localhost:8131/v1"
nohup .venv/bin/python scripts/run_all_local.py --workers 3 \
  --benchmark hotpotqa hover pupa ifbench livebench \
  &> logs/orchestrator_32b.log &
```

The orchestrator is idempotent — safe to restart, skips completed experiments.

---

## Monitoring

No automated crons needed — check progress manually:

```bash
# Check how many results exist per model
find runs/qwen3-1.7b -name result.json | wc -l
find runs/qwen3-4b -name result.json | wc -l
find runs/qwen3-32b -name result.json | wc -l

# Watch orchestrator logs live
tail -f logs/orchestrator_32b.log

# Send a manual Telegram update
.venv/bin/python scripts/monitor_multi_model.py --mode 30min
```

---

## Results

Results land in `runs/` (local, gitignored):
```
runs/qwen3-1.7b/<benchmark>/<method>/<seed>/result.json
runs/qwen3-4b/...
runs/qwen3-32b/...
```

`result.json` key fields: `test_score`, `train_scores`, `elapsed`.

---

## Known Issues / MLX Notes

- **Mac-specific:** MLX-LM does not support `--enforce-eager` or `--dtype auto` flags — those are vLLM-only. The serve scripts use only MLX-compatible flags.
- **Thinking mode:** Qwen3 models have a thinking mode. MLX-LM serves them without special flags; the orchestrator sends standard chat completions.
- **Concurrency:** MLX-LM server handles concurrent requests but is single-process. Keep workers ≤ 6 for 1.7B/4B, ≤ 3 for 32B to avoid OOM.
- **gepa submodule patch:** `gepa/src/gepa/core/state.py` has an `os.makedirs` patch — do NOT pull submodule upstream without re-applying it. See [CLAUDE/known_bugs_and_fixes.md](CLAUDE/known_bugs_and_fixes.md).

## Reference Docs

- [CLAUDE/mac_setup.md](CLAUDE/mac_setup.md) — MLX setup, model IDs, memory requirements, serve script flags
- [CLAUDE/sweep_execution.md](CLAUDE/sweep_execution.md) — full orchestrator reference (env vars, run directory layout, execution order)
- [CLAUDE/known_bugs_and_fixes.md](CLAUDE/known_bugs_and_fixes.md) — gepa state fix, _env_model_tag collision, etc.

## Conventions

- Paper hyperparameters: `temp=0.6`, `top_p=0.95`, `top_k=20`, `minibatch=3`
- Paper baseline scores in `src/gepa_mutations/config.py`
- AIME excluded from active sweep
- Always smoke test before full sweeps
