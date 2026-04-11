# gepa-mutations

Two Claude Code instances share this repo on separate branches:

| Instance | Branch | Hardware | Models |
|----------|--------|----------|--------|
| **Cluster** | `sweep/cluster` | SLURM (vLLM) | Qwen3-8B, 14B, 27B-AWQ |
| **Mac** | `sweep/mac` | MacBook M5 Max 48GB (MLX-LM) | Qwen3-0.6B→32B, Gemma3 1B→27B, Llama3.2 1B/3B |

**Combined scaling curve:** 0.6B → 1.7B → 4B → 8B → 12B → 14B → 27B → 32B  
plus Gemma 3 and Llama 3.2 for cross-architecture validation.

---

## Cluster Models (vLLM)

| Model | Node | GPU | Port |
|-------|------|-----|------|
| Qwen3-27B-AWQ | manifold | RTX 5090 32GB | 8124 |
| Qwen3-8B | archimedes | RTX 5090 32GB | 8125 |
| Qwen3-14B | kolmogorov | RTX 4090 24GB | 8128 |

## Mac Models (MLX-LM, localhost)

| Model | Port | Memory |
|-------|------|--------|
| Qwen3-0.6B | 8132 | ~0.4 GB |
| Qwen3-1.7B | 8125 | ~1 GB |
| Qwen3-4B | 8126 | ~3 GB |
| Qwen3-32B | 8131 | ~20 GB |
| Gemma3-1B | 8133 | ~0.6 GB |
| Gemma3-4B | 8134 | ~2.5 GB |
| Gemma3-12B | 8135 | ~7 GB |
| Gemma3-27B | 8136 | ~15 GB |
| Llama3.2-1B | 8137 | ~0.6 GB |
| Llama3.2-3B | 8138 | ~2 GB |

---

## Step 1: Start Inference Servers

**Cluster:**
```bash
# 27B-AWQ on manifold — chain jobs for 24h coverage (capstone = 8h max)
J1=$(sbatch scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
J2=$(sbatch --dependency=afterany:$J1 scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
J3=$(sbatch --dependency=afterany:$J2 scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')

# 8B on archimedes (ray-cluster = 90d)
sbatch scripts/serve_vllm_8b.sh

# 14B on kolmogorov — chain jobs for 16h (student-gpu = 2h max)
J1=$(sbatch scripts/serve_vllm_14b_kolmogorov.sh | awk '{print $4}')
J2=$(sbatch --dependency=afterany:$J1 scripts/serve_vllm_14b_kolmogorov.sh | awk '{print $4}')
```

Verify cluster servers:
```bash
curl -sf http://10.0.10.69:8124/v1/models && echo "27B OK"
curl -sf http://10.0.10.58:8125/v1/models && echo "8B OK"
curl -sf http://10.0.10.52:8128/v1/models && echo "14B OK"
```

**Mac** (48GB fits all small models simultaneously, ~14GB):
```bash
mkdir -p logs
bash scripts/serve_mlx_0.6b_mac.sh       &> logs/mlx_0.6b.log       &
bash scripts/serve_mlx_1b_mac.sh         &> logs/mlx_1b.log         &
bash scripts/serve_mlx_4b_mac.sh         &> logs/mlx_4b.log         &
bash scripts/serve_mlx_gemma3_1b_mac.sh  &> logs/mlx_gemma3_1b.log  &
bash scripts/serve_mlx_gemma3_4b_mac.sh  &> logs/mlx_gemma3_4b.log  &
bash scripts/serve_mlx_llama_1b_mac.sh   &> logs/mlx_llama_1b.log   &
bash scripts/serve_mlx_llama_3b_mac.sh   &> logs/mlx_llama_3b.log   &
# Large models — run together or individually (~42GB combined)
bash scripts/serve_mlx_32b_mac.sh        &> logs/mlx_32b.log        &
bash scripts/serve_mlx_gemma3_12b_mac.sh &> logs/mlx_gemma3_12b.log &
bash scripts/serve_mlx_gemma3_27b_mac.sh &> logs/mlx_gemma3_27b.log &
```

Verify Mac servers:
```bash
for port in 8125 8126 8131 8132 8133 8134 8135 8136 8137 8138; do
  curl -sf http://localhost:$port/v1/models > /dev/null && echo "port $port OK" || echo "port $port DOWN"
done
```

---

## Step 2: Smoke Test

Always smoke-test before a full sweep. Get explicit go/no-go before launching.

**Cluster:**
```bash
for MODEL_ARG in "Qwen/Qwen3-27B-AWQ|http://10.0.10.69:8124/v1" \
                 "Qwen/Qwen3-8B|http://10.0.10.58:8125/v1" \
                 "Qwen/Qwen3-14B|http://10.0.10.52:8128/v1"; do
  MODEL=$(echo $MODEL_ARG | cut -d'|' -f1)
  URL=$(echo $MODEL_ARG | cut -d'|' -f2)
  export GEPA_MODEL="$MODEL" GEPA_BASE_URL="$URL"
  .venv/bin/python scripts/run_all_local.py --smoke-test --workers 4 \
    --benchmark hotpotqa pupa ifbench
done
```

**Mac:**
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

**Cluster:**
```bash
# 27B-AWQ
export GEPA_MODEL="Qwen/Qwen3-27B-AWQ" GEPA_BASE_URL="http://10.0.10.69:8124/v1"
nohup .venv/bin/python scripts/run_all_local.py --workers 8 \
  --benchmark hotpotqa hover pupa ifbench livebench \
  &> logs/orchestrator_27b.log &

# 8B
export GEPA_MODEL="Qwen/Qwen3-8B" GEPA_BASE_URL="http://10.0.10.58:8125/v1"
nohup .venv/bin/python scripts/run_all_local.py --workers 10 \
  --benchmark hotpotqa hover pupa ifbench livebench \
  &> logs/orchestrator_8b.log &

# 14B
export GEPA_MODEL="Qwen/Qwen3-14B" GEPA_BASE_URL="http://10.0.10.52:8128/v1"
nohup .venv/bin/python scripts/run_all_local.py --workers 6 \
  --benchmark hotpotqa hover pupa ifbench livebench \
  &> logs/orchestrator_14b.log &
```

**Mac:**
```bash
launch() {
  local model=$1 port=$2 tag=$3 workers=$4
  export GEPA_MODEL="$model" GEPA_BASE_URL="http://localhost:$port/v1"
  nohup .venv/bin/python scripts/run_all_local.py --workers $workers \
    --benchmark hotpotqa hover pupa ifbench livebench \
    &> logs/orchestrator_${tag}.log &
  echo "Launched $tag (PID $!)"
}

launch mlx-community/Qwen3-0.6B-4bit              8132 0.6b        8
launch mlx-community/Qwen3-1.7B-4bit              8125 1b          6
launch mlx-community/Qwen3-4B-4bit                8126 4b          6
launch mlx-community/Qwen3-32B-4bit               8131 32b         3
launch mlx-community/gemma-3-1b-it-4bit           8133 gemma3_1b   8
launch mlx-community/gemma-3-4b-it-4bit           8134 gemma3_4b   6
launch mlx-community/gemma-3-12b-it-4bit          8135 gemma3_12b  4
launch mlx-community/gemma-3-27b-it-4bit          8136 gemma3_27b  3
launch mlx-community/Llama-3.2-1B-Instruct-4bit   8137 llama_1b    8
launch mlx-community/Llama-3.2-3B-Instruct-4bit   8138 llama_3b    6
```

The orchestrator is idempotent — safe to restart, skips completed experiments.

---

## Monitoring

```bash
# Progress counts
for tag in qwen3-27b-awq qwen3-8b qwen3-14b \
           qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-32b \
           gemma3-1b gemma3-4b gemma3-12b gemma3-27b \
           llama3-1b llama3-3b; do
  count=$(find runs/$tag -name result.json 2>/dev/null | wc -l | tr -d ' ')
  echo "$tag: $count / 300"
done

# Telegram update
.venv/bin/python scripts/monitor_multi_model.py --mode 30min

# Cluster GPU load
curl -s http://10.0.10.58:8125/metrics | grep num_requests_running

# Live log (example)
tail -f logs/orchestrator_8b.log
```

---

## Results

```
runs/{model_tag}/{benchmark}/{method}/{seed}/result.json
```

`result.json` key fields: `test_score`, `train_scores`, `elapsed`.

---

## Reference Docs

- [CLAUDE/cluster_infrastructure.md](CLAUDE/cluster_infrastructure.md) — nodes, IPs, ports, GPU specs, partition limits, vLLM env
- [CLAUDE/mac_setup.md](CLAUDE/mac_setup.md) — MLX model IDs, memory groupings, serve script flags
- [CLAUDE/sweep_execution.md](CLAUDE/sweep_execution.md) — orchestrator reference, run directory layout
- [CLAUDE/monitoring.md](CLAUDE/monitoring.md) — cron setup, Telegram, health checks
- [CLAUDE/known_bugs_and_fixes.md](CLAUDE/known_bugs_and_fixes.md) — gepa state fix, vLLM IPC bug, tag collision fix

## Conventions

- Paper hyperparameters: `temp=0.6`, `top_p=0.95`, `top_k=20`, `minibatch=3`
- Paper baseline scores in `src/gepa_mutations/config.py`
- AIME excluded from active sweep
- Always smoke test before full sweeps; get explicit go/no-go before launching
- gepa submodule is patched — do NOT pull upstream without re-applying the makedirs fix
