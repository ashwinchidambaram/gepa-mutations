# gepa-mutations — Cluster Instance

This is the **`sweep/cluster` branch**. This Claude Code instance operates the SLURM cluster
and runs experiments for the **large models only**: Qwen3-8B, Qwen3-14B, Qwen3-27B-AWQ.

Small models (1.7B, 4B) have been retired from the cluster and moved to the Mac (`sweep/mac`
branch). A separate Claude Code instance on the Mac handles those independently via MLX-LM.
The Mac also runs Qwen3-32B — a new size not covered here.

**Scaling curve:** 1.7B, 4B, 32B → Mac | **8B, 14B, 27B-AWQ → this cluster**

---

## Active Models

| Model | Node | GPU | Port | Partition |
|-------|------|-----|------|-----------|
| Qwen3-27B-AWQ | manifold | RTX 5090 32GB | 8124 | capstone (8h, chain jobs) |
| Qwen3-8B | archimedes | RTX 5090 32GB | 8125 | ray-cluster (90d) |
| Qwen3-14B | kolmogorov | RTX 4090 24GB | 8128 | student-gpu (2h, chain jobs) |

---

## Step 1: Start Inference Servers

```bash
# 27B-AWQ on manifold — chain 3 jobs for 24h coverage (capstone = 8h max)
J1=$(sbatch scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
J2=$(sbatch --dependency=afterany:$J1 scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
J3=$(sbatch --dependency=afterany:$J2 scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')

# 8B on archimedes — single long-running job (ray-cluster = 90d)
sbatch scripts/serve_vllm_8b.sh

# 14B on kolmogorov — chain 8 jobs for 16h coverage (student-gpu = 2h max)
J1=$(sbatch scripts/serve_vllm_14b_kolmogorov.sh | awk '{print $4}')
J2=$(sbatch --dependency=afterany:$J1 scripts/serve_vllm_14b_kolmogorov.sh | awk '{print $4}')
J3=$(sbatch --dependency=afterany:$J2 scripts/serve_vllm_14b_kolmogorov.sh | awk '{print $4}')
# add more links as needed
```

Verify servers are up before launching experiments:
```bash
curl -sf http://10.0.10.69:8124/v1/models && echo "27B OK"
curl -sf http://10.0.10.58:8125/v1/models && echo "8B OK"
curl -sf http://10.0.10.52:8128/v1/models && echo "14B OK"
```

---

## Step 2: Smoke Test

Always smoke-test before a full sweep. Run against all three models:

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

---

## Step 3: Full Sweep

300 experiments per model (5 benchmarks × 12 methods × 5 seeds). Run all three in parallel:

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

The orchestrator is idempotent — safe to restart, skips completed experiments.

---

## Monitoring

```bash
# Per-model Telegram updates (one message per model)
.venv/bin/python scripts/monitor_multi_model.py --mode 15min

# Consolidated health dashboard (one message)
.venv/bin/python scripts/monitor_multi_model.py --mode 30min
```

Cron jobs run these automatically. Check actual GPU load directly:
```bash
curl -s http://10.0.10.58:8125/metrics | grep num_requests_running
```

---

## Results

```
runs/qwen3-27b-awq/<benchmark>/<method>/<seed>/result.json
runs/qwen3-8b/...
runs/qwen3-14b/...
```

`result.json` key fields: `test_score`, `train_scores`, `elapsed`.

---

## Worktree Layout

```
/users/achidamb/projects/gepa-mutations/   ← master branch (shared reference)
/users/achidamb/projects/gepa-cluster/     ← this worktree (sweep/cluster)
```

---

## Reference Docs

- [CLAUDE/cluster_infrastructure.md](CLAUDE/cluster_infrastructure.md) — all nodes, IPs, ports, GPU specs, partition limits, vLLM env
- [CLAUDE/sweep_execution.md](CLAUDE/sweep_execution.md) — full orchestrator reference, worker counts, execution order
- [CLAUDE/monitoring.md](CLAUDE/monitoring.md) — cron setup, Telegram, health checks, snapshot file
- [CLAUDE/known_bugs_and_fixes.md](CLAUDE/known_bugs_and_fixes.md) — gepa state fix, vLLM IPC bug, 14b/4b tag collision

## Conventions

- Paper hyperparameters: `temp=0.6`, `top_p=0.95`, `top_k=20`, `minibatch=3`
- Paper baseline scores in `src/gepa_mutations/config.py`
- AIME excluded from active sweep
- Always smoke test before full sweeps; get explicit go/no-go before launching
