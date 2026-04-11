# Sweep Execution

## Overview

The sweep runs 300 experiments per model: 5 benchmarks × 12 methods × 5 seeds.
(AIME excluded from active sweep — no timing data, skip with `--benchmark` filter.)

Active benchmarks: `hotpotqa`, `hover`, `pupa`, `ifbench`, `livebench`

Methods: `gepa`, `best_of_k_K3`, `contrastive_reflection`, `failure_stratified_k_K3`,
`synaptic_pruning`, `tournament`, `slime_mold`, `ant_colony`,
`active_minibatch`, `contrastive_synthesis`, `ecological_succession`, `modular`

Seeds: `42, 123, 456, 789, 1024`

## Per-Model Launch

Each model runs as an independent orchestrator process (SLURM srun or background):

```bash
# Set model-specific env vars before launching
export GEPA_MODEL="Qwen/Qwen3-27B-AWQ"
export GEPA_BASE_URL="http://10.0.10.69:8124/v1"
srun --partition=capstone --nodelist=manifold --gres=gpu:0 --time=8:00:00 \
  .venv/bin/python scripts/run_all_local.py --workers 8 \
  --benchmark hotpotqa hover pupa ifbench livebench \
  &> logs/orchestrator_27b.log &
```

Model → env var → log file mapping:

| Model | GEPA_MODEL | GEPA_BASE_URL | Log |
|-------|-----------|---------------|-----|
| Qwen3-27B-AWQ | `Qwen/Qwen3-27B-AWQ` | `http://10.0.10.69:8124/v1` | `orchestrator_27b.log` |
| Qwen3-8B | `Qwen/Qwen3-8B` | `http://10.0.10.58:8125/v1` | `orchestrator_8b.log` |
| Qwen3-4B (sapphire) | `Qwen/Qwen3-4B` | `http://10.0.100.99:8126/v1` | `orchestrator_4b.log` |
| Qwen3-1.7B (sar) | `Qwen/Qwen3-1.7B` | `http://10.0.50.99:8127/v1` | `orchestrator_1b.log` |
| Qwen3-14B (kolmogorov) | `Qwen/Qwen3-14B` | `http://10.0.10.52:8128/v1` | `orchestrator_14b.log` |
| Qwen3-4B (mandelbrot) | `Qwen/Qwen3-4B` | `http://10.0.10.53:8129/v1` | `orchestrator_4b_mandelbrot.log` |
| Qwen3-1.7B (gho) | `Qwen/Qwen3-1.7B` | `http://10.0.50.69:8130/v1` | `orchestrator_1b_gho.log` |

## GEPA_MODEL → Run Directory

`run_all_local.py:_env_model_tag()` maps `GEPA_MODEL` to a directory prefix:

| GEPA_MODEL contains | Directory prefix |
|--------------------|-----------------|
| `27b` | `runs/qwen3-27b-awq/` |
| `14b` | `runs/qwen3-14b/` |
| `8b` | `runs/qwen3-8b/` |
| `4b` | `runs/qwen3-4b/` |
| `1.7b` or `1b` | `runs/qwen3-1.7b/` |

**Important:** `14b` check must come before `4b` check — "qwen3-14b" contains "4b" as substring.
This is already fixed in `run_all_local.py`.

## Worker Count Recommendations

vLLM batches concurrent requests automatically. More workers = more throughput up to GPU saturation:

| Model | Recommended workers | Notes |
|-------|--------------------|----|
| 27B-AWQ | 6–8 | RTX 5090 32GB; AWQ reduces memory/latency |
| 8B | 8–12 | RTX 5090 32GB; plenty of headroom |
| 4B | 8–12 | RTX 5090 or 4090 |
| 1.7B | 10–14 | Small model, saturates fast |
| 14B | 4–6 | RTX 4090 24GB with fp8 quant |
| 4B (2nd) | 6–8 | RTX 4060 Ti 16GB; less VRAM |
| 1.7B (2nd) | 8–10 | RTX 4060 Ti 16GB |

## Smoke Test Before Sweep

Always run smoke test first:

```bash
export GEPA_MODEL="Qwen/Qwen3-8B"
export GEPA_BASE_URL="http://10.0.10.58:8125/v1"
.venv/bin/python scripts/run_all_local.py --smoke-test --workers 4
```

Smoke test uses `--subset 5` (5 examples per run). Verify at least 3/4 methods pass before go/no-go.

## Execution Order

`run_all_local.py` sorts by: benchmark priority first, then method duration (fastest-first within each benchmark).
This ensures quick wins early and avoids long-tail experiments blocking the queue.

Benchmark priority: `ifbench → pupa → livebench → hotpotqa → hover → aime`

## Results Layout

```
runs/
  qwen3-27b-awq/          # model-tagged directory
    hotpotqa/
      gepa/
        42/result.json    # seed=42
        123/result.json
        ...
      best_of_k_K3/
        ...
  qwen3-8b/
    ...
```

Each `result.json` contains at minimum:
- `test_score`: float — final held-out test accuracy
- `train_scores`: list[float] — per-round train scores
- `elapsed`: float — wall-clock seconds

## Parallelism Notes

- Two models can run against the same model size (e.g., 4B on sapphire + 4B on mandelbrot) because results go to the same `runs/qwen3-4b/` directory — each experiment has a unique `benchmark/method/seed` path, so there are no write conflicts.
- The orchestrator skips already-completed experiments on restart (`is_done` checks for `result.json`).
- `tournament` and other long methods don't write checkpoint files until completion. `get_in_progress()` uses a 45-min stale window, so these appear as "0 running" in monitoring even when active. Confirm with vLLM `/metrics` endpoint if concerned.
