# gepa-mutations

Experimental framework for reproducing and extending [GEPA](https://arxiv.org/abs/2507.19457) (ICLR 2026 Oral) — an automatic prompt evolution framework that uses a language model to iteratively improve task-specific prompts via natural-language reflection and Pareto-frontier selection.

## Background

GEPA replaces RL-based prompt optimization with an evolutionary loop:

1. **Trajectory Sampling** — collect execution traces on a minibatch of training examples
2. **Natural Language Reflection** — an LLM diagnoses failure patterns and proposes prompt improvements
3. **Pareto Frontier Selection** — maintain a diverse population, each candidate best on different examples
4. **System-Aware Merge** — crossover operator that combines complementary improvements

The paper reports +9.62pp aggregate improvement over baseline Qwen3-8B across 6 benchmarks, outperforming GRPO and MIPROv2 using up to 35× fewer rollouts.

This project reproduces the GEPA baseline and systematically tests 11 algorithm mutations across 4 model sizes (1.7B → 27B-AWQ).

---

## Methods

| Method | Core Idea |
|--------|-----------|
| `gepa` | Paper-faithful baseline |
| `best_of_k_K3` | Generate K candidate prompts per round, keep the best |
| `contrastive_reflection` | Feed the LLM contrastive success/failure pairs to guide reflection |
| `failure_stratified_k_K3` | Oversample failure examples when building reflection minibatches |
| `synaptic_pruning` | Generate a long structured prompt then ablate sections to find the minimal effective subset |
| `tournament` | Bracket tournament across a pool of LLM-generated candidates |
| `slime_mold` | Multi-round colony: spawn candidates, prune survivors, reinforce best-performing |
| `ant_colony` | Pheromone-weighted selection: examples that previously drove improvement get higher sampling weight |
| `active_minibatch` | Preferentially select high-variance examples (disagreement across candidates) for reflection |
| `contrastive_synthesis` | Distill contrastive pairs into an abstract improvement principle via a synthesis LLM call |
| `ecological_succession` | Multi-phase GEPA: easy examples first, then gradually introduce harder ones |
| `modular` | Decompose prompt into functional modules and optimize each independently |

---

## Benchmarks

| Benchmark | Task Type | Notes |
|-----------|-----------|-------|
| HotpotQA | Multi-hop QA | F1 scoring; requires 2+ step reasoning chains |
| HoVer | Fact verification | Binary SUPPORTS / NOT_SUPPORTED over structured evidence |
| PUPA | Privacy redaction | Rewrite queries replacing PII with `[REDACTED]` tags |
| IFBench | Instruction following | All constraints must be satisfied; partial credit only |
| LiveBench-Math | Math reasoning | AMC/AIME-difficulty problems, exact-match scoring |

AIME-2025 is in the codebase but excluded from the active sweep (insufficient timing data).

---

## Cluster Setup

Inference runs on a local SLURM cluster. Each model size is served independently via vLLM:

| Model | Node | GPU | VRAM | Port |
|-------|------|-----|------|------|
| Qwen3-27B-AWQ | manifold | RTX 5090 | 32 GB | 8124 |
| Qwen3-8B | archimedes | RTX 5090 | 32 GB | 8125 |
| Qwen3-4B | sapphire | RTX 4090 | 24 GB | 8126 |
| Qwen3-1.7B | sar-gpu-vm | RTX 3090 | 24 GB | 8127 |
| Qwen3-14B | kolmogorov | RTX 4090 | 24 GB | 8128 |

See [CLAUDE/cluster_infrastructure.md](CLAUDE/cluster_infrastructure.md) for full details, SLURM job chain setup, and downed node status.

---

## Quickstart

### Install

```bash
git clone --recurse-submodules <repo-url>
cd gepa-mutations-raycluster
uv sync
cp .env.example .env   # add TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
```

### Launch a vLLM server

```bash
# Single job (ray-cluster partition — no time limit)
sbatch scripts/serve_vllm_8b.sh

# Job chain for time-limited partitions (capstone = 8h max)
J1=$(sbatch scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
J2=$(sbatch --dependency=afterany:$J1 scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
J3=$(sbatch --dependency=afterany:$J2 scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
```

### Smoke test (run first before every full sweep)

```bash
export GEPA_MODEL="Qwen/Qwen3-8B"
export GEPA_BASE_URL="http://10.0.10.58:8125/v1"
python scripts/run_all_local.py --smoke-test --workers 4
```

### Full sweep

```bash
export GEPA_MODEL="Qwen/Qwen3-8B"
export GEPA_BASE_URL="http://10.0.10.58:8125/v1"
python scripts/run_all_local.py --workers 8 --benchmark hotpotqa hover pupa ifbench livebench
```

The orchestrator is idempotent — it skips already-completed experiments on restart.

### Monitoring

```bash
# Per-model Telegram status (one message per model)
python scripts/monitor_multi_model.py --mode 15min

# Consolidated sweep health (one message total)
python scripts/monitor_multi_model.py --mode 30min
```

---

## Configuration

Set in `.env`:

```
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

Set via environment before launching the orchestrator:

```bash
export GEPA_MODEL="Qwen/Qwen3-27B-AWQ"    # maps to runs/qwen3-27b-awq/
export GEPA_BASE_URL="http://10.0.10.69:8124/v1"
```

Paper hyperparameters and baseline scores are in `src/gepa_mutations/config.py`.

---

## Results Layout

```
runs/
  qwen3-27b-awq/
    hotpotqa/
      gepa/42/result.json
      best_of_k_K3/42/result.json
      ...
  qwen3-8b/
    ...
```

`result.json` key fields:
- `test_score` — final held-out accuracy
- `train_scores` — per-round training scores  
- `elapsed` — wall-clock seconds

---

## Repository Structure

```
gepa/                    GEPA submodule (v0.1.1, patched — see CLAUDE/known_bugs_and_fixes.md)
src/gepa_mutations/      Shared infrastructure
  benchmarks/            Dataset loaders and evaluators
  runner/                Experiment runner, LM wrapper, callbacks
  notifications/         Telegram alerts
  config.py              Settings and paper baseline scores
methods/                 Algorithm mutations (one editable package each)
scripts/
  run_all_local.py       Parallel experiment orchestrator
  monitor_multi_model.py Telegram monitoring (15-min and 30-min modes)
  check_node_recovery.sh Cron: alerts when downed nodes recover
  smoke_test_all.py      Pre-sweep smoke test runner
  serve_vllm_*.sh        SLURM job scripts for each vLLM worker
CLAUDE/                  Operational knowledge for AI assistants
docs/                    Planning docs, mutation selection report
configs/                 Experiment configurations
notebooks/               Analysis notebooks
tests/                   Test suite
data/                    Dataset cache (raw/ gitignored)
runs/                    Results (gitignored)
logs/                    Logs (gitignored)
```

---

## Known Issues

See [CLAUDE/known_bugs_and_fixes.md](CLAUDE/known_bugs_and_fixes.md):

- **gepa state save `FileNotFoundError` on NFS** — patched in `gepa/src/gepa/core/state.py`
- **vLLM IPC socket path length limit** — fix: `cd /tmp` before launching (all serve scripts do this)
- **`_env_model_tag()` 14B/4B substring collision** — fixed in `scripts/run_all_local.py`
- **`tournament` invisible to monitoring** — method writes no intermediate files; use vLLM `/metrics` to confirm activity
