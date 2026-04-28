# gepa-mutations

Research framework for **ISO (Inductive Strategy Optimization)** — a prompt optimization algorithm that discovers task-specific skills via LLM reflection, then evolves candidates through tournament-style pruning and cross-mutation. Benchmarked against [GEPA](https://arxiv.org/abs/2507.19457) (ICLR 2026 Oral) and 12 other optimization methods across 6 benchmarks.

## Status

Infrastructure complete. ISO optimizer, experiment harness, and 14 method implementations are built and tested. Full experiment sweep pending.

---

## What is ISO?

ISO (Inductive Strategy Optimization) is a prompt optimization method that works in two phases:

1. **Skill Discovery** — analyzes training examples to identify failure modes, then generates skill clusters (reusable prompt strategies) via LLM reflection.
2. **Tournament Optimization** — instantiates candidates from discovered skills, evaluates on multi-minibatch samples, prunes underperformers, reflects on failures to propose mutations, and cross-mutates survivors. Repeats until convergence or budget exhaustion.

Key properties:
- Discovers *why* prompts fail before trying to fix them
- Supports multiple pruning/reflection/merge variants (sprint, grove, tide, lens, storm)
- Dual-model architecture: task LM (Qwen3-8B) + reflection LM (Qwen3-32B-AWQ)
- Full checkpoint/resume, 7-layer data collection, MLflow tracking

---

## Repository Structure

```
src/
  gepa_mutations/            Shared infrastructure
    benchmarks/              Dataset loaders + evaluators (6 benchmarks)
    metrics/                 MetricsCollector, TrackedLM, token tracking
    runner/                  ExperimentRunner, callbacks, progress
    storage/                 Atomic JSON persistence
    notifications/           Telegram notifier
    cli.py                   Main CLI entry point
  iso_harness/               ISO optimizer + experiment orchestration
    optimizer/               Core algorithm, evaluation, reflection, pruning, merge
    experiment/              Orchestrator, checkpoint, JSONL/Parquet, MLflow, telemetry
    meta/                    Meta-learning (atlas, cartographer, scout)

methods/                     14 optimization method implementations
  iso/                       Inductive Strategy Optimization
  gepa/                      GEPA baseline (paper-faithful)
  miprov2/                   MIPro v2 (DSPy)
  contrastive_reflection/    GEPA + contrastive pairs
  synaptic_pruning/          Generate-ablate-prune pipeline
  tournament/                Single-elimination bracket
  random_search/             Random baseline
  ...                        + 7 more (ant_colony, active_minibatch, etc.)

configs/                     Experiment YAML configs (pilot, full)
scripts/                     Serve, run, validate, monitor (shell + Python)
tests/                       58 test files (unit/ + validation/)
gepa/                        GEPA submodule (v0.1.1, patched)
docs/                        Design decisions, infrastructure guides
```

---

## Quick Start

### Install

```bash
git clone --recurse-submodules https://github.com/ashwinchidambaram/gepa-mutations.git
cd gepa-mutations
pip install uv && uv sync
cp .env.example .env   # add API keys
```

### Run a single experiment

```bash
export GEPA_MODEL="Qwen/Qwen3-8B"
export GEPA_BASE_URL="http://localhost:8125/v1"

# Smoke test (fast validation)
python scripts/run_all_local.py --smoke-test --workers 4 \
  --benchmark hotpotqa --method iso --seeds 42

# Full experiment
python scripts/run_all_local.py --workers 6 \
  --benchmark hotpotqa --method gepa iso miprov2 \
  --seeds 42 123 456 789 1024
```

### Run baselines (no optimization)

```bash
python scripts/run_baseline.py --benchmark hotpotqa hover pupa ifbench livebench
```

### Validate results

```bash
python scripts/validate_sweep.py              # all models
python scripts/validate_sweep.py --model qwen3-8b  # one model
```

---

## Methods

| Method | Type | Description |
|--------|------|-------------|
| **ISO** | Primary | Skill discovery + tournament optimization with reflection |
| **GEPA** | Baseline | Paper's 4-step loop: sample, reflect, Pareto select, merge |
| **MIPROv2** | Baseline | DSPy's MIPro v2 optimizer |
| Contrastive Reflection | Mutation | GEPA + mined contrastive pairs |
| Synaptic Pruning | Standalone | Generate overspecified prompts, ablate, prune |
| Tournament | Standalone | 64-candidate single-elimination bracket |
| Random Search | Floor | Random prompt sampling (sanity check) |
| Ant Colony | Mutation | Ant colony optimization on prompt space |
| Active Minibatch | Mutation | Active learning for minibatch selection |
| Contrastive Synthesis | Mutation | Synthesis from success/failure contrasts |
| Ecological Succession | Mutation | Population pruning with succession phases |
| Failure Stratified K | Mutation | K-selection stratified by failure type |
| Best of K | Baseline | Top-k selection (no optimization) |
| Modular | Mutation | Modular prompt decomposition + optimization |

---

## Benchmarks

| Benchmark | Task Type | Rollout Budget | Test Size |
|-----------|-----------|---------------|-----------|
| HotpotQA | Multi-hop QA | 6,871 | 300 |
| HoVer | Fact verification | 2,426 | 300 |
| PUPA | Privacy redaction | 3,936 | 300 |
| IFBench | Instruction following | 3,593 | 300 |
| LiveBench-Math | Math reasoning | 1,839 | ~100 |
| AIME-2025 | Competition math | 7,051 | 30 |

### Hyperparameters

All shared parameters match the GEPA paper (arXiv:2507.19457v2):

| Parameter | Value |
|-----------|-------|
| Temperature | 0.6 |
| top_p / top_k | 0.95 / 20 |
| Max context | 16,384 |
| Minibatch size | 3 |
| Seeds | 42, 123, 456, 789, 1024 |

---

## Infrastructure

### RunPod (vLLM)

Dual-model serving for ISO experiments:
- **Task LM:** Qwen3-8B (bfloat16)
- **Reflection LM:** Qwen3-32B-AWQ

Setup and deployment scripts in `scripts/iso_*.sh`. Configuration in `configs/`.

See [docs/iso-experiment/README.md](docs/iso-experiment/README.md) for full deployment guide.

---

## Results

> Experiments not yet run. See `configs/pilot.yaml` and `configs/full.yaml` for planned sweep configuration.

### Score Table

<!-- TODO: Fill after experiments complete -->

| Method | HotpotQA | IFBench | HoVer | PUPA | LiveBench | Aggregate | vs GEPA |
|--------|----------|---------|-------|------|-----------|-----------|---------|
| Baseline | | | | | | | — |
| GEPA | | | | | | | — |
| ISO | | | | | | | |
| MIPROv2 | | | | | | | |

### Convergence Curves

<!-- TODO: Add plots after experiments complete -->
<!-- Per (benchmark), all methods overlaid, rollout budget on x-axis -->

---

## Data Schema

Each experiment produces files in `runs/{model_tag}/{benchmark}/{method}/{seed}/`:

### result.json

| Field | Type | Description |
|-------|------|-------------|
| `test_score` | float | Held-out test accuracy |
| `val_score` | float | Best validation score during optimization |
| `train_score` | float | Training set score (overfitting indicator) |
| `seed_prompt_test_score` | float | Un-optimized prompt on test set |
| `seed_prompt_val_score` | float | Un-optimized prompt on val set |
| `best_prompt` | dict | The optimized system prompt |
| `all_candidates` | list | Top 20 candidates with scores |
| `test_example_scores` | list | Per-example test scores |
| `rollout_count` | int | Total LLM evaluation calls used |
| `wall_clock_seconds` | float | Total runtime |

### metrics.json

| Field | Type | Description |
|-------|------|-------------|
| `total_tokens` | int | Total tokens consumed |
| `task_error_count` | int | LLM errors during evaluation |
| `reflection_error_count` | int | LLM errors during reflection |
| `val_score_trajectory` | list | (rollout, score) pairs |
| `prompt_length_trajectory` | list | (rollout, char_length) pairs |
| `stage_timings` | list | Per-stage seconds + rollouts |

### test_outputs.json

Raw model responses for qualitative error analysis.

---

## Known Issues

See [docs/issues.md](docs/issues.md).

---

## Citation

```bibtex
@inproceedings{gepa2026,
  title={GEPA: Genetic Evolution of Prompts with Adaptation},
  author={...},
  booktitle={ICLR},
  year={2026}
}
```

## License

MIT
