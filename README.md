# gepa-mutations

An exploratory research project for understanding, evaluating, and extending the [GEPA (Genetic-Pareto)](https://arxiv.org/abs/2507.19457) prompt evolution framework. GEPA uses natural language reflection and Pareto-based selection to optimize LLM prompts, achieving results that outperform reinforcement learning approaches like GRPO while using up to 35x fewer rollouts.

This project has two phases:

1. **Reproduce** the GEPA paper's results with experimental rigor to establish a reliable baseline
2. **Mutate** GEPA's core algorithms to explore whether further performance improvements are possible

## Background

GEPA replaces traditional RL-based prompt optimization with an evolutionary approach:

- **Trajectory Sampling** — collects execution traces including reasoning, tool calls, and outputs
- **Natural Language Reflection** — an LLM diagnoses failure patterns and proposes prompt improvements
- **Pareto Frontier Selection** — maintains a diverse population of candidates, each best on different task instances
- **System-Aware Merge** — crossover operator that combines complementary improvements from different candidates

The paper reports a **+9.62 aggregate improvement** over baseline on Qwen3-8B across 6 benchmarks, outperforming GRPO (+3.68) and MIPROv2 (+2.61) while using far fewer rollouts.

## Phase 1: Reproduction

Reproduce the paper's Qwen3-8B experiments as faithfully as possible:

| Benchmark | Paper Baseline | Paper GEPA | Our Result |
|-----------|---------------|------------|------------|
| HotpotQA | 42.33 | 62.33 | — |
| IFBench | 36.90 | 38.61 | — |
| HoVer | 35.33 | 52.33 | — |
| PUPA | 80.82 | 91.85 | — |
| AIME-2025 | 27.33 | 32.00 | — |
| LiveBench-Math | 48.70 | 51.95 | — |

**Model**: Qwen3-8B via OpenRouter (`qwen/qwen3-8b`)
**Inference params**: temperature=0.6, top_p=0.95, top_k=20, context=16K
**Hyperparams**: minibatch_size=3, round-robin module selection, up to 5 merge invocations

### Deviations from Paper

We use `gepa.api.optimize()` (not `optimize_anything()`) with custom `GEPAAdapter` implementations per benchmark. This matches the paper's actual internal implementation and avoids mismatched defaults in the higher-level API:

| Setting | `optimize_anything()` | `optimize()` (paper) | Ours |
|---------|----------------------|---------------------|------|
| Frontier type | `"hybrid"` | `"instance"` | `"instance"` |
| Reflection prompt | Verbose parameter-optimization template | Classic `<curr_param>` / `<side_info>` | Classic (default) |
| skip_perfect_score | `False` | `True` | `True` |
| perfect_score | `None` | `1.0` | `1.0` |

Additional deviations for statistical rigor:

- **5 seeds per benchmark** (42, 123, 456, 789, 1024) — the paper reports single runs with no confidence intervals. We run 5 seeds to compute bootstrap 95% CIs.
- **Custom `LM` wrapper** — GEPA v0.1.1 (the paper release) does not include the `gepa.lm.LM` class (added post-release in commit `ad4e94c`). When `reflection_lm` is passed as a string, the v0.1.1 code creates an inline closure that does NOT forward `temperature`, `top_p`, `top_k`, or `max_tokens` to LiteLLM. We provide our own `LM` class in `runner/experiment.py` that explicitly passes these parameters to ensure the reflection LLM uses the paper's inference settings.
- **`MetricsCallback`** — captures per-iteration diagnostics (acceptance rate, Pareto front evolution, merge stats, wall-clock timing) during optimization. This data is not part of the paper but is essential for Phase 2 mutation design.

### Reproduction Criteria

| Level | Criteria |
|-------|---------|
| STRONG MATCH | All 6 benchmarks within tolerance, aggregate within 1pp |
| ACCEPTABLE | 5/6 within tolerance, aggregate within 2pp |
| FAILED | 3+ outside tolerance OR aggregate differs by 3+pp |

Per-benchmark tolerance: ±2pp (IFBench), ±3pp (HotpotQA, HoVer, PUPA, LiveBench), ±4pp (AIME)

## Phase 2: Mutations

Once baseline reproduction is established, test mutations to GEPA's core components:

- **Candidate Selection** — alternatives to Pareto sampling (epsilon-greedy, top-k, beam search)
- **Reflection Strategy** — modified meta-prompts, multi-step reflection, chain-of-thought reflection
- **Crossover Operators** — alternative merge strategies beyond the paper's system-aware merge
- **Population Dynamics** — varying population size, elitism strategies, diversity preservation
- **Feedback Functions** — richer diagnostic signals, structured vs. free-form feedback

Each mutation is compared against the Phase 1 baseline to measure impact.

## Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone (includes GEPA v0.1.1 as a subdir dependency)
git clone <repo-url>
cd gepa-mutations

# If gepa/ is not in the repo, clone the paper's release:
git clone --branch v0.1.1 https://github.com/gepa-ai/gepa.git gepa
rm -rf gepa/.venv gepa/uv.lock  # prevent uv workspace confusion

# Create venv and install dependencies
uv sync

# Copy and fill in environment variables
cp .env.example .env
# Edit .env with your OpenRouter API key, HF token, etc.
```

### Required Credentials

| Service | Purpose | Env Variable |
|---------|---------|-------------|
| OpenRouter | Qwen3-8B API access | `OPENROUTER_API_KEY` |
| HuggingFace | Benchmark dataset access | `HF_TOKEN` |
| Telegram | Experiment status notifications (optional) | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |
| AWS | EC2 compute, S3 storage, SNS notifications (optional) | AWS CLI credentials |

## Usage

```bash
# Validate config and data loading
gepa-mutations run aime --dry-run

# Quick smoke test (5 examples)
gepa-mutations run aime --subset 5 --seed 42

# Full single-seed run
gepa-mutations run aime --seed 42

# Multi-seed run for statistical analysis
gepa-mutations run aime --seeds 42,123,456,789,1024

# Run without merge (ablation)
gepa-mutations run aime --seed 42 --no-merge

# Check completed experiments
gepa-mutations status

# Compare results against paper baselines
gepa-mutations compare

# Upload results to S3
gepa-mutations upload runs/
```

## Project Structure

```
gepa-mutations/
├── gepa/                       # Official GEPA source v0.1.1 (reference, do not modify)
├── src/gepa_mutations/
│   ├── cli.py                  # CLI entry point (run, status, compare, upload)
│   ├── config.py               # Settings, paper baselines, rollout budgets
│   ├── runner/
│   │   ├── experiment.py       # ExperimentRunner using gepa.api.optimize()
│   │   └── callbacks.py        # MetricsCallback for diagnostic capture
│   ├── benchmarks/
│   │   ├── loader.py           # BenchmarkData + load_benchmark() dispatcher
│   │   ├── evaluators.py       # GEPAAdapter implementations per benchmark
│   │   ├── signatures.py       # DSPy signatures (MathSolverSignature)
│   │   ├── aime.py             # AIME-2025 loader (MathArena/aime_2025)
│   │   ├── hotpotqa.py         # HotpotQA loader (distractor setting)
│   │   ├── ifbench.py          # IFBench loader (multi-constraint)
│   │   ├── hover.py            # HoVer loader (fact verification)
│   │   ├── pupa.py             # PUPA loader
│   │   └── livebench.py        # LiveBench-Math loader
│   ├── notifications/
│   │   └── notifier.py         # Telegram + SNS notification system
│   ├── storage/
│   │   ├── local.py            # Local filesystem persistence (runs/<bm>/<method>/<seed>/)
│   │   └── s3.py               # S3 upload/download/list
│   ├── analysis/
│   │   ├── statistics.py       # Bootstrap CIs, Cohen's d, reproduction verdict
│   │   └── visualize.py        # Comparison bar charts, convergence curves
│   └── mutations/              # GEPA mutation variants (Phase 2)
├── scripts/
│   ├── aws_setup.py            # Idempotent AWS infrastructure setup
│   └── launch_experiment.py    # EC2 spot instance experiment launcher
├── configs/                    # Experiment configurations
├── notebooks/                  # Jupyter notebooks for analysis
├── data/                       # Local data cache
└── tests/
```

## GEPA Dependency

This project depends on GEPA v0.1.1 (the paper's official release), installed from a local clone rather than PyPI:

- **Why not PyPI?** The PyPI release (v0.0.26) predates the callback system and several API parameters our code requires.
- **Why not latest main?** We pin to the v0.1.1 tag for maximum parity with the paper's published results.
- **The `gepa/` directory** is a clean `git clone --branch v0.1.1 https://github.com/gepa-ai/gepa.git`. It must not be modified.
- **Important**: Remove `gepa/.venv` and `gepa/uv.lock` if present — these cause uv to resolve dependencies from the wrong project.

## Benchmarks

All 6 benchmarks from the paper, publicly available:

| Benchmark | Task | Source |
|-----------|------|--------|
| HotpotQA | Multi-hop QA | Yang et al. 2018 |
| IFBench | Instruction following | HuggingFace `allenai/IF_multi_constraints_upto5` |
| AIME-2025 | Competition math | HuggingFace `MathArena/aime_2025` |
| LiveBench-Math | Cross-domain math | White et al. 2025 |
| HoVer | Fact verification | Jiang et al. 2020 |
| PUPA | Privacy-preserving delegation | Li et al. 2025 |

## Cost Estimate

API cost dominates (95%+). Per-rollout: ~$0.00043 via OpenRouter.

| Phase | Rollouts | Est. API Cost |
|-------|----------|---------------|
| Smoke test (AIME, subset=5) | ~50 | < $0.10 |
| Single benchmark, 1 seed | ~5,000 | ~$2–$5 |
| All 6 benchmarks, 1 seed | ~25,700 | ~$11–$25 |
| All 6 benchmarks, 5 seeds | ~128,500 | ~$55–$130 |

## Paper Reference

> **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning**
> Agrawal, Tan, Soylu, Ziems, Khare, Opsahl-Ong, Singhvi, Shandilya, Ryan, Jiang, Potts, Sen, Dimakis, Stoica, Klein, Zaharia, Khattab
> ICLR 2026 (Oral) | [arXiv:2507.19457](https://arxiv.org/abs/2507.19457) | [GitHub](https://github.com/gepa-ai/gepa)
