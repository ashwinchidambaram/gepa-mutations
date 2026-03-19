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

# Create venv and install dependencies
uv sync

# Copy and fill in environment variables
cp .env.example .env
# Edit .env with your OpenRouter API key, AWS credentials, etc.
```

### Required Credentials

| Service | Purpose | Env Variable |
|---------|---------|-------------|
| OpenRouter | Qwen3-8B API access | `OPENROUTER_API_KEY` |
| AWS | EC2 compute, S3 storage, SNS notifications | AWS CLI credentials |
| Telegram | Experiment status notifications | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |
| HuggingFace | Benchmark dataset access | `HF_TOKEN` |

## Usage

```bash
# Run a benchmark experiment
gepa-mutations run hotpotqa

# Run with a specific config
gepa-mutations run aime --config configs/default.py

# Compare results against paper baselines
gepa-mutations compare results/

# Upload results to S3
gepa-mutations upload results/
```

## Project Structure

```
gepa-mutations/
├── gepa/                       # Official GEPA source (reference, do not modify)
├── src/gepa_mutations/
│   ├── cli.py                  # CLI entry point
│   ├── config.py               # Settings, paper baselines, and hyperparameters
│   ├── runner/                 # Experiment execution and AWS deployment
│   ├── benchmarks/             # Benchmark dataset loading (paper splits)
│   ├── notifications/          # SNS / Telegram status alerts
│   ├── storage/                # S3 result storage
│   ├── analysis/               # Visualization and comparison charts
│   └── mutations/              # GEPA mutation variants (Phase 2)
├── configs/                    # Experiment configurations
├── notebooks/                  # Jupyter notebooks for analysis
├── data/                       # Local data cache
└── tests/
```

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

## Paper Reference

> **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning**
> Agrawal, Tan, Soylu, Ziems, Khare, Opsahl-Ong, Singhvi, Shandilya, Ryan, Jiang, Potts, Sen, Dimakis, Stoica, Klein, Zaharia, Khattab
> ICLR 2026 (Oral) | [arXiv:2507.19457](https://arxiv.org/abs/2507.19457) | [GitHub](https://github.com/gepa-ai/gepa)
