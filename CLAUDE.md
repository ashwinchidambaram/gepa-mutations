# gepa-mutations

## Project Overview
Experimental framework for reproducing and mutating the GEPA prompt evolution framework (arXiv:2507.19457, ICLR 2026 Oral).

## Tech Stack
- Python 3.12, uv for dependency management
- `gepa[full]` — official GEPA package (uses LiteLLM internally)
- OpenRouter API for Qwen3-8B model access
- AWS (boto3) — S3 for results, EC2 for compute, SNS for notifications
- pandas, numpy, matplotlib, plotly — data analysis and visualization
- typer, rich — CLI
- pydantic-settings — configuration

## Directory Structure
- `gepa/` — official GEPA source code (reference, do not modify)
- `src/gepa_mutations/` — our experiment code
  - `runner/` — experiment execution and AWS deployment
  - `benchmarks/` — dataset loading (HotpotQA, IFBench, AIME-2025, LiveBench-Math, HoVer, PUPA)
  - `notifications/` — SNS and Telegram notification system
  - `storage/` — S3 result storage
  - `analysis/` — visualization and comparison against paper baselines
  - `mutations/` — GEPA mutation variants (Phase 2)
- `configs/` — experiment configurations
- `notebooks/` — Jupyter notebooks for analysis
- `data/` — local data cache
- `tests/` — test suite

## Conventions
- All experiment configs should match paper hyperparameters by default (temp=0.6, top_p=0.95, top_k=20, minibatch=3, round-robin module selection)
- Paper baseline scores are stored in `src/gepa_mutations/config.py`
- Model access via OpenRouter: `openrouter/qwen/qwen3-8b`

## Phases
1. Reproduce GEPA paper results as-is (proof of concept)
2. Introduce and test mutations to GEPA's algorithms
