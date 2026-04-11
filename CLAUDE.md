# gepa-mutations

## Project Overview

Experimental framework for reproducing and extending the GEPA prompt evolution framework
(arXiv:2507.19457, ICLR 2026 Oral). Phase 1 reproduces paper results across 5 benchmarks
and 12 methods. Phase 2 introduces mutations to the core GEPA algorithm.

## Tech Stack

- Python 3.12, `uv` for dependency management, `.venv/` at repo root
- `gepa/` — official GEPA package as git submodule (v0.1.1, patched — see CLAUDE/known_bugs_and_fixes.md)
- Inference: local vLLM cluster serving Qwen3 models (1.7B through 27B-AWQ) over OpenAI-compatible API
- Notifications: Telegram bot (bot token + chat ID from `.env`)
- Storage: local `runs/` directory (NFS-mounted, gitignored)

## Directory Structure

```
gepa/                    — GEPA submodule (v0.1.1, do NOT pull upstream without re-applying patches)
src/gepa_mutations/      — shared infrastructure
  runner/                — experiment runner, LM wrapper, timeout, callbacks
  benchmarks/            — dataset loaders + scoring (HotpotQA, IFBench, LiveBench, HoVer, PUPA)
  base.py                — MutationConfig + run_mutation() for Phase 2
  notifications/         — Telegram notification system
  config.py              — Settings (pydantic-settings), paper baseline scores
methods/                 — one subdirectory per mutation method (editable packages)
  best_of_k/
  contrastive_reflection/
  failure_stratified_k/
  synaptic_pruning/
  tournament/
  slime_mold/
  ant_colony/
  active_minibatch/
  contrastive_synthesis/
  ecological_succession/
  modular/
scripts/
  run_all_local.py       — multi-worker experiment orchestrator
  monitor_multi_model.py — Telegram monitoring (15-min per-model + 30-min consolidated)
  check_node_recovery.sh — cron: pings downed nodes, alerts on recovery
  smoke_test_all.py      — pre-sweep smoke test runner
  serve_vllm_*.sh        — SLURM job scripts to launch vLLM per node
CLAUDE/                  — Claude knowledge docs (see below)
docs/                    — project planning, mutation selection report
configs/                 — experiment configurations
notebooks/               — analysis notebooks
data/                    — local dataset cache (raw/ gitignored)
tests/                   — test suite
runs/                    — experiment results (gitignored)
logs/                    — orchestrator and SLURM logs (gitignored)
```

## CLAUDE/ Knowledge Folder

Critical operational knowledge for Claude in future sessions:

- [CLAUDE/cluster_infrastructure.md](CLAUDE/cluster_infrastructure.md) — nodes, IPs, ports, GPU specs, partition limits, vLLM env
- [CLAUDE/sweep_execution.md](CLAUDE/sweep_execution.md) — how to launch/resume the multi-model sweep
- [CLAUDE/monitoring.md](CLAUDE/monitoring.md) — cron setup, Telegram alerts, health checks
- [CLAUDE/known_bugs_and_fixes.md](CLAUDE/known_bugs_and_fixes.md) — gepa state save fix, vLLM IPC path fix, 14b/4b collision, etc.

## Conventions

- Paper hyperparameters by default: `temp=0.6`, `top_p=0.95`, `top_k=20`, `minibatch=3`, round-robin module selection
- Paper baseline scores stored in `src/gepa_mutations/config.py`
- Model access: `GEPA_MODEL` env var selects model; `GEPA_BASE_URL` sets vLLM endpoint
- Run results: `runs/<model-tag>/<benchmark>/<method>/<seed>/result.json`
- AIME excluded from active sweep (no timing data, not in BENCHMARK_PRIORITY)
- Always smoke test before launching full sweeps; get explicit go/no-go from user

## Phases

1. **Reproduce** — run all 300 experiments/model (5 benchmarks × 12 methods × 5 seeds) across all model sizes
2. **Mutate** — introduce algorithmic mutations to GEPA; compare against Phase 1 baselines
