# gepa-mutations

Research framework for ISO (Inductive Strategy Optimization) prompt optimization,
benchmarked against GEPA and other methods. Currently on `master` branch.

---

## Project Focus

**Primary goal:** Run controlled ISO experiments on RunPod (Qwen3-8B task + Qwen3-32B-AWQ reflection)
and compare against GEPA and MIPROv2 baselines across 5 benchmarks.

**Current state:** Infrastructure complete, 332 unit tests passing, no experiment runs yet.
The ISO harness (`src/iso_harness/`) is the main codebase — optimizer, experiment orchestrator,
checkpoint/resume, JSONL/Parquet logging, MLflow tracking.

---

## Architecture

```
src/gepa_mutations/       Shared: benchmarks, metrics, CLI, storage, notifications
src/iso_harness/
  optimizer/              ISO algorithm: skill discovery, tournament, pruning, reflection
  experiment/             Orchestrator, checkpoint, JSONL writer, MLflow, telemetry
  meta/                   Meta-learning (atlas, cartographer, scout)
methods/                  14 method packages (iso, gepa, miprov2, + 11 more)
configs/                  pilot.yaml, full.yaml, pilot_phase_a.yaml, pilot_phase_b.yaml
scripts/                  Serve (vLLM), run, validate, monitor
tests/                    unit/ (332 tests) + validation/ (119 tests)
gepa/                     Submodule (v0.1.1, PATCHED — see below)
```

---

## Key Entry Points

| Task | Command |
|------|---------|
| ISO CLI | `uv run python -m iso_harness.optimizer.cli` |
| Run experiments | `uv run python scripts/run_all_local.py --method iso gepa miprov2` |
| Run baselines | `uv run python scripts/run_baseline.py --benchmark hotpotqa hover pupa ifbench livebench` |
| Validate results | `uv run python scripts/validate_sweep.py` |
| Run tests | `uv run pytest tests/unit/` |

---

## Infrastructure: RunPod (vLLM)

ISO experiments use dual-model vLLM serving:
- **Task LM:** Qwen3-8B (bfloat16, port 8000)
- **Reflection LM:** Qwen3-32B-AWQ (port 8001)

Setup scripts: `scripts/iso_*.sh`
Config: `configs/pilot.yaml`, `configs/full.yaml`
Full guide: [docs/iso-experiment/README.md](docs/iso-experiment/README.md)

### RunPod Credentials

- API key in `.env` (`RUNPOD_API_KEY`)
- Existing pods (both EXITED, no volume): `gepa-pilot-04`, `gepa-pilot-04-pod2` (A40s)
- These are from the old exp-04 pilot — new pods needed for ISO runs

---

## Conventions

- Paper hyperparameters: `temp=0.6`, `top_p=0.95`, `top_k=20`, `minibatch=3`
- Seeds: 42, 123, 456, 789, 1024
- AIME excluded from active experiments
- Always smoke test before full sweeps
- Qwen3 thinking mode disabled (`--default-chat-template-kwargs '{"enable_thinking": false}'`)
- `<think>` blocks stripped from responses in parsing layer

---

## Submodule: gepa/

The `gepa/` submodule is pinned at v0.1.1 with **two critical patches applied**:

1. **makedirs fix** — `gepa/src/gepa/core/state.py`: added `os.makedirs(run_dir, exist_ok=True)`
   to `_atomic_write_json()` and `save()` to prevent FileNotFoundError on NFS paths.

2. **Pareto dominance timeout** — `gepa/src/gepa/gepa_utils.py`: added 120s timeout to
   `remove_dominated_programs()` to prevent O(n^2) stall on fast local models.

**Do NOT pull upstream gepa updates without re-applying these patches.**

---

## Known Issues

See [CLAUDE/known_bugs_and_fixes.md](CLAUDE/known_bugs_and_fixes.md) for full details.

| Issue | Status |
|-------|--------|
| GEPA Pareto stall (O(n^2)) | Fixed (timeout patch) |
| gepa state FileNotFoundError | Fixed (makedirs patch) |
| vLLM IPC socket path length | Workaround: `cd /tmp` before launch |
| Tournament invisible to monitoring | Use vLLM `/metrics` to confirm activity |
| HF_HOME must be set explicitly | All serve scripts set it |

---

## Recent Fixes (last 10 commits)

These are all in `src/iso_harness/optimizer/` and address issues found during local validation:

- `pool_floor` termination uses `<` instead of `<=`
- Evaluation error handling + JSONL/LoggingLM wired into CLI
- `--default-chat-template-kwargs` added to vLLM server startup
- Qwen3 thinking mode disabled + `<think>` block stripping
- Prompt formatting shortened to reduce token usage
- `max_tokens` reduced to fit vLLM context limits
- CLI builds LMs via `dspy.LM` for BaseLM compatibility
- CLI `_run_with_real_lm` uses `adapter._score`, handles `**kwargs`
- Rollout logging added to valset evaluation
- `dspy.LM` `list[str]` return type handled in LoggingLM and parsers

---

## Reference Docs

- [CLAUDE/known_bugs_and_fixes.md](CLAUDE/known_bugs_and_fixes.md) — patched bugs
- [CLAUDE/sweep_execution.md](CLAUDE/sweep_execution.md) — orchestrator, run directory layout
- [CLAUDE/mac_setup.md](CLAUDE/mac_setup.md) — MLX model setup (for local dev/testing)
- [docs/iso-experiment/README.md](docs/iso-experiment/README.md) — RunPod deployment guide
