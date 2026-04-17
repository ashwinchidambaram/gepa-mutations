# Design: Inductive Strategy Discovery for Slime Mold

**Date:** 2026-04-15 (last revised 2026-04-16 after bar-raiser review)
**Model:** Qwen3-8B (bf16) — GEPA paper's primary model
**Deployment:** RunPod, 1x A40 48GB (with A40 throughput smoke benchmark before commit; fallback 1x RTX 4090)
**Budget:** 2,500 rollouts per run (hard cap); reflection calls tracked separately

> **⚠️ For final decisions, see the "Review Resolutions (Final)" section at the end of this document. That section supersedes any earlier content where they conflict.**

---

## Problem

Slime Mold has the highest ceiling on HotpotQA (0.719 on qwen3-1.7b) but huge variance across seeds. Root cause: personality-based generation strategies (Analytical, Creative, Minimal, Expert) produce prompts that sound different but fail on the same examples — they lack functional diversity.

## Solution

Three layered improvements to Slime Mold's candidate generation and mutation:

1. **Inductive Strategy Discovery** — LLM examines benchmark examples, identifies task-specific skills, generates prompts specialized per skill.
2. **Cross-Pollination Mutation** — Mutation prompt includes the strategy context plus concrete techniques borrowed from other candidates that succeed where this one fails.
3. **Refresh Pass** — After Round 1, re-run discovery on universally hard examples to find missed skills, generate new candidates.

These compose independently. Each can be tested in isolation or combined.

---

## Experiment Design

### Benchmarks

4 benchmarks: HotpotQA, HOVER, PUPA, IFBench.

Skip AIME (8B can't solve math) and LiveBench (no method separation in previous sweep).

### Seeds

5 seeds: 42, 123, 456, 789, 1024.

### Evaluation Axes

1. **Score at fixed budget (primary):** Mean +/- std across 5 seeds at 2,000 calls.
2. **Score at 500 calls:** Where Slime Mold currently converges — the "headline" comparison.
3. **Variance across seeds:** Std dev. Lower = more reliable = the #1 goal.
4. **Convergence point:** Where each method plateaus.
5. **Budget to reach 80% of best:** Efficiency for "good enough" use cases.

---

## Tier 1 — Run First

5 methods, 4 benchmarks, 5 seeds = 100 runs. ~15 hours on 2x RTX 4090. ~$12.

### Method 1: `iso` (existing baseline)

- **Strategies:** 4 personality (Analytical, Creative, Minimal, Expert)
- **Pool:** 4 strategies x 5 prompts = 19 + seed = 20
- **Mutation:** Blind (no strategy context)
- **Refresh:** None
- **Calls:** ~652
- **Purpose:** Control. Already have partial data for qwen3-8b (PUPA, IFBench).

### Method 2: `iso_prescribed8`

- **Strategies:** 8 universal problem-solving strategies:
  1. Divide & Conquer — break into sub-problems, solve easiest first, assemble
  2. Working Backward — start from end state, reason backward
  3. Analogy / Pattern Matching — map onto known solved problems
  4. Abstraction — strip specifics, solve simplified version, re-add detail
  5. Root Cause Analysis — ask "why?" iteratively before solving
  6. Trial & Error — systematically hypothesize, test, refine
  7. Constraint Satisfaction — enumerate all rules, solve within narrowed space
  8. Lateral Thinking — challenge assumptions, reframe, explore unconventional
- **Pool:** 8 strategies x 3 prompts = 24 + seed = 25
- **Mutation:** Blind
- **Refresh:** None
- **Calls:** ~748
- **Purpose:** Tests whether smarter prescribed strategies help, without task-adaptiveness.

### Method 3: `iso_inductive_k5_crosspollin`

- **Strategies:** Inductive discovery (K=5 fixed). LLM examines 8-10 seed-dependent train examples, identifies 5 skills with failure patterns.
- **Pool:** 5 skills x 4 prompts = 20 + seed = 21
- **Mutation:** Strategy-aware with cross-pollination. After evaluation, build failure matrix. For each survivor being mutated, find another candidate that succeeded where it failed. Include that donor's strategy name and relevant prompt section in the mutation prompt.
- **Refresh:** None
- **Calls:** ~664
- **Purpose:** The core experiment. Does task-adaptive generation + informed mutation help?

### Method 4: `iso_inductive_k5_refresh_expand`

- **Strategies:** Same as Method 3
- **Pool:** 21 initially, grows after refresh
- **Mutation:** Cross-pollination
- **Refresh:** After R1, collect "hard examples" — examples that >=70% of candidates failed on (threshold configurable via runner arg `hard_example_threshold`, default 0.7). Run focused discovery ("what skills are missing?"). Generate 4 prompts per new skill (expect 2 new skills = 8 candidates). New candidates JOIN the survivor pool for R2 (R2 pool: ~10 survivors + ~8 new = ~18).
- **Calls:** ~787
- **Purpose:** Does a second discovery pass catch missed skills? Does expanding the pool help?

### Method 5: `iso_inductive_k5_refresh_replace`

- **Strategies:** Same as Method 3
- **Pool:** 21, stays constant after refresh
- **Mutation:** Cross-pollination
- **Refresh:** Same discovery as Method 4, but new candidates REPLACE the weakest survivors instead of expanding. R2 pool stays at ~10.
- **Calls:** ~667
- **Purpose:** Expand pool vs replace weakest — which is better?

### Tier 1 Comparisons

| Comparison | What it isolates |
|-----------|-----------------|
| 1 vs 2 | Strategy quality (personality -> universal) |
| 1 vs 3 | Full inductive approach vs baseline (the big question) |
| 2 vs 3 | Task-adaptiveness (prescribed vs discovered) |
| 3 vs 4/5 | Refresh value |
| 4 vs 5 | Expand vs replace |

---

## Full Matrix — All 14 Methods

### Baselines (2 methods)

| # | Method | Strategies | Mutation | Refresh | Pool | Calls |
|---|--------|-----------|----------|---------|------|-------|
| 1 | `iso` | 4 personality | blind | none | 20 | ~652 |
| 2 | `iso_prescribed8` | 8 universal | blind | none | 25 | ~748 |

### K=3 — small pool, tests if few high-quality strategies suffice (4 methods)

| # | Method | Mutation | Refresh | Pool | Calls |
|---|--------|----------|---------|------|-------|
| 3 | `iso_inductive_k3` | blind | none | 13 | ~457 |
| 4 | `iso_inductive_k3_crosspollin` | cross-pollination | none | 13 | ~457 |
| 5 | `iso_inductive_k3_refresh_expand` | cross-pollination | expand | 13->~21 | ~636 |
| 6 | `iso_inductive_k3_refresh_replace` | cross-pollination | replace | 13 | ~460 |

**What K=3 tests:** Can 3 well-chosen skills match 5? If so, we get comparable quality at 30% fewer calls.

**Cascade (K=3, pool=13):** 13 -> 7 -> 4 -> 2 -> 1

### K=5 — matches current pool size, cleanest comparison (4 methods)

| # | Method | Mutation | Refresh | Pool | Calls |
|---|--------|----------|---------|------|-------|
| 7 | `iso_inductive_k5` | blind | none | 21 | ~664 |
| 8 | `iso_inductive_k5_crosspollin` | cross-pollination | none | 21 | ~664 |
| 9 | `iso_inductive_k5_refresh_expand` | cross-pollination | expand | 21->~29 | ~787 |
| 10 | `iso_inductive_k5_refresh_replace` | cross-pollination | replace | 21 | ~667 |

Tier 1 methods 3/4/5 correspond to full matrix #8, #9, #10.

**Cascade (K=5, pool=21):** 21 -> 10 -> 5 -> 3 -> 1

### K=adaptive — LLM decides how many skills exist (4 methods)

| # | Method | Mutation | Refresh | Pool | Calls |
|---|--------|----------|---------|------|-------|
| 11 | `iso_inductive_kadaptive` | blind | none | 4K+1 | ~747 |
| 12 | `iso_inductive_kadaptive_crosspollin` | cross-pollination | none | 4K+1 | ~747 |
| 13 | `iso_inductive_kadaptive_refresh_expand` | cross-pollination | expand | varies | ~870 |
| 14 | `iso_inductive_kadaptive_refresh_replace` | cross-pollination | replace | 4K+1 | ~750 |

**What K=adaptive tests:** Does the model find different numbers of skills for different benchmarks? If K varies meaningfully (e.g., 4 for PUPA, 7 for HotpotQA), that's evidence discovery is genuinely task-adaptive. If it always says ~6, fixed K is more reliable.

**Cascade (K=adaptive, est. K~6, pool~25):** 25 -> 12 -> 5 -> 3 -> 1

### Full Ablation Structure

| Comparison | What it isolates |
|-----------|-----------------|
| 1 vs 2 | Prescribed strategy quality (personality -> universal) |
| 1 vs 7 | Inductive discovery, blind mutation |
| 2 vs 7 | Task-adaptiveness (prescribed vs discovered, both blind) |
| 7 vs 8 | Cross-pollination effect |
| 8 vs 9/10 | Refresh effect |
| 9 vs 10 | Expand vs replace |
| 3 vs 7 vs 11 | K sensitivity (3 vs 5 vs adaptive), blind mutation |
| 4 vs 8 vs 12 | K sensitivity with cross-pollination |
| 5 vs 9 vs 13 | K sensitivity with refresh+expand |

### Cost Summary

| Scope | Methods | Runs | Total calls | 2x4090 est. | Cost |
|-------|---------|------|-------------|-------------|------|
| Tier 1 | 5 | 100 | ~65K | ~15 hrs | ~$12 |
| Tier 1 + K=3 | 9 | 180 | ~106K | ~24 hrs | ~$19 |
| Full matrix | 14 | 280 | ~185K | ~37 hrs | ~$29 |

---

## Implementation Components

### 1. Strategy Discovery (`colony.py`)

```python
@dataclass
class Strategy:
    name: str              # e.g. "multi-hop entity tracking"
    description: str       # what capability this is
    failure_pattern: str   # what going wrong looks like
    stress_examples: list  # which examples test this hardest

def discover_strategies(
    reflection_lm, task_description, examples, k=None
) -> list[Strategy]:
    """Phase 1: LLM examines examples, identifies distinct skills.

    Args:
        k: If int, discover exactly k skills. If None, adaptive (LLM decides).

    Uses 8-10 seed-dependent train examples.
    Returns structured list of Strategy objects.
    """
```

**Discovery prompt (fixed K):**
> Here is the task description and N examples. Identify exactly K distinct skills, reasoning patterns, or capabilities required to solve them well. For each: name (2-3 words), description, which examples stress-test it most, what failure looks like.

**Discovery prompt (adaptive K):**
> ...Identify the distinct skills required. There may be as few as 3 or as many as 10 — report however many you actually find. Don't pad or merge artificially.

**Parsing:** Numbered list format with regex parser (same pattern as existing `_parse_prompts`). Fall back to splitting on double-newlines. Each skill parsed into a Strategy dataclass.

### 2. Specialized Generation (`colony.py`)

```python
def generate_specialized_prompts(
    reflection_lm, strategy, task_description, examples, n=4
) -> list[str]:
    """Phase 2: Generate n prompts specialized for one skill.

    One LLM call that produces n prompts, each excelling at strategy.name
    while handling the general task competently.
    """
```

### 3. Prescribed-8 Strategies (`colony.py`)

```python
PRESCRIBED_STRATEGIES = [
    ("divide_and_conquer", "Break the problem into smaller sub-problems..."),
    ("working_backward", "Start from the desired end state..."),
    ("analogy", "Map the problem onto known solved patterns..."),
    ("abstraction", "Strip away specifics to find core logic..."),
    ("root_cause_analysis", "Ask 'why?' iteratively..."),
    ("trial_and_error", "Systematically generate hypotheses..."),
    ("constraint_satisfaction", "Enumerate all rules and limits upfront..."),
    ("lateral_thinking", "Challenge assumptions, reframe the problem..."),
]
```

### 4. Strategy-Aware Mutation with Cross-Pollination (`colony.py`)

```python
def mutate_prompt_with_strategy(
    reflection_lm, prompt, score, failures,
    strategy_name=None,           # what skill this prompt specializes in
    donor_strategy_name=None,     # strategy that succeeds on our failures
    donor_prompt_section=None,    # relevant section from the donor prompt
) -> str:
```

**Cross-pollination flow:**
1. After evaluation, build failure matrix: `{candidate_idx: set(failed_example_ids)}`
2. For each survivor S being mutated:
   - Find candidate D where `D.successes & S.failures` is maximized
   - Extract D's strategy name and the relevant section of D's prompt text
   - Include in mutation prompt: "Another prompt using {D.strategy} succeeded on examples you failed on. Here's its approach: {section}. Incorporate this while preserving your strengths."
3. Cost: zero extra LLM calls — uses existing evaluation data

### 5. Refresh Pass (`colony.py`)

```python
def discover_refresh_strategies(
    reflection_lm, hard_examples, existing_strategies, k_new=2
) -> list[Strategy]:
    """Find 1-3 new skills by examining examples that all strategies struggled with.

    Args:
        hard_examples: Examples where >70% of candidates failed.
        existing_strategies: Current skill list (to avoid rediscovery).
    """
```

**Expand variant:** New candidates added to survivor pool before R2.
**Replace variant:** New candidates replace weakest survivors, pool size stays constant.

### 6. Parameterized Runner (`runner.py`)

Single runner function with config flags:

```python
def run_iso(
    benchmark, seed,
    strategy_mode="personality",  # "personality" | "prescribed8" | "inductive"
    k=None,                       # 3 | 5 | None (adaptive). Ignored for personality/prescribed8.
    mutation_mode="blind",        # "blind" | "crosspollin"
    refresh_mode="none",          # "none" | "expand" | "replace"
    subset=None, max_metric_calls=None, settings=None,
) -> ExperimentResult:
```

**Method name derivation:**
- `strategy_mode="personality"` -> `iso`
- `strategy_mode="prescribed8"` -> `iso_prescribed8`
- `strategy_mode="inductive", k=5, mutation_mode="crosspollin", refresh_mode="none"` -> `iso_inductive_k5_crosspollin`
- etc.

**Pruning schedule scales with pool size:**
- Pool <= 15: 4 rounds, cascade ~halving (13 -> 7 -> 4 -> 2 -> 1)
- Pool 16-25: 4 rounds (21 -> 10 -> 5 -> 3 -> 1)
- Pool 26-35: 4 rounds (29 -> 14 -> 6 -> 3 -> 1)

### 7. Orchestrator Registration (`run_all_local.py`)

Add all 14 method names to `METHODS` list and `METHOD_COMMANDS` dict. Each maps to the same runner script with different CLI flags.

---

## Data Organization Strategy

We've now run 2 experiments (SLURM sweep, RunPod baseline) and this is our 3rd. Going forward, experiments are self-contained directories under `experiments/`.

### Directory Structure

```
experiments/
  01-slurm-sweep-2026-03/            # Experiment 1 (current runs_archive_slurm_sweep/)
    README.md                        # Purpose, dates, findings, status
    runs/                            # {model_tag}/{benchmark}/{method}/{seed}/
    plots/                           # Analysis plots specific to this experiment
    logs/                            # Orchestrator and vLLM logs
    report.md                        # Findings write-up

  02-runpod-baseline-2026-04/        # Experiment 2 (current runs/)
    README.md
    runs/
    plots/
    logs/
    report.md

  03-inductive-discovery-2026-04/    # Experiment 3 (this one)
    README.md
    spec.md -> ../../docs/superpowers/specs/2026-04-15-inductive-strategy-discovery-design.md
    runs/
    plots/
    logs/
    report.md

# Shared at repo root:
runs/                                # SYMLINK -> experiments/{active}/runs/
docs/superpowers/specs/              # All design specs
analysis/                            # Cross-experiment comparison plots only
scripts/                             # Reusable orchestration
```

### Naming Convention

`{NN}-{short-slug}-{YYYY-MM}`
- Sequential number makes progression obvious
- Short slug identifies purpose
- Date suffix for when it started

### Symlink Strategy

`runs/` at the repo root is a symlink to the active experiment's `runs/`. This keeps all existing code (`save_result`, `validate_sweep`, `run_all_local.py`) working unchanged — they still read/write `runs/{model_tag}/{benchmark}/{method}/{seed}/`.

When starting a new experiment:
```bash
mkdir -p experiments/NN-slug-YYYY-MM/{runs,plots,logs}
ln -sfn experiments/NN-slug-YYYY-MM/runs runs
```

### Experiment README Template

Each experiment has a `README.md`:

```markdown
# Experiment NN: Short Title

**Dates:** YYYY-MM-DD to YYYY-MM-DD
**Status:** active | complete | abandoned
**Model(s):** qwen3-Xb
**Deployment:** RunPod 2x RTX 4090 / SLURM cluster / Mac MLX
**Spec:** docs/superpowers/specs/YYYY-MM-DD-name-design.md
**Report:** report.md (filled in after completion)

## Purpose
One paragraph on what this experiment tested and why.

## Methods Tested
List of method names and brief description.

## Benchmarks x Seeds
Which benchmarks, which seeds, total runs.

## Key Findings
Top 3-5 results (filled in when experiment completes).

## Cost
Total GPU hours and $ spent.
```

### Immutability Rule

**Finalized experiments are read-only.** Once `report.md` is filled in and committed, nobody re-runs methods into that directory. If a bug is found or more data is needed, start a new experiment (e.g., `05-inductive-rerun-2026-05/`) rather than polluting the original.

### Per-experiment Git Strategy

- Commit `runs/` data incrementally during execution (per benchmark)
- Commit `plots/` when analysis is done
- Commit `report.md` when findings are written
- Tag the repo when experiment completes: `git tag exp-03-complete`

### Migration (Phase 0 of Implementation)

```bash
# Experiment 1 — rename archive
git mv runs_archive_slurm_sweep experiments/01-slurm-sweep-2026-03/runs
# + write README.md

# Experiment 2 — move active runs
git mv runs experiments/02-runpod-baseline-2026-04/runs
# + write README.md

# Experiment 3 — create fresh
mkdir -p experiments/03-inductive-discovery-2026-04/{runs,plots,logs}
# + write README.md + spec.md symlink

# Point active symlink
ln -s experiments/03-inductive-discovery-2026-04/runs runs
```

### Orchestrator Log Location

`run_all_local.py` writes logs to `$EXPERIMENT_LOGS_DIR/` when set, else falls back to `logs/`. Set `EXPERIMENT_LOGS_DIR=experiments/03-inductive-discovery-2026-04/logs` in the pod's shell env.

---

## Pre-flight Fixes — Issues Already Resolved

Reviewed `docs/issues.md` and `CLAUDE/known_bugs_and_fixes.md`. Most critical issues are already fixed. Confirmed:

| Issue | Status | Evidence |
|-------|--------|----------|
| #1 Sequential test eval (slow) | **FIXED** | `base.py:206` uses `ThreadPoolExecutor(max_workers=workers)` for test eval |
| #2 PUPA/LiveBench dataset IDs | FIXED | Committed |
| #3 PUPA scoring (substring match) | FIXED | PUPA adapter uses LLM-as-judge + PII leakage check |
| #4 No progress logging during test eval | FIXED | Per-10-example logging added |
| #5 dspy ChainOfThought hangs (no timeout) | FIXED | `timeout=settings.lm_timeout` at base.py:88, 104 |
| #8 DSPy + thinking mode conflict | N/A | Skipping AIME; `enable_thinking: False` set globally |
| #10 BestOfK callback wiring | FIXED (N/A) | Not using BestOfK |
| #11 IFBench substring evaluator (CRITICAL) | FIXED | Programmatic constraint checking |
| #12 HoVer substring evaluator (CRITICAL) | FIXED | `_extract_hover_verdict()` with ordered patterns |
| #13 AIME duplicated questions | N/A | Skipping AIME |
| gepa state FileNotFoundError | FIXED | `os.makedirs(exist_ok=True)` in submodule |
| model_tag substring collision (14b vs 4b) | FIXED | 14b check before 4b |
| vLLM IPC socket path length | FIXED | `cd /tmp` before vLLM launch |

### Infrastructure Confirmed Working

- ✅ Atomic JSON writes (`storage/local.py:_atomic_json_write`)
- ✅ `is_done` resumption check (`run_all_local.py:211`) — if an experiment has completed, orchestrator skips it
- ✅ vLLM health monitor (2-min polling, `run_all_local.py:374`)
- ✅ Stall detection (30-min threshold, kills hung subprocesses)
- ✅ Telegram notifications (25%/50%/75% milestones, failures, completion)
- ✅ `MetricsCollector` unified schema for standalone methods (Slime Mold family uses this)

## Pre-flight Fixes — Still Needed for This Experiment

### Critical

1. **Register all 14 new method names in `run_all_local.py`.**
   - Add to `METHODS` list
   - Add to `METHOD_COMMANDS` dict mapping each to the parameterized iso runner with appropriate CLI flags
   - Add to `METHOD_PRIORITY` fallback ordering
   - Missing registration means orchestrator can't launch the method.

2. **Parameterize the iso runner to accept new flags.**
   - Currently the runner hardcodes personality strategies and blind mutation
   - Add CLI args: `--strategy-mode`, `--k`, `--mutation-mode`, `--refresh-mode`
   - Method name derivation logic (mapping flags to method name for `save_result`)

3. **Verify `is_done` works for new method names.**
   - `is_done` checks for `runs/{model_tag}/{benchmark}/{method}/{seed}/result.json`
   - Should work automatically for new method names — but smoke test verifies

### Important

4. **Smoke test all 5 Tier 1 methods before full launch.**
   - `scripts/run_all_local.py --smoke-test --workers 4 --method <each new method>`
   - Catch parsing/config bugs in the new discovery/mutation code with 5 examples
   - ~10 minutes total on 2x RTX 4090

5. **Verify vLLM health monitor handles dual-endpoint setup.**
   - Currently polls one `GEPA_BASE_URL` — with nginx, all traffic routes through one endpoint (nginx port 8124)
   - If nginx down, health monitor correctly detects failure
   - If one backend vLLM dies, nginx fails over — health monitor doesn't see it directly
   - Add backend health check: `curl localhost:8125/v1/models && curl localhost:8126/v1/models` in pre-flight

6. **Experiment directory migration before first run.**
   - Migrate existing `runs/` and `runs_archive_slurm_sweep/` to new structure
   - Create `experiments/03-inductive-discovery-2026-04/` skeleton with README
   - Update symlink

### Nice to Have

7. **Pre-flight check script** (`scripts/preflight_check.sh`):
   - Verify both vLLM endpoints respond
   - Verify nginx routing works
   - Verify disk space (need ~1GB for runs + logs)
   - Verify `.env` has TELEGRAM_BOT_TOKEN if notifications wanted
   - Run `python scripts/run_all_local.py --dry-run` to print experiment order
   - Exit non-zero if any check fails; orchestrator won't start.

8. **Cross-experiment result schema check.**
   - Add `--validate-schema` mode to `scripts/validate_results.py` that confirms all runs have: `test_score`, `val_score`, `rollout_count`, `val_score_trajectory`, `best_prompt`, `seed_prompt_test_score`
   - Run after Tier 1 completes; flag any malformed results before analysis.

9. **Live dashboard** (`scripts/live_dashboard.py`):
   - Static PNG generation every 5 minutes (simpler than web server)
   - Reads completed results from `experiments/{active}/runs/`
   - Writes to `experiments/{active}/plots/`
   - Can run in a tmux pane on the pod or locally with rsync'd data

### Non-blockers (Defer)

- Issue #15 (Metrics format split) — Slime Mold family uses unified `MetricsCollector`, not affected
- Issue #7 (Contrastive reflection needs large subsets) — not using contrastive_reflection
- Issue #14 (Contrastive AIME regression) — skipping AIME
- Issue #16 (DSPy framework leak) — tolerable; refactor later

---

## Deployment Plan

### Infrastructure

- **Pod:** 2x RTX 4090 on RunPod (~$0.78/hr)
- **vLLM:** 2 instances of Qwen3-8B (bf16), one per GPU
  - GPU 0: port 8125
  - GPU 1: port 8126
- **Load balancing:** nginx round-robin across both ports
- **Workers:** 12 parallel via `run_all_local.py`

### Git Commit Strategy

Results are committed incrementally so partial data is always safe:

1. **Per-benchmark checkpoint:** After all seeds of a benchmark complete for the current method batch, auto-commit:
   ```
   data: {model_tag} {benchmark} — {n} results ({methods_list})
   ```
2. **Per-tier checkpoint:** After Tier 1 completes, commit + push:
   ```
   data: Tier 1 complete — {total} results across {benchmarks}
   ```
3. **Final checkpoint:** After full matrix completes, commit + push.

Implementation: Add a `--git-checkpoint` flag to `run_all_local.py` that triggers `git add runs/ && git commit` after each benchmark batch. Push after each tier.

### Live Dashboard

A monitoring script that auto-refreshes as results land:

**`scripts/live_dashboard.py`** — reads completed `result.json` files, generates:

1. **Progress table:** Methods x benchmarks grid showing completed/total runs with color coding.
2. **Score comparison:** Bar chart of mean +/- std per method per benchmark (updates as seeds complete).
3. **Convergence curves:** Per-benchmark subplot with one line per method, shaded std bands. Built from the `val_score_trajectory` data in each result's metrics.
4. **Variance tracker:** Std dev per method per benchmark — the primary metric we care about.

Runs as a local web server (e.g., `streamlit` or plain HTML with auto-refresh) or generates static PNGs on a configurable interval (default: every 5 minutes).

Lightweight — reads JSON files, uses matplotlib/plotly, no dependencies beyond what we already have.

### Execution Order

1. **Smoke test** (5 examples per combo, ~10 min): Catch bugs in new code before burning GPU time.
2. **Tier 1, HotpotQA first:** This is where Slime Mold has the most variance — fastest signal on whether discovery helps. Git checkpoint after HotpotQA.
3. **Tier 1, remaining benchmarks:** HOVER, PUPA, IFBench. Git checkpoint after each.
4. **Analyze Tier 1 results.** If discovery shows clear wins:
   - Run K=3 arm to test if fewer skills suffice.
   - Run K=adaptive arm to test if the model finds different K per benchmark.
5. **Full matrix** if compute allows.

### Monitoring

- `scripts/runpod_progress.sh` for quick CLI progress checks
- Live dashboard for visual monitoring
- Telegram notifications (already configured): start, 25%/50%/75% milestones, failures, completion

---

## Reporting Template

### Table 1: Score at Equal Budget (primary result)

| Method | HotpotQA | HOVER | PUPA | IFBench | Mean |
|--------|----------|-------|------|---------|------|
| iso (baseline) | X +/- Y | ... | ... | ... | ... |
| iso_prescribed8 | ... | ... | ... | ... | ... |
| iso_inductive_k5_crosspollin | ... | ... | ... | ... | ... |
| iso_inductive_k5_refresh_expand | ... | ... | ... | ... | ... |
| iso_inductive_k5_refresh_replace | ... | ... | ... | ... | ... |

### Table 2: Variance (std dev across seeds)

| Method | HotpotQA | HOVER | PUPA | IFBench |
|--------|----------|-------|------|---------|
| ... | ... | ... | ... | ... |

### Table 3: Convergence Summary

| Method | Seeds converged (of 5) | Avg convergence point | Score at 500 calls |
|--------|----------------------|----------------------|-------------------|

### Figure 1: Convergence Curves

One subplot per benchmark, all methods overlaid, mean +/- std shading. The money plot.

### Table 4: Ablation — What Matters?

| Component | Score delta vs baseline | Variance delta | Worth the complexity? |
|-----------|----------------------|----------------|----------------------|
| Prescribed 8 (vs personality) | +X.XX | ... | Yes/No |
| Inductive discovery (vs prescribed) | +X.XX | ... | Yes/No |
| Cross-pollination (vs blind) | +X.XX | ... | Yes/No |
| Refresh expand (vs no refresh) | +X.XX | ... | Yes/No |
| Refresh replace (vs no refresh) | +X.XX | ... | Yes/No |

### Table 5: Strategy Analysis

| Benchmark | Discovered skills | K value (adaptive) | Most surviving strategy | Failure overlap (pairwise avg) |
|-----------|------------------|-------------------|------------------------|-------------------------------|

---

## Open Questions (to resolve during implementation)

1. **Discovery prompt format:** Numbered list vs JSON. Start with numbered list (simpler for 8B), add JSON fallback if parsing fails frequently.
2. **Cross-pollination prompt section extraction:** How to extract the "relevant section" from a donor prompt. Start with including the full donor prompt (truncated to 500 chars) with the strategy label. Refine if mutation quality is poor.
3. **Pruning schedule for refresh+expand:** When refresh adds candidates to R2, the schedule needs to handle variable pool sizes. Use the same approach: roughly halve each round, adjusting keep-k proportionally.
4. **Existing baseline data:** Reuse qwen3-8b iso results for PUPA and IFBench if evaluation setup is identical. Re-run HotpotQA and HOVER baselines since we don't have them.

## Future Ideas (not in scope)

- **Fixed K experiment:** Run with K fixed at 3, 5, 8 (no adaptive) as a separate ablation.
- **Tournament with inductive discovery:** Apply discovery to Tournament's frozen-pool generation.
- **Multi-perspective discovery:** Run Phase 1 2-3 times with different framings, deduplicate.
- **Larger pool (40):** Scale pool to 40 candidates. Try after free improvements are tested.

---
---

# Review Resolutions (Final)

This section reflects the final design after a bar-raiser review (2026-04-16). Where it conflicts with earlier content in this document, these decisions win.

## Terminology

Use consistently throughout this experiment, in code, logs, and reports:

- **"rollouts"** — evaluation calls on data examples (train/val/test). These are enforced by `--max-metric-calls`.
- **"reflection calls"** — discovery, generation, and mutation LLM calls. Tracked separately in `collector.reflection_call_count`.
- **"total LLM work"** — rollouts + reflection calls + test_eval. The honest resource count.
- **"budget"** — the `--max-metric-calls` flag value. Currently 2,500. Does NOT include reflection calls or test eval.

## Tier 1 — Final 5 Methods

| # | Method | Strategies | Mutation | Refresh | Pool | Cascade |
|---|--------|-----------|----------|---------|------|---------|
| 1 | `iso` | 4 personality | blind | none | 20 | 20→10→5→3→1 |
| 2 | `iso_prescribed8` | 8 universal | blind | none | 25 | 25→13→7→3→1 |
| 3 | `iso_inductive_k5` | inductive K=5 | blind | none | 21 | 21→11→6→3→1 |
| 4 | `iso_inductive_k5_crosspollin` | inductive K=5 | cross-pollination | none | 21 | 21→11→6→3→1 |
| 5 | `iso_inductive_k5_refresh_expand` | inductive K=5 | cross-pollination | expand (+8 at R2) | 21→(R2=19) | 21→11→(+8)=19→10→3→1 |

**Swapped from original Tier 1:** removed `refresh_replace`, added `inductive_k5` (blind). This gives clean isolation:
- 1 vs 2: strategy quality (personality → universal)
- 1 vs 3: inductive discovery alone (no cross-pollination)
- 2 vs 3: task-adaptiveness (prescribed vs discovered)
- 3 vs 4: cross-pollination effect
- 4 vs 5: refresh effect

**Cascade rule:** Halve each round, floor at 3, final = 1. Applied uniformly to all methods including baseline for consistency.

## Final Call Count Estimates (measured + projected)

Per-run rollouts by benchmark (post-cascade standardization, including 200 hold-out trajectory rollouts):

| Method | HotpotQA | HOVER | PUPA | IFBench | Avg |
|--------|----------|-------|------|---------|-----|
| M1 baseline | ~807 | ~1,452 | ~1,042 | ~1,452 | ~1,188 |
| M2 prescribed8 | ~869 | ~1,574 | ~1,164 | ~1,574 | ~1,295 |
| M3 inductive_k5 (blind) | ~814 | ~1,464 | ~1,054 | ~1,464 | ~1,199 |
| M4 inductive_k5_crosspollin | ~814 | ~1,464 | ~1,054 | ~1,464 | ~1,199 |
| M5 inductive_k5_refresh_expand | ~857 | ~1,587 | ~1,177 | ~1,587 | ~1,302 |

**Includes:** cascade rollouts + mutation probes + reflection calls + champion val eval + test eval + 200 hold-out trajectory rollouts (~50 × 4 rounds).

**Budget cap at 2,500 rollouts** gives ~2x headroom even for the most expensive benchmark.

## Tier 1 Total Workload

- 5 methods × 5 seeds × 4 benchmarks = **100 runs**
- Plus 5 methods × 5 extra seeds × HotpotQA = **25 extra runs** (for variance claim statistical power)
- **Total: 125 runs**
- **Total rollouts:** ~120,000-130,000

## Comparison Against GEPA Baseline

Measured GEPA on qwen3-8b:
- IFBench: ~4,090 total LLM calls
- PUPA: ~4,135 total LLM calls

**Our efficiency story:** Slime Mold variants use ~1,050-1,300 rollouts + test eval per run. 3-7x cheaper than GEPA depending on benchmark.

**Caveat:** No HOVER or HotpotQA GEPA data on 8B exists. For those comparisons, either (a) run GEPA on 8B for those benchmarks (~4 hrs × 5 seeds = 20 extra hrs), or (b) frame the paper as Slime Mold variant comparisons with GEPA efficiency noted only where we have data.

## Hardware & Deployment

**Primary choice: 1x A40 48GB on RunPod** (~$0.47/hr)

Rationale:
- Qwen3-8B bf16 fits in 16GB → 32GB free for vLLM KV cache (vs 8GB on RTX 4090)
- Larger batches → estimated 2x throughput vs RTX 4090
- Same or cheaper total cost

**Pre-launch smoke benchmark (required):** Before committing Tier 1:
1. Provision 1x A40 pod
2. Run 1 smoke test (5 examples × 1 method, ~10 min)
3. Measure throughput in calls/sec
4. If ≥1.5x RTX 4090 baseline (0.4+ calls/sec), commit Tier 1 on A40
5. If underperforms, fall back to 1x RTX 4090

**Estimated Tier 1 wall time at A40 throughput (~0.55 calls/sec):**
- 125K rollouts ÷ 0.55 = ~63 hours = ~2.6 days
- Cost: 63 × $0.47 = **~$30**

Fallback 1x RTX 4090: ~125 hours = ~5 days at ~$47. Both cheaper than original 2x4090 plan.

## Final Decisions by Review Item

### Blockers (all resolved)

**B1. Stall detection:** Runner writes `gepa_state/progress.json` after each round and mutation batch with current `rollouts_used`. Adds hard 2-hour wall-clock timeout per subprocess in `run_all_local.py`. Both mechanisms active simultaneously.

**B2. Method name derivation:** Build `_derive_method_name(strategy_mode, k, mutation_mode, refresh_mode) → str` helper in runner. All 14 combinations produce unique directory names. Unit test validates this. `save_result(method=derived_name)`.

**B3. Budget definition:** Separate tracking, rollouts-only enforcement. `max_metric_calls` caps rollouts only (2,500). `reflection_call_count` tracked but not capped. Every trajectory point logs both (`cumulative_rollouts`, `cumulative_reflection_calls`, `best_score`).

**B4. Cross-pollination failure matrix:**
- **Threshold:** Binary, 0.5 (scores ≥0.5 pass, <0.5 fail)
- **Scope:** Reset per round; matrix built from that round's eval only
- **Tiebreaker:** Prefer cross-strategy donor (different strategy from survivor), break ties by highest score on survivor's failed examples
- **Logging:** Every mutation records `cross_pollination_events` entry with full metadata (survivor/donor strategies, shared failures, `cross_strategy: bool`, `no_donor_found: bool`)
- **Fallback:** If no donor exists (all candidates failed on same examples), fall back to blind mutation with `no_donor_found: true` flag
- **Implementation:** Modify `run_pruning_round` to return per-example scores. Build matrix post-evaluation.

**B5. Tier 1 isolation:** Done via Tier 1 final list above.

### Important (all resolved)

**I1. Sample size:** 10 seeds on HotpotQA, 5 on other benchmarks. Paired analysis (same seed across methods) for "does X help" claims. If budget gets tight, drop HotpotQA to 5 seeds.

**I2. Hold-out trajectory:** 50 fixed trainset examples sampled at run start (same across seeds for a benchmark). Evaluate best-so-far prompt on these 50 at end of each round + at run start. Adds ~200 rollouts/run. Gives comparable trajectory points across methods.

**I3. Call counts:** Rebuilt from measurements (see table above). Budget = 2,500 rollouts.

**I4. Adaptive K preflight probe:** Before committing methods 11-14 (full matrix K=adaptive arm), run `discover_strategies(k=None)` 3 times each on all 4 benchmarks. If K range across benchmarks < 2, skip methods 11-14 and document "adaptive K is not actually task-adaptive on 8B" as a finding.

**I5. Discovery quality preflight:** Manual inspection. 3 benchmarks × 3 seeds = 9 discovery outputs, read by hand. If outputs are benchmark-specific and concrete (e.g., "multi-hop entity tracking" vs "careful reading"), proceed. If generic/uniform, **stop and reconsider** — don't proceed on a known failure mode. Include 1-2 example discoveries per benchmark in the final report.

**I6. Refresh pool management:** R1 prunes normally (21→11 for K=5). Refresh generates ~8 new candidates. R2 pool = 11 survivors + 8 new = 19. All 19 re-evaluated fresh on 15 R2 examples. Simple, clean, priors ignored.

**I7. Cascade standardization:** Halve each round, floor at 3, final = 1. Applied to all methods.

**I8. Re-run baselines:** Don't reuse old qwen3-8b iso data. All methods run fresh against current evaluator versions.

**I9. Single-backend vLLM:** Moot with 1x GPU. If we upgrade to 2 GPUs later, add per-backend health polling to `_health_monitor`.

### Minor & Nitpick (all resolved)

**M1:** Leave `task_description` as-is.

**M2:** Discovery parser retries once on <K outputs. If still <K, fall back to prescribed-8 strategies with `discovery_fallback: true` in metadata and a loud log warning.

**M3:** Migration in own commit. Tag `pre-experiment-3` before migration.

**M4:** Add `--runs-dir` flag to `run_all_local.py` for explicit experiment path override. Symlink remains default.

**M5:** `hard_example_threshold` configurable, default 0.7. Log `hard_examples_found_count` per refresh.

**M6:** Telegram final summary filters to methods that produced results in this run.

**M7, M8:** Full spec consistency pass done via this section.

**N1:** Test eval uses `workers=8`. Logged in config snapshot.

**N2:** Same 5 seeds across experiments accepted as within-experiment consistency measure.

**N3:** Money plot spec formalized: x=cumulative_rollouts, y=best_score_so_far (on hold-out), one line per method (mean across seeds), ±1σ shading, one subplot per benchmark. Implemented in `scripts/live_dashboard.py`.

**N4:** Terminology standardized (see Terminology section above).

**N5:** `analysis/` at repo root = cross-experiment comparison plots only. Empty during this experiment; populated once experiment completes and we want comparisons vs prior experiments.

## Data Collection Schema

Every run's `metrics.json` must include (for analysis integrity):

```python
{
    # Existing fields (keep all)
    "rollout_count": int,
    "reflection_call_count": int,
    "test_score": float,
    "val_score": float,
    "val_score_trajectory": list[{
        "iteration": int,
        "cumulative_rollouts": int,
        "cumulative_reflection_calls": int,
        "score_on_holdout": float,
        "best_so_far": float,
        "prompt_length": int,
    }],
    
    # New required fields
    "method": str,               # derived method name
    "early_stopped": false,      # always false (we removed early stopping)
    "convergence_round": int,    # round at which best_score stopped improving (for analysis)
    "discovery_fallback": bool,  # true if discovery failed and we padded with prescribed-8
    "discovery_outputs": list[{  # raw discovery LLM outputs per discovery call
        "pass": "initial" | "refresh",
        "k_requested": int | None,
        "skills": list[{"name": str, "description": str, "failure_pattern": str}],
        "raw_llm_output": str,   # for manual inspection
    }],
    "hard_examples_found_count": int,  # only for refresh variants
    "cross_pollination_events": list[{
        "round": int,
        "survivor_candidate_idx": int,
        "survivor_strategy": str,
        "survivor_score": float,
        "survivor_failed_examples": list[int],
        "donor_candidate_idx": int | None,
        "donor_strategy": str | None,
        "shared_failures_covered": int,
        "donor_score_on_failures": float | None,
        "cross_strategy": bool,
        "no_donor_found": bool,
    }],
}
```

## Pre-Flight Checklist (run before Tier 1)

1. **Code ready:**
   - [ ] `_derive_method_name` helper with unit test covering all 14 combinations
   - [ ] Runner accepts `--strategy-mode`, `--k`, `--mutation-mode`, `--refresh-mode` flags
   - [ ] Progress writer (writes `gepa_state/progress.json` after each round)
   - [ ] Hard 2-hour wall-clock timeout in `run_all_local.py`
   - [ ] Discovery function with fallback to prescribed-8 on failure
   - [ ] Cross-pollination failure matrix + event logging
   - [ ] Hold-out trajectory evaluator (50 examples, evaluated per round)
   - [ ] All 14 method names registered in `METHODS` and `METHOD_COMMANDS`
   - [ ] `--runs-dir` flag for explicit experiment path

2. **Infrastructure:**
   - [ ] A40 throughput smoke benchmark complete
   - [ ] vLLM server running with `enable_thinking: False`
   - [ ] `.env` has `TELEGRAM_BOT_TOKEN`
   - [ ] `EXPERIMENT_LOGS_DIR` env var set

3. **Data migration:**
   - [ ] Tag `pre-experiment-3` created
   - [ ] `experiments/01-slurm-sweep-2026-03/` created with README
   - [ ] `experiments/02-runpod-baseline-2026-04/` created with README  
   - [ ] `experiments/03-inductive-discovery-2026-04/` created with README, spec.md symlink
   - [ ] Root `runs/` symlinks to active experiment

4. **Quality preflight:**
   - [ ] Run inductive discovery 9 times (3 benchmarks × 3 seeds), manually inspect outputs
   - [ ] Decision point: are discoveries benchmark-specific and concrete? **If no, STOP.**
   - [ ] Smoke test all 5 Tier 1 methods on HotpotQA (5 examples each)
   - [ ] Verify `is_done` correctly identifies new method names (won't re-run completed)

5. **Execution order once greenlit:**
   - [ ] Tier 1 on HotpotQA first (10 seeds, 5 methods = 50 runs, ~4-6 hours)
   - [ ] Git commit after HotpotQA complete
   - [ ] Then HOVER, PUPA, IFBench (5 seeds each, 15 runs each, ~8 hours each)
   - [ ] Git commit after each benchmark
   - [ ] Run `validate_results.py --validate-schema` after Tier 1 complete
   - [ ] Generate analysis plots to `experiments/03-.../plots/`
   - [ ] Write `report.md` with findings
   - [ ] Tag `exp-03-complete` when done
