# Design: Inductive Strategy Discovery for Slime Mold

**Date:** 2026-04-15
**Model:** Qwen3-8B (bf16) — GEPA paper's primary model
**Deployment:** RunPod, 2x RTX 4090
**Budget:** 2,000 LLM calls per run

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

### Method 1: `slime_mold` (existing baseline)

- **Strategies:** 4 personality (Analytical, Creative, Minimal, Expert)
- **Pool:** 4 strategies x 5 prompts = 19 + seed = 20
- **Mutation:** Blind (no strategy context)
- **Refresh:** None
- **Calls:** ~652
- **Purpose:** Control. Already have partial data for qwen3-8b (PUPA, IFBench).

### Method 2: `slime_mold_prescribed8`

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

### Method 3: `slime_mold_inductive_k5_crosspollin`

- **Strategies:** Inductive discovery (K=5 fixed). LLM examines 8-10 seed-dependent train examples, identifies 5 skills with failure patterns.
- **Pool:** 5 skills x 4 prompts = 20 + seed = 21
- **Mutation:** Strategy-aware with cross-pollination. After evaluation, build failure matrix. For each survivor being mutated, find another candidate that succeeded where it failed. Include that donor's strategy name and relevant prompt section in the mutation prompt.
- **Refresh:** None
- **Calls:** ~664
- **Purpose:** The core experiment. Does task-adaptive generation + informed mutation help?

### Method 4: `slime_mold_inductive_k5_refresh_expand`

- **Strategies:** Same as Method 3
- **Pool:** 21 initially, grows after refresh
- **Mutation:** Cross-pollination
- **Refresh:** After R1, collect "hard examples" — examples that >=70% of candidates failed on (threshold configurable via runner arg `hard_example_threshold`, default 0.7). Run focused discovery ("what skills are missing?"). Generate 4 prompts per new skill (expect 2 new skills = 8 candidates). New candidates JOIN the survivor pool for R2 (R2 pool: ~10 survivors + ~8 new = ~18).
- **Calls:** ~787
- **Purpose:** Does a second discovery pass catch missed skills? Does expanding the pool help?

### Method 5: `slime_mold_inductive_k5_refresh_replace`

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
| 1 | `slime_mold` | 4 personality | blind | none | 20 | ~652 |
| 2 | `slime_mold_prescribed8` | 8 universal | blind | none | 25 | ~748 |

### K=3 — small pool, tests if few high-quality strategies suffice (4 methods)

| # | Method | Mutation | Refresh | Pool | Calls |
|---|--------|----------|---------|------|-------|
| 3 | `slime_mold_inductive_k3` | blind | none | 13 | ~457 |
| 4 | `slime_mold_inductive_k3_crosspollin` | cross-pollination | none | 13 | ~457 |
| 5 | `slime_mold_inductive_k3_refresh_expand` | cross-pollination | expand | 13->~21 | ~636 |
| 6 | `slime_mold_inductive_k3_refresh_replace` | cross-pollination | replace | 13 | ~460 |

**What K=3 tests:** Can 3 well-chosen skills match 5? If so, we get comparable quality at 30% fewer calls.

**Cascade (K=3, pool=13):** 13 -> 7 -> 4 -> 2 -> 1

### K=5 — matches current pool size, cleanest comparison (4 methods)

| # | Method | Mutation | Refresh | Pool | Calls |
|---|--------|----------|---------|------|-------|
| 7 | `slime_mold_inductive_k5` | blind | none | 21 | ~664 |
| 8 | `slime_mold_inductive_k5_crosspollin` | cross-pollination | none | 21 | ~664 |
| 9 | `slime_mold_inductive_k5_refresh_expand` | cross-pollination | expand | 21->~29 | ~787 |
| 10 | `slime_mold_inductive_k5_refresh_replace` | cross-pollination | replace | 21 | ~667 |

Tier 1 methods 3/4/5 correspond to full matrix #8, #9, #10.

**Cascade (K=5, pool=21):** 21 -> 10 -> 5 -> 3 -> 1

### K=adaptive — LLM decides how many skills exist (4 methods)

| # | Method | Mutation | Refresh | Pool | Calls |
|---|--------|----------|---------|------|-------|
| 11 | `slime_mold_inductive_kadaptive` | blind | none | 4K+1 | ~747 |
| 12 | `slime_mold_inductive_kadaptive_crosspollin` | cross-pollination | none | 4K+1 | ~747 |
| 13 | `slime_mold_inductive_kadaptive_refresh_expand` | cross-pollination | expand | varies | ~870 |
| 14 | `slime_mold_inductive_kadaptive_refresh_replace` | cross-pollination | replace | 4K+1 | ~750 |

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
def run_slime_mold(
    benchmark, seed,
    strategy_mode="personality",  # "personality" | "prescribed8" | "inductive"
    k=None,                       # 3 | 5 | None (adaptive). Ignored for personality/prescribed8.
    mutation_mode="blind",        # "blind" | "crosspollin"
    refresh_mode="none",          # "none" | "expand" | "replace"
    subset=None, max_metric_calls=None, settings=None,
) -> ExperimentResult:
```

**Method name derivation:**
- `strategy_mode="personality"` -> `slime_mold`
- `strategy_mode="prescribed8"` -> `slime_mold_prescribed8`
- `strategy_mode="inductive", k=5, mutation_mode="crosspollin", refresh_mode="none"` -> `slime_mold_inductive_k5_crosspollin`
- etc.

**Pruning schedule scales with pool size:**
- Pool <= 15: 4 rounds, cascade ~halving (13 -> 7 -> 4 -> 2 -> 1)
- Pool 16-25: 4 rounds (21 -> 10 -> 5 -> 3 -> 1)
- Pool 26-35: 4 rounds (29 -> 14 -> 6 -> 3 -> 1)

### 7. Orchestrator Registration (`run_all_local.py`)

Add all 14 method names to `METHODS` list and `METHOD_COMMANDS` dict. Each maps to the same runner script with different CLI flags.

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
| slime_mold (baseline) | X +/- Y | ... | ... | ... | ... |
| slime_mold_prescribed8 | ... | ... | ... | ... | ... |
| slime_mold_inductive_k5_crosspollin | ... | ... | ... | ... | ... |
| slime_mold_inductive_k5_refresh_expand | ... | ... | ... | ... | ... |
| slime_mold_inductive_k5_refresh_replace | ... | ... | ... | ... | ... |

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
4. **Existing baseline data:** Reuse qwen3-8b slime_mold results for PUPA and IFBench if evaluation setup is identical. Re-run HotpotQA and HOVER baselines since we don't have them.

## Future Ideas (not in scope)

- **Fixed K experiment:** Run with K fixed at 3, 5, 8 (no adaptive) as a separate ablation.
- **Tournament with inductive discovery:** Apply discovery to Tournament's frozen-pool generation.
- **Multi-perspective discovery:** Run Phase 1 2-3 times with different framings, deduplicate.
- **Larger pool (40):** Scale pool to 40 candidates. Try after free improvements are tested.
