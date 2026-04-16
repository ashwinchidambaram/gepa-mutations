# Experiment Design: Comparing Prompt Optimization Methods

## Goal

Determine which prompt optimization method produces the best prompts for the least cost, with honest convergence analysis to validate results.

## Methods Under Test

### Existing (already implemented)

| Method | Description | Approx LLM calls | Key trait |
|--------|-------------|-------------------|-----------|
| **GEPA (K=1)** | Single-lineage iterative refinement. Reflect on failures → mutate → evaluate → repeat. | ~400 | Depth-first, sequential |
| **Best-of-K** | Like GEPA but generates K candidates per iteration, keeps best | ~400-800 (varies by K) | Wider per-iteration search |
| **Failure-Stratified-K** | K candidates each see different failure subsets | ~400-800 | Diverse failure exposure |
| **Slime Mold (current)** | 20 candidates, personality-based strategies, 4-round cascade with mutation | ~577 | Breadth-first, parallelizable |
| **Tournament** | 64 candidates, single-elimination bracket, no mutation | ~967 | Maximum initial diversity, frozen pool |
| **Contrastive Reflection** | Mine success/failure pairs from history, inject into reflection | ~400 (same as GEPA) | Better reflection signal, zero extra cost |
| **Synaptic Pruning** | Generate overspecified → ablate sections → prune → strengthen | varies | Component-level surgery |

### Proposed (new ideas)

| Method | Description | Extra cost vs Slime Mold | Proposal doc |
|--------|-------------|--------------------------|--------------|
| **Prescribed Strategies (8)** | Replace personality-based with 8 universal problem-solving strategies (Divide & Conquer, Working Backward, Analogy, Abstraction, Root Cause Analysis, Trial & Error, Constraint Satisfaction, Lateral Thinking) | 0 extra calls | `strategy-aware-mutations.md` |
| **Strategy-Aware Mutation** | Mutation LLM knows the strategy that generated the prompt → targeted improvements + cross-pollination | 0 extra calls | `strategy-aware-mutations.md` |
| **Inductive Discovery** | LLM examines benchmark examples, discovers task-specific skills, generates one prompt per skill | ~3-5 extra calls | `inductive-strategy-discovery.md` |
| **Inductive + Refresh** | Discovery + one refresh pass after Round 1 using failure data | ~5-7 extra calls | `inductive-strategy-discovery.md` |
| **Full Hybrid** | Inductive discovery + strategy-aware mutation + cross-pollination + 1 refresh | ~5-7 extra calls | both docs |
| **Larger Pool (40)** | Double candidate pool from 20 to 40 | ~275-575 extra evaluation calls | `larger-slime-mold-pool.md` |

### Composition Matrix

These ideas are modular. Test them independently AND in combination:

```
                          Personality  Prescribed-8  Inductive
Blind mutation            baseline     A             C
Strategy-aware mutation   —            B             D (full hybrid)
+ refresh                 —            —             E
+ larger pool (40)        —            B+pool        E+pool
```

## Comparison Axes

### Axis 1: Score at Fixed Budget (PRIMARY)

> "Given exactly N LLM calls, which method gets the highest test score?"

This is the fairest comparison. All methods get the same total budget of LLM calls (generation + mutation + evaluation — all count equally).

**How to determine N:** Pilot run (see Convergence Methodology below).

**Report as:** Mean +/- std across 5 seeds per benchmark.

### Axis 2: Budget to Reach Target Score

> "How many LLM calls to reach X% of GEPA's final score?"

Shows efficiency for "good enough" use cases. Particularly relevant if our methods converge faster but to a lower ceiling.

**Report as:** Median calls across 5 seeds, with min/max range.

### Axis 3: Ceiling (Best Score Regardless of Budget)

> "If we gave each method unlimited budget, what's the highest it can achieve?"

Tests whether the method's structure limits its potential. GEPA's many iterations might find prompts that Slime Mold's 4 rounds never could.

**Report as:** Best score observed across all seeds at maximum budget.

### Axis 4: Score Variance Across Seeds

> "How consistent is each method?"

A method scoring 0.82 +/- 0.01 across 5 seeds beats one scoring 0.84 +/- 0.08. Consistency matters for production use.

**Report as:** Standard deviation across seeds.

### Axis 5: Wall-Clock Time (Parallelism)

> "How long does this take with P parallel workers?"

Slime Mold/Tournament can parallelize evaluation within each round. GEPA is inherently sequential. At equal budgets, parallel methods may finish much faster.

**Report as:** Sequential rounds/iterations required (not actual time — this is hardware-independent). Fewer sequential steps = more parallelizable.

## Convergence Methodology

### Why Convergence Curves Matter

Without convergence data, we don't know if:
- A method was still improving when the budget ran out (result is inconclusive)
- A method converged early and wasted remaining budget (could have stopped sooner)
- One method converges fast to a low ceiling while another converges slow to a high ceiling

### What to Log

Every method logs this at every step (round for Slime Mold, iteration for GEPA):

```python
{
    "method": "slime_mold_inductive",
    "benchmark": "hotpotqa",
    "seed": 42,
    "step": 3,
    "cumulative_llm_calls": 340,   # ALL calls count equally
    "best_score_so_far": 0.78,
    "improvement_from_last_step": 0.03,
    "num_active_candidates": 5,    # survivors / frontier size
    "step_type": "evaluation",     # or "mutation", "generation"
}
```

### Pilot Run Protocol

Before the full experiment:

1. Pick 1 seed, 1 benchmark (e.g., HotpotQA seed 42)
2. Run every method with a **generous budget** (~2x what you plan to use)
3. Plot convergence curves for all methods on the same axes
4. Identify where each method's curve flattens
5. Set experiment budget to be comfortably past the LATEST convergence point
6. If any method hasn't flattened by 2x budget, either increase budget or note that method needs more calls than we're willing to spend

### Convergence Validity Check

After the full experiment, for each run:

**Converged:** Improvement < 1% of current score for 2 consecutive steps before budget ran out.

**Possibly converged:** Improvement < 1% for the last step only (ambiguous).

**Not converged:** Still improving >1% when budget ran out (inconclusive for this seed).

Report the count: "4/5 seeds converged for Slime Mold; 3/5 for GEPA." If a method frequently doesn't converge within budget, the comparison is unfair to that method — either increase budget or caveat the results.

### The Money Plot

One figure per benchmark:

```
x-axis: cumulative LLM calls (0 to budget)
y-axis: best test score so far
lines:  one per method (mean across seeds)
bands:  +/- 1 std dev shading

    Score
    0.85 │               ╭─── GEPA (slow start, high ceiling?)
    0.80 │         ╭─────┤
    0.75 │   ╭─────╯     ╰─── Hybrid (fast start, ???)
    0.70 │  ╱╱
    0.65 │╱╱
         └────────────────────────
         0   200   400   600   800
              Cumulative LLM calls
```

This single chart answers almost every question:
- Which method is winning at any given budget?
- Where does each method plateau?
- Which is more consistent? (narrower bands)
- Is the experiment budget sufficient? (curves flat at the right edge)

## Reporting Template

### Table 1: Score at Equal Budget (primary result)

| Method | HotpotQA | IFBench | AIME | Mean |
|--------|----------|---------|------|------|
| GEPA (K=1) | 0.XX +/- 0.XX | ... | ... | ... |
| Slime Mold (personality) | ... | ... | ... | ... |
| Slime Mold (prescribed 8) | ... | ... | ... | ... |
| Slime Mold (inductive) | ... | ... | ... | ... |
| Slime Mold (full hybrid) | ... | ... | ... | ... |
| Tournament | ... | ... | ... | ... |

### Figure 1: Convergence Curves

One subplot per benchmark, all methods overlaid. The most important figure.

### Table 2: Budget to Reach 80% of Best Score

| Method | HotpotQA | IFBench | AIME |
|--------|----------|---------|------|
| GEPA | N calls | ... | ... |
| Slime Mold (hybrid) | N calls | ... | ... |
| ... | ... | ... | ... |

### Table 3: Convergence Summary

| Method | Seeds converged (of 5) | Avg convergence point | Final improvement rate |
|--------|----------------------|----------------------|----------------------|
| GEPA | 4/5 | call 2,100 | 0.2% |
| Slime Mold (hybrid) | 5/5 | call 420 | 0.1% |
| ... | ... | ... | ... |

### Table 4: Strategy Analysis (our methods only)

| Strategy | Survival rate | Avg champion lineage | Failure overlap with nearest strategy |
|----------|--------------|---------------------|--------------------------------------|
| Divide & Conquer | 40% | 2/10 champions | 35% with Abstraction |
| Working Backward | 25% | 1/10 champions | 20% with Root Cause |
| ... | ... | ... | ... |

### Table 5: Ablation — What Matters?

| Component | Score delta vs baseline | Worth the complexity? |
|-----------|----------------------|----------------------|
| Prescribed 8 (vs personality) | +X.XX | Yes/No |
| Strategy-aware mutation (vs blind) | +X.XX | Yes/No |
| Inductive discovery (vs prescribed) | +X.XX | Yes/No |
| Refresh (vs no refresh) | +X.XX | Yes/No |
| Cross-pollination (vs same-strategy) | +X.XX | Yes/No |
| Larger pool 40 (vs 20) | +X.XX | Yes/No |

## Experiment Matrix

### Full experiment (after pilot confirms budget)

3 benchmarks x 5 seeds x N methods:

**Benchmarks:** HotpotQA, HOVER, PUPA, IFBench (skip AIME — model can't solve it; skip LiveBench — no method separation)

**Seeds:** 42, 123, 456, 789, 1024

**Methods (ordered by priority):**

*Tier 1 — run first (baselines + highest-value new idea):*
1. GEPA K=1 (existing baseline — likely have data already)
2. Slime Mold — personality strategies (existing baseline — likely have data already)
3. Slime Mold — inductive discovery, blind mutation (isolate discovery effect)
4. Slime Mold — inductive discovery + strategy-aware mutation + 1 refresh (full hybrid)

*Tier 2 — run next (ablations to understand what matters):*
5. Slime Mold — prescribed 8 strategies, blind mutation (compare prescribed vs inductive)
6. Slime Mold — prescribed 8, strategy-aware mutation (isolate mutation effect)

*Tier 3 — run if compute allows:*
7. Tournament — inductive discovery (does discovery help frozen-pool methods too?)
8. Tournament — prescribed 8 strategies

**Rationale:** Inductive Strategy Discovery is the most novel contribution and the hardest to predict from first principles. Prescribed strategies are a known quantity — we can reason about them without running experiments. Discovery might find strategies we'd never think of, so we prioritize learning whether that's true.

**Reuse existing data:** Methods 1-2 likely already have results from previous sweeps at the same seeds. Reuse if the evaluation setup is identical — don't waste compute re-running baselines.

### Ablation order

The methods are ordered to isolate each variable:
- 2 vs 3: effect of better strategies (personality → prescribed 8)
- 3 vs 4: effect of strategy-aware mutation
- 3 vs 5: effect of inductive discovery (vs prescribed)
- 5 vs 6: effect of mutation awareness + refresh + cross-pollination
- Any method vs same+pool: effect of larger pool

## Learnings from Previous Experimental Sweep (qwen3-1.7b)

We ran 6 benchmarks × 6 methods × 5 seeds on qwen3-1.7b. Key findings that shape this experiment:

### Method Performance Summary

| Benchmark | Best Method | Score | vs Baseline | Notable |
|-----------|------------|-------|-------------|---------|
| HotpotQA | Slime Mold | 0.719 | +14% | Highest ceiling but huge variance |
| HOVER | GEPA | 0.50 | +26% | Steady convergence |
| PUPA | GEPA / Tournament | 0.49 | +44% | Synaptic Pruning wildly unstable |
| IFBench | Contrastive Refl. | 0.101 | +48% | All methods show modest gains |
| LiveBench | Tournament | 0.191 | +12% | Tight clustering, minimal separation |
| AIME | Nothing | ~0.01 | — | 1.7B can't solve math problems |

No single method dominates. Each benchmark rewards a different approach.

### Convergence Behavior

- Most methods plateau by ~1,500 calls. GEPA keeps climbing to ~4,000 on some benchmarks but with diminishing returns after 1,500.
- Slime Mold converges fastest (by ~300-500 calls) but has the **widest variance bands** — some seeds near-perfect, others barely beat baseline.
- Tournament shows stepped convergence with occasional wild spikes (one seed finding a great candidate).
- Synaptic Pruning is unstable — sometimes spikes dramatically then crashes back. High variance, sometimes hurts.
- GEPA and Contrastive Reflection show the most steady, predictable improvement curves.

### Implications for This Experiment

1. **Budget: 2,000 calls is sufficient.** Most methods plateau by 1,500. We don't need to match GEPA's full 4,000 — we need to show we converge to a comparable score faster.

2. **Primary metric: variance reduction.** Slime Mold already has competitive ceilings. If inductive discovery tightens the variance bands (same mean, lower std), that's a strong result — it means the method is reliable, not just lucky.

3. **Report score-at-500-calls.** This is roughly Slime Mold's current budget and where real separation between methods appears. "At 500 calls, inductive Slime Mold scores X ± Y vs personality Slime Mold at X ± Y" is the cleanest headline comparison.

4. **Benchmarks: 4, not 6.** Skip AIME (model can't do it) and LiveBench (no method separation). Focus on HotpotQA, HOVER, PUPA, IFBench — these show real differentiation between methods.

5. **Watch for Synaptic Pruning syndrome.** Our pipeline adds complexity (discovery, refresh, cross-pollination). Complexity can introduce instability. Track per-seed results, flag any seed where our method hurts vs baseline, and investigate why.

6. **Slime Mold's variance is the #1 opportunity.** The convergence curves show Slime Mold has the right structure (parallel, fast) but inconsistent candidate quality. Better strategies should directly address this — more relevant candidates entering the cascade = less dependence on getting lucky with the seed.

## Open Questions

1. **What model for experiments?** Qwen3-1.7B has the most baseline data for comparison. Also test on 8B to check if gains transfer to larger models?
2. **Reuse existing data?** Previous sweep results at same seeds serve as baselines 1-2. Reuse if evaluation setup is identical — don't waste compute re-running.
3. **How to measure "strategy relevance"?** Need a concrete metric for whether discovered strategies actually target different skills vs just sounding different (the same problem personality strategies had).
