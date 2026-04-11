# Mutation Plan: best_of_k

**Date:** 2026-03-19
**Status:** Ready for implementation
**Priority:** 1 (first mutation to run -- only mutation with validated positive signal)

---

## Overview

Instead of proposing a single mutated prompt per GEPA iteration, generate K independent mutations from the same parent candidate and reflective dataset, evaluate all K on the same minibatch, and keep only the best-scoring one. This converts GEPA's 1-of-1 proposal into a K-of-K selection, increasing the probability that at least one mutation improves on the parent without changing any other algorithmic component.

**Prior result:** K=3 on AIME (budget=50, gpt-oss-20b): 0.400 vs baseline 0.100. This is the only mutation in the portfolio with a validated positive signal.

---

## 1. Implementation Approach

### Step-by-step code changes

**Step 1: Create `BestOfKProposer` class**

Create `src/gepa_mutations/experiments/best_of_k/proposer.py` containing a `BestOfKProposer` that subclasses (or wraps) GEPA's `ReflectiveMutationProposer`. The subclass overrides `propose()` to insert the K-loop around steps 5-6 of the original flow.

The K-loop replaces the single `propose_new_texts() -> evaluate` sequence:

```
# Original flow (steps 5-6):
new_texts = self.propose_new_texts(curr_prog, reflective_dataset, components_to_update)
new_candidate = build_candidate(curr_prog, new_texts)
new_scores = evaluate(new_candidate, minibatch)
return CandidateProposal(candidate=new_candidate, ...)

# best_of_k flow:
best_candidate = None
best_score = -inf
seen_hashes = set()

for k in range(K):
    new_texts = self.propose_new_texts(curr_prog, reflective_dataset, components_to_update)
    text_hash = hash(frozenset((k, v) for k, v in new_texts.items()))
    if text_hash in seen_hashes:
        continue  # deduplication (Bug #2 fix)
    seen_hashes.add(text_hash)

    new_candidate = build_candidate(curr_prog, new_texts)
    new_scores = evaluate(new_candidate, minibatch)
    score_sum = sum(new_scores)

    # emit per-k evaluation callback (Bug #1 fix)
    emit_evaluation_event(k, new_scores)

    if score_sum > best_score:
        best_score = score_sum
        best_candidate = new_candidate
        best_scores = new_scores

return CandidateProposal(candidate=best_candidate, ...)
```

**Step 2: Create `runner.py`**

Create `src/gepa_mutations/experiments/best_of_k/runner.py` that:
1. Constructs a `MutationConfig` with `mutation_candidates` field
2. Instantiates the `BestOfKProposer` (passing K from config)
3. Patches it into the optimize() call via `custom_candidate_proposer` or by constructing the engine directly
4. Calls `run_mutation()` from `base.py`

**Step 3: Integrate with `optimize()` API**

The cleanest integration path is to create a wrapper function that:
1. Calls the same setup as `optimize()` (data loading, LM construction, adapter creation)
2. Constructs a `BestOfKProposer` instead of the default `ReflectiveMutationProposer`
3. Passes it to `GEPAEngine` directly

This avoids monkey-patching `optimize()` while reusing all its configuration logic.

**Step 4: Add `BestOfKMetricsCallback`**

Extend `MetricsCallback` to track per-K diagnostics:
- Number of unique candidates generated per iteration (after deduplication)
- Score of each K candidate (not just the winner)
- Which K index won (to detect if early K candidates are systematically better/worse)
- Deduplication rate per iteration

**Step 5: Create sweep script**

Create `src/gepa_mutations/experiments/best_of_k/sweep.py` that runs the full parameter sweep (K in [1, 3, 5, 7]) across selected benchmarks and seeds.

---

## 2. GEPA Components to Modify

### Approach: Clean extension, no patches to `gepa/`

The `gepa/` directory is the official GEPA source pinned at v0.1.1 and must not be modified. All changes live in `src/gepa_mutations/`.

| Component | Action | Rationale |
|-----------|--------|-----------|
| `ReflectiveMutationProposer` | **Subclass** (not patch) | Override `propose()` to add K-loop. Inherit `propose_new_texts()`, all constructor params, callback wiring. |
| `GEPAEngine` | **Direct construction** | Bypass `optimize()` wrapper to inject custom proposer. Reuse all other engine setup from `optimize()`. |
| `CandidateProposal` | **Use as-is** | Return type unchanged. Add K-metadata via the existing `metadata` dict field. |
| `MutationConfig` | **Extend** | Add `mutation_candidates: int` field. |
| `MetricsCallback` | **Extend** | Subclass to add K-specific tracking. |
| `run_mutation()` | **Extend** | Add code path to detect `mutation_candidates > 1` and swap in `BestOfKProposer`. |

### Files to create

```
src/gepa_mutations/experiments/best_of_k/
    __init__.py
    mutation_plan.md          # this file
    proposer.py               # BestOfKProposer
    runner.py                 # MutationConfig construction + run_mutation()
    sweep.py                  # parameter sweep orchestration
    callbacks.py              # BestOfKMetricsCallback
    test_proposer.py          # unit tests
    test_integration.py       # integration tests
```

### Files to modify

```
src/gepa_mutations/base.py   # add mutation_candidates field to MutationConfig
```

---

## 3. New MutationConfig Fields

### Addition to `MutationConfig` in `base.py`

```python
@dataclass
class MutationConfig:
    # ... existing fields ...

    # best_of_k mutation parameters
    mutation_candidates: int = 1
    """Number of independent mutations to propose per iteration (K).
    K=1 is equivalent to vanilla GEPA. K>1 generates K mutations from
    the same parent and reflective dataset, evaluates all K on the same
    minibatch, and keeps only the best-scoring one."""
```

| Field | Type | Default | Validation | Rationale |
|-------|------|---------|------------|-----------|
| `mutation_candidates` | `int` | `1` | Must be >= 1. Values > 10 should warn (diminishing returns, budget blow-up). | K=1 recovers vanilla GEPA, making the field backward-compatible. The name `mutation_candidates` is preferred over `k` for clarity in config dumps. |

### Validation rules

1. `mutation_candidates >= 1` -- enforced in `run_mutation()` before constructing proposer
2. `mutation_candidates > 10` -- emit a warning (each additional K costs ~1 LLM call + 1 minibatch evaluation per iteration; at K=10 this is already 10x the reflection cost)
3. When `mutation_candidates == 1`, use the standard `ReflectiveMutationProposer` (no overhead)

---

## 4. Parameter Sweep Values

### Primary sweep: `mutation_candidates`

| K | Justification |
|---|---------------|
| **1** | Control condition. Must exactly reproduce vanilla GEPA paper baseline. Any deviation indicates a bug in the best_of_k infrastructure (since K=1 should be a no-op). |
| **3** | Prior validated signal (K=3 on AIME: 0.400 vs 0.100). This is the highest-priority data point for confirmation. |
| **5** | Tests whether gains continue to scale beyond K=3. If K=5 >> K=3, the mutation is strongly compute-scaling; if K=5 ~ K=3, diminishing returns set in early. |
| **7** | Upper bound probe. At K=7, each iteration uses 7x the reflection LLM calls and 7x the minibatch evaluations. Tests whether the increased cost is justified. If K=7 ~ K=5, we know K=5 is sufficient and can save compute on failure_stratified_k. |

### Why not K=2, 4, 6?

- K=2 is too close to K=1 to reliably detect a signal at 3 seeds
- K=4, K=6 are interior points that add sweep cost without answering the key questions (does K>1 help? do gains scale? where do they saturate?)
- The odd values [1, 3, 5, 7] provide 4 well-spaced points on the K-performance curve

### Cost analysis

Per iteration, each K candidate requires:
- 1 call to `propose_new_texts()` (1 reflection LLM call per component being updated)
- 1 minibatch evaluation (3 examples at default minibatch_size=3)

At K=7, this is 7 reflection calls + 21 minibatch evaluations per iteration vs 1+3 for vanilla GEPA. The rollout budget (max_metric_calls) counts minibatch evaluations, so K=7 will exhaust the budget ~2.3x faster (7 new-candidate evaluations vs 1, plus the 1 parent evaluation that is shared). This means K=7 runs fewer total iterations but each iteration explores more candidates.

### Fixed parameters (not swept)

All other parameters match paper defaults exactly:
- `candidate_selection_strategy = "pareto"`
- `frontier_type = "instance"`
- `module_selector = "round_robin"`
- `use_merge = True`
- `max_merge_invocations = 5`
- `reflection_minibatch_size = 3`
- `temperature = 0.6, top_p = 0.95, top_k = 20`

---

## 5. Hypotheses and Success Criteria

### Primary Hypothesis (H1)

**Statement:** Generating K>1 independent mutations per iteration and selecting the best improves GEPA's test-set performance on AIME-2025 relative to K=1 (vanilla GEPA).

**Metric:** AIME-2025 accuracy (exact match, 0-1 scoring per problem)

**Success criterion:** K=3 mean test accuracy > K=1 mean test accuracy, with effect size d >= 0.8 (large effect) across 3 seeds. Specifically, if the v2/v3 signal (+0.300) replicates at even half its magnitude (+0.150), that is still a large effect detectable at 3 seeds.

**Falsification:** If K=3 mean <= K=1 mean across 3 seeds, H1 is rejected. This would indicate the v2/v3 result was an artifact of the smaller budget or different model, and would warrant investigation before proceeding to failure_stratified_k.

### Secondary Hypothesis (H2)

**Statement:** The best_of_k gain generalizes beyond AIME-2025 to at least one other benchmark (HotpotQA, IFBench, HoVer, PUPA, or LiveBench-Math).

**Metric:** Per-benchmark test accuracy

**Success criterion:** On at least one non-AIME benchmark, K=3 mean test accuracy > K=1 mean test accuracy by >= 2 percentage points.

**Falsification:** If K=3 <= K=1 on all 5 non-AIME benchmarks, the mutation is AIME-specific and its generalizability claim is rejected. The mutation may still be valuable for AIME alone, but this narrows its scope.

### Scaling Hypothesis (H3)

**Statement:** Performance scales with K, with diminishing returns.

**Expected pattern:** K=1 < K=3 < K=5, with K=7 ~ K=5 (plateau).

**Success criterion:** Monotonic increase from K=1 to K=5, with K=7 - K=5 < K=5 - K=3 (concave scaling curve).

**Falsification:** If K=5 < K=3 (non-monotonic), the mechanism may be introducing noise at higher K values, possibly due to the reduced number of total iterations under the same budget.

### Per-Benchmark Specific Criteria

| Benchmark | Paper Baseline (K=1 target) | K=3 Success Threshold | Rationale |
|-----------|---------------------------|----------------------|-----------|
| AIME-2025 | 32.00% | >= 38.00% (+6pp) | Conservative: half the v2/v3 signal of +30pp on a smaller budget |
| HotpotQA  | 62.33% | >= 64.00% (+1.67pp) | Smallest expected gain; already high baseline |
| IFBench   | 38.61% | >= 40.00% (+1.39pp) | Instruction-following may benefit from diverse proposals |
| HoVer     | 52.33% | >= 54.00% (+1.67pp) | Binary classification; diverse prompts may explore verification strategies |
| PUPA      | 91.85% | >= 92.50% (+0.65pp) | Near ceiling; small absolute gain expected |
| LiveBench | 51.95% | >= 53.50% (+1.55pp) | Math reasoning; similar mechanism to AIME |

### Diagnostic Metrics (not success criteria, but tracked)

1. **Deduplication rate:** What fraction of K candidates are duplicates? If > 50%, the reflection LM has low diversity and K>3 is wasteful.
2. **Win position distribution:** Does K=1 (first candidate) win disproportionately, or is the winner uniformly distributed? Non-uniform distribution suggests the LLM's temperature/sampling creates systematic biases.
3. **Budget utilization:** At K=7, how many iterations complete before budget exhaustion vs K=1? If K=7 completes < 30% of K=1's iterations, the iteration reduction may negate the per-iteration gain.

---

## 6. Test Strategy

### Unit Tests (`test_proposer.py`)

1. **K=1 identity test:** `BestOfKProposer` with K=1 produces identical behavior to vanilla `ReflectiveMutationProposer`. Mock the adapter and reflection LM; verify the same candidate is returned.

2. **K=3 best-selection test:** Mock `propose_new_texts()` to return 3 deterministic candidates with known scores [0.2, 0.8, 0.5]. Verify the K=2 candidate (score 0.8) is returned.

3. **Deduplication test:** Mock `propose_new_texts()` to return 2 identical candidates and 1 unique. Verify only 2 evaluations occur (not 3) and the unique candidate's score determines the winner if it is higher.

4. **Callback emission test:** Verify that K evaluation events are emitted (one per unique candidate), not just 1. This validates the Bug #1 fix.

5. **Empty proposal test:** If `propose_new_texts()` raises an exception for one K, the remaining K-1 candidates should still be evaluated. The proposer should not abort on a single failure.

6. **All-duplicate test:** If all K candidates are identical (after deduplication), only 1 evaluation occurs and that candidate is returned. No wasted compute.

7. **Metadata test:** Verify `CandidateProposal.metadata` contains `{"mutation_candidates": K, "unique_candidates": N, "winning_k_index": i, "all_k_scores": [...]}`.

### Integration Tests (`test_integration.py`)

1. **Smoke test with mock adapter:** Run `BestOfKProposer` with K=3 through a full `GEPAEngine` loop (2 iterations) using a mock adapter that returns deterministic scores. Verify the engine completes without errors and produces a `GEPAResult`.

2. **Budget accounting test:** Run K=3 for a fixed budget (e.g., 50 metric calls). Verify `state.total_num_evals` accounts for all K evaluations, not just the winning one. This is critical: if evaluations are undercounted, the budget stopper will allow more iterations than intended.

3. **Config round-trip test:** Create a `MutationConfig` with `mutation_candidates=5`, serialize via `config_snapshot()`, verify the snapshot includes the `mutation_candidates` field.

4. **K=1 regression test:** Run the full pipeline with K=1 and verify the result matches a vanilla GEPA run (same seed, same budget, same adapter). This guards against the best_of_k infrastructure introducing unintended behavioral changes.

### Smoke Tests (pre-run validation)

1. **API connectivity test:** Verify OpenRouter API responds with a valid completion for the reflection LM (Qwen3-8B).

2. **Benchmark loading test:** Load each target benchmark, verify train/val/test splits match expected sizes.

3. **Budget estimation test:** For each (benchmark, K) pair, estimate the number of iterations that will complete within the paper budget. Print a table so we can visually verify no configuration will exhaust budget in < 5 iterations (which would be too few for meaningful optimization).

4. **Dry run test:** Run `run_mutation()` with `dry_run=True` for each config to validate all parameters parse correctly.

---

## 7. Known Bugs to Fix

### Bug #1: Callback Undercounting

**Problem:** When K>1, K evaluations of new candidates fire within a single `propose()` call, but the current code only emits `on_evaluation_start` / `on_evaluation_end` once (for the final winning candidate). The `MetricsCallback` therefore undercounts evaluations, and any downstream analysis of evaluation frequency is incorrect.

**Fix:** In `BestOfKProposer.propose()`, emit evaluation callbacks for each K candidate:

```python
for k in range(K):
    new_texts = self.propose_new_texts(...)
    new_candidate = build_candidate(...)

    # Emit start event for this K candidate
    notify_callbacks(self.callbacks, "on_evaluation_start", EvaluationStartEvent(
        iteration=i,
        candidate_idx=None,
        batch_size=len(minibatch),
        capture_traces=False,
        parent_ids=[curr_prog_id],
        inputs=minibatch,
        is_seed_candidate=False,
    ))

    outputs, scores, objective, actual_evals = state.cached_evaluate_full(...)
    state.increment_evals(actual_evals)

    # Emit end event for this K candidate
    notify_callbacks(self.callbacks, "on_evaluation_end", EvaluationEndEvent(
        iteration=i,
        candidate_idx=None,
        scores=scores,
        has_trajectories=False,
        parent_ids=[curr_prog_id],
        outputs=outputs,
        ...
    ))

    track_best(k, scores)
```

**Validation:** Unit test #4 above. Additionally, after a full run, verify `sum(metric_calls_delta)` across all iterations matches `state.total_num_evals`.

### Bug #2: No Deduplication

**Problem:** When K>1, the reflection LLM may propose identical mutations across K candidates (especially for simple prompts or low-temperature settings). Evaluating duplicate candidates wastes minibatch evaluations (which count against the budget) without exploring new points in prompt space.

**Fix:** Hash-compare proposed texts before evaluation:

```python
seen_hashes: set[int] = set()

for k in range(K):
    new_texts = self.propose_new_texts(...)

    # Deterministic hash of the proposed text content
    text_hash = hash(tuple(sorted(new_texts.items())))
    if text_hash in seen_hashes:
        # Log deduplication event for diagnostics
        self.logger.log(f"Iteration {i}: K={k} is duplicate, skipping evaluation")
        dedup_count += 1
        continue
    seen_hashes.add(text_hash)

    # ... evaluate ...
```

**Validation:** Unit test #3 above. Additionally, track `dedup_count` per iteration in the `BestOfKMetricsCallback` to measure deduplication rates across benchmarks and K values.

**Edge case:** If ALL K candidates are duplicates of each other (complete deduplication), return the single unique candidate. If the single unique candidate also equals the parent, this is equivalent to a no-op iteration (no improvement). The proposer should still return a `CandidateProposal` (not `None`) so the engine can evaluate it against the full val set.

---

## 8. Benchmark Selection

### Tier 1: Run First (highest priority)

| Benchmark | Rationale |
|-----------|-----------|
| **AIME-2025** | Prior validated signal (K=3: 0.400 vs 0.100). Primary confirmation target. Must run first to validate/invalidate H1. |
| **HotpotQA** | Largest training set, most stable signal. If best_of_k helps on AIME but not HotpotQA, the mutation may be specific to math reasoning. |

### Tier 2: Run Second (generalizability)

| Benchmark | Rationale |
|-----------|-----------|
| **HoVer** | Binary classification (SUPPORTED/NOT_SUPPORTED). Tests whether diverse prompt proposals help in a structurally different task. Moderate budget (2426 rollouts) keeps cost low. |
| **LiveBench-Math** | Math reasoning like AIME but different problem distribution. Tests within-domain generalizability. Lowest budget (1839 rollouts) = cheapest to run. |

### Tier 3: Run if Tier 1-2 show signal

| Benchmark | Rationale |
|-----------|-----------|
| **IFBench** | Instruction-following. Orthogonal task type. |
| **PUPA** | Near-ceiling baseline (91.85%). Expected gain is small. Run last to see if best_of_k can push past the ceiling. |

### Execution order within each tier

Within a tier, run all benchmarks for a single K value before moving to the next K value. This allows early stopping: if K=3 shows no gain on Tier 1 benchmarks, we can skip K=5 and K=7 to save compute.

```
Tier 1:
  AIME K=1 (3 seeds) -> AIME K=3 (3 seeds) -> HotpotQA K=1 (3 seeds) -> HotpotQA K=3 (3 seeds)
  [CHECKPOINT: evaluate H1, H2. Go/no-go for K=5, K=7]
  AIME K=5 (3 seeds) -> AIME K=7 (3 seeds) -> HotpotQA K=5 (3 seeds) -> HotpotQA K=7 (3 seeds)

Tier 2:
  HoVer K=1,3 (3 seeds each) -> LiveBench K=1,3 (3 seeds each)

Tier 3 (conditional):
  IFBench K=1,3 (3 seeds each) -> PUPA K=1,3 (3 seeds each)
```

---

## 9. Seed Strategy

### Number of seeds: 3 per condition

**Justification:** The prior signal (d > 1.0) is detectable at 3 seeds with ~80% power. Running 5 seeds would increase power to ~95% but doubles compute cost. Since we have 4 K values x 6 benchmarks = 24 conditions (worst case), 3 seeds keeps the total at 72 runs (vs 120 for 5 seeds). If the signal is weaker than expected (d ~ 0.5), 3 seeds will be insufficient, but we accept this risk: a d=0.5 effect is too small to be practically useful given the compute cost of K>1.

### Which seeds: [42, 123, 456]

These are the first 3 from `DEFAULT_SEEDS = [42, 123, 456, 789, 1024]` defined in `src/gepa_mutations/runner/experiment.py`.

| Seed | Role |
|------|------|
| 42 | Primary seed. Used in all preliminary testing. Results are directly comparable to Phase 1 reproduction runs. |
| 123 | Secondary seed. Provides first replication data point. |
| 456 | Tertiary seed. Provides the minimum required for computing a meaningful standard deviation. |

### Seed usage

- The seed controls: GEPA's `random.Random(seed)` for candidate selection, batch sampling, and merge partner selection
- The seed does NOT control: LLM sampling (controlled by temperature/top_p/top_k). This is intentional -- we want to measure variance from both the search procedure and the LLM's stochasticity.
- Data loading always uses `seed=0` (matching `load_benchmark(benchmark, seed=0)` convention) so train/val/test splits are identical across seeds.

### Expansion protocol

If 3-seed results show a promising but ambiguous signal (0.3 < p < 0.1), extend to seeds [789, 1024] for a total of 5. This keeps the initial compute commitment low while allowing post-hoc power increase.

---

## 10. Rollout Budget

### Per-benchmark budgets

Use the exact paper budgets from `PAPER_ROLLOUTS["gepa"]` in `config.py`. The best_of_k mutation must operate within the same total evaluation budget as vanilla GEPA to ensure a fair comparison.

| Benchmark | Paper Budget (`max_metric_calls`) | Est. Iterations at K=1 | Est. Iterations at K=3 | Est. Iterations at K=7 |
|-----------|----------------------------------|------------------------|------------------------|------------------------|
| AIME-2025 | 7,051 | ~881 | ~503 | ~352 |
| HotpotQA  | 6,871 | ~858 | ~490 | ~343 |
| IFBench   | 3,593 | ~449 | ~256 | ~179 |
| HoVer     | 2,426 | ~303 | ~173 | ~121 |
| PUPA      | 3,936 | ~492 | ~281 | ~197 |
| LiveBench | 1,839 | ~229 | ~131 | ~91 |

**Iteration estimates:** Each iteration uses `minibatch_size` evaluations for the parent (3) plus `minibatch_size * K` for the K new candidates (3K). Merge proposals add additional evaluations when triggered. The formula is approximately:

```
iterations ~ budget / (minibatch_size * (1 + K) + merge_overhead)
           ~ budget / (3 * (1 + K) + ~2)  # merge_overhead ~ 2 avg
```

For K=7: `iterations ~ budget / (3 * 8 + 2) = budget / 26`

### Budget fairness principle

The comparison between K values must be fair on total compute (metric calls), not on number of iterations. K=7 runs fewer iterations but explores more candidates per iteration. The budget is the equalizer.

This means:
- K=1 and K=7 both use exactly `max_metric_calls` evaluations
- K=1 completes ~2.5x more iterations
- K=7 evaluates ~7x more candidate prompts per iteration
- The question is whether quality-per-candidate (K=7) beats quantity-of-iterations (K=1)

### Budget accounting correctness

Every minibatch evaluation of every K candidate must increment `state.total_num_evals`. The `MaxMetricCallsStopper` checks `state.total_num_evals` against `max_metric_calls`, so undercounting would allow K>1 runs to exceed the intended budget. This is related to Bug #1 (callback undercounting) but the budget accounting is a separate code path (`state.increment_evals(actual_evals_count)` in the proposer).

**Verification:** After each run, log `state.total_num_evals` and verify it is within `[max_metric_calls - minibatch_size, max_metric_calls]` (the budget can be slightly under if the last iteration was interrupted by the stopper).

### No budget scaling for K>1

We explicitly do NOT scale the budget by K. The research question is: "given a fixed compute budget, does spending it on K candidates per iteration beat spending it on more iterations?" Scaling the budget would answer a different question ("does more compute help?") which is trivially yes.

---

## Appendix: Total Compute Estimate

### Tier 1 runs (AIME + HotpotQA, all K values, 3 seeds)

```
AIME:     4 K-values x 3 seeds x 7,051 budget  = 84,612 metric calls
HotpotQA: 4 K-values x 3 seeds x 6,871 budget  = 82,452 metric calls
Total Tier 1: 167,064 metric calls
```

### Tier 2 runs (HoVer + LiveBench, K=1 and K=3 only, 3 seeds)

```
HoVer:    2 K-values x 3 seeds x 2,426 budget  = 14,556 metric calls
LiveBench: 2 K-values x 3 seeds x 1,839 budget = 11,034 metric calls
Total Tier 2: 25,590 metric calls
```

### Grand total (Tier 1 + 2): ~192,654 metric calls

Each metric call = 1 LLM inference (task model). At OpenRouter Qwen3-8B pricing, this is the primary cost driver. Reflection LM calls (for `propose_new_texts()`) are additional but scale with iterations, not metric calls, so they are relatively cheaper at high K.
