# Mutation Plan: failure_stratified_k

**Mutation:** Failure-Stratified K -- partition failing examples across K mutation candidates so each candidate specializes in fixing a different failure subset.

**Date:** 2026-03-19
**Dependency:** Requires best_of_k infrastructure (K-loop in `ReflectiveMutationProposer.propose()`)
**Comparison baseline:** best_of_k with the same K value (NOT vanilla GEPA)

---

## 1. Implementation Approach

### Relationship to best_of_k

failure_stratified_k is a strict extension of best_of_k. It reuses the K-loop infrastructure from best_of_k and modifies only the reflective dataset that each candidate receives. The implementation sequence is:

1. best_of_k adds the K-loop to `ReflectiveMutationProposer.propose()`, calling `propose_new_texts()` K times and selecting the best result.
2. failure_stratified_k adds a partitioning step before the K-loop that splits failing examples across K candidates.

Without best_of_k's K-loop, failure_stratified_k has no mechanism to operate.

### Step-by-step code changes

**Step 1: Add `_partition_reflective_dataset()` method to the patched proposer**

Create a method on the patched `ReflectiveMutationProposer` (or a wrapper/subclass) that:
- Accepts the full reflective dataset `Mapping[str, Sequence[Mapping[str, Any]]]` and the minibatch scores `list[float]`.
- Identifies failing examples: those with `score < perfect_score` (default: `score < 1.0`).
- If `len(failing_examples) < K`, returns `None` to signal fallback to standard best_of_k.
- Otherwise, partitions failing examples into K groups using round-robin assignment (sorted by score ascending so the worst failures are distributed evenly, not clustered).
- Returns `list[Mapping[str, Sequence[Mapping[str, Any]]]]` -- K reflective datasets, one per candidate.

**Step 2: Modify the K-loop to use partitioned datasets**

Inside the best_of_k K-loop (which iterates `for k in range(K)`):
- Before calling `propose_new_texts(candidate, reflective_dataset, components)`:
  - If `use_failure_stratified_k` is True and partitioned datasets are available, replace `reflective_dataset` with `partitioned_datasets[k]`.
  - If fallback was triggered (failing < K), use the full reflective dataset for all K (identical to standard best_of_k).

**Step 3: Add config fields and threading**

- Add `use_failure_stratified_k: bool = False` to `MutationConfig`.
- Thread this flag through `run_mutation()` to the patched proposer.
- Validate at config time: if `use_failure_stratified_k=True` and `mutation_candidates <= 1`, raise `ValueError`.

**Step 4: Add diagnostic logging**

- Log whether stratification was applied or fell back to standard K-sampling.
- Log the partition sizes (e.g., "Stratified K=3: partitions [2, 1, 1] from 4 failing examples").
- Log per-candidate scores after the K-loop to enable post-hoc analysis of whether stratification produced more diverse candidates.

**Step 5: Extend MetricsCallback**

- Add a `stratification_applied: bool` field to `IterationMetrics`.
- Add `partition_sizes: list[int] | None` to capture the partition distribution.
- Add `per_k_scores: list[float] | None` to capture all K candidates' scores (not just the winner).

### Implementation pattern: monkey-patch vs subclass

Use the same patching approach as best_of_k. The failure_stratified_k runner will:
1. Import the best_of_k patched proposer (which has the K-loop).
2. Further patch or subclass it to add the partitioning logic.
3. This ensures failure_stratified_k inherits all best_of_k bugfixes (callback undercounting, deduplication).

The layered patching is cleaner than a single combined patch because it preserves the ability to run best_of_k independently as a control condition.

---

## 2. GEPA Components to Modify

### Patches (do NOT modify files under `gepa/`)

| Component | File | Change Type | Description |
|-----------|------|-------------|-------------|
| `ReflectiveMutationProposer.propose()` | `gepa/src/gepa/proposer/reflective_mutation/reflective_mutation.py` | Monkey-patch (via best_of_k) | K-loop wrapping `propose_new_texts()` -- inherited from best_of_k |
| `ReflectiveMutationProposer.propose()` | Same | Additional monkey-patch | Partition injection before each `propose_new_texts()` call in the K-loop |

### Clean extensions (our code)

| Component | File | Description |
|-----------|------|-------------|
| `_partition_reflective_dataset()` | `src/gepa_mutations/experiments/failure_stratified_k/proposer.py` | Core partitioning logic |
| `FailureStratifiedConfig` | `src/gepa_mutations/experiments/failure_stratified_k/config.py` | Extends MutationConfig with stratification fields |
| `runner.py` | `src/gepa_mutations/experiments/failure_stratified_k/runner.py` | Experiment runner constructing config + calling `run_mutation()` |
| `test_stratification.py` | `tests/test_failure_stratified_k.py` | Unit and integration tests |

### Files NOT modified

- `gepa/src/gepa/api.py` -- no new `optimize()` parameters needed; the mutation operates inside the proposer
- `gepa/src/gepa/core/engine.py` -- no changes to acceptance logic or main loop
- `gepa/src/gepa/strategies/instruction_proposal.py` -- the partitioned dataset has the same schema as the full dataset; `InstructionProposalSignature` operates unchanged
- `src/gepa_mutations/base.py` -- `MutationConfig` and `run_mutation()` remain generic; mutation-specific config is composed in the experiment's `runner.py`

---

## 3. New MutationConfig Fields

### Fields to add to `MutationConfig` in `src/gepa_mutations/base.py`

```python
# --- failure_stratified_k ---
use_failure_stratified_k: bool = False
```

### Fields inherited from best_of_k (already present after best_of_k implementation)

```python
# --- best_of_k (prerequisite) ---
mutation_candidates: int = 1
```

### Validation rules

1. If `use_failure_stratified_k=True`, then `mutation_candidates` must be > 1. Enforced at config construction time in the experiment's `runner.py` (not in MutationConfig itself, to keep MutationConfig generic).
2. If `use_failure_stratified_k=True` and `mutation_candidates=1`, raise:
   ```
   ValueError("failure_stratified_k requires mutation_candidates > 1. "
              "Set mutation_candidates to 3, 5, or 7.")
   ```
3. Default `use_failure_stratified_k=False` ensures MutationConfig remains backward-compatible: all existing experiments behave identically.

### Type details

| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `use_failure_stratified_k` | `bool` | `False` | If True, requires `mutation_candidates > 1` |
| `mutation_candidates` | `int` | `1` | Must be >= 1; from best_of_k |

---

## 4. Parameter Sweep Values

### Primary sweep: stratification vs standard K-sampling

| Parameter | Values | Justification |
|-----------|--------|---------------|
| `use_failure_stratified_k` | `[True, False]` | The core A/B comparison. `False` = standard best_of_k (control). `True` = stratified (treatment). |
| `mutation_candidates` | `[3, 5, 7]` | Same K values as best_of_k sweep. K=3 is the minimum for meaningful stratification with minibatch_size=3. K=5 and K=7 test whether more candidates with finer partitions help or hurt. |

### Fixed parameters (paper defaults)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `reflection_minibatch_size` | `3` | Paper default. With minibatch=3, the maximum number of failing examples is 3, so K > 3 will always trigger fallback for some iterations. This is intentional -- we want to measure the fraction of iterations where stratification activates. |
| `candidate_selection_strategy` | `"pareto"` | Paper default |
| `module_selector` | `"round_robin"` | Paper default |
| `use_merge` | `True` | Paper default |
| `max_merge_invocations` | `5` | Paper default |

### Sweep matrix

The full sweep is 3 (K values) x 2 (stratification on/off) = 6 conditions. The `False` conditions are shared with best_of_k's sweep (same runs, reusable results):

| Condition | K | Stratified | Notes |
|-----------|---|------------|-------|
| bok_k3 | 3 | False | Reuse from best_of_k sweep |
| bok_k5 | 5 | False | Reuse from best_of_k sweep |
| bok_k7 | 7 | False | Reuse from best_of_k sweep |
| fsk_k3 | 3 | True | New: failure_stratified_k with K=3 |
| fsk_k5 | 5 | True | New: failure_stratified_k with K=5 |
| fsk_k7 | 7 | True | New: failure_stratified_k with K=7 |

### Why these K values

- **K=3:** Matches `reflection_minibatch_size=3`. With 3 examples in the minibatch, at most 3 can fail, so K=3 is the sweet spot where each candidate can receive exactly 1 failing example. This is the cleanest test of the hypothesis.
- **K=5:** Tests behavior when K > max_failing_examples. Some candidates will receive 0 failing examples and get the full dataset (fallback). This tests whether a mix of specialized and general candidates is beneficial.
- **K=7:** Further explores the K > failures regime. Most iterations will have K > failing_count, so this tests the fallback path heavily.

### Budget consideration

Each K candidate requires one `propose_new_texts()` call (LM reflection) and one minibatch evaluation. At K=3, this is 3x the per-iteration cost of vanilla GEPA. The paper rollout budgets are sufficient because the extra cost is in reflection tokens (not counted in rollout budget) and minibatch evaluations (3 examples per candidate = 9 total evaluations per iteration vs 3 for vanilla). The budget impact is identical to best_of_k at the same K.

---

## 5. Hypotheses and Success Criteria

All hypotheses compare failure_stratified_k against best_of_k at the **same K value** (paired comparison), not against vanilla GEPA.

### Primary hypothesis (H1): Stratification increases candidate diversity

**Statement:** Failure-stratified K=3 will produce accepted candidates with lower pairwise overlap in their per-example success vectors compared to standard best_of_k K=3, measured as the mean Jaccard similarity of success patterns across accepted candidates on the validation set.

**Metric:** Mean pairwise Jaccard similarity of binary success vectors (score >= 0.5) across accepted candidates.

**Success criterion:** Mean Jaccard similarity is at least 0.05 lower (more diverse) for failure_stratified_k vs best_of_k at the same K, averaged across seeds.

**Falsifiability:** If mean Jaccard similarity is equal or higher for failure_stratified_k, the hypothesis is falsified -- stratification did not increase diversity.

### Secondary hypothesis (H2): Stratification improves aggregate accuracy on heterogeneous-failure benchmarks

**Statement:** Failure-stratified K=3 will achieve higher test accuracy than best_of_k K=3 on benchmarks with heterogeneous failure modes (HotpotQA, HoVer, IFBench), but not on benchmarks with homogeneous failures (AIME).

**Per-benchmark success criteria (measured against best_of_k K=3 baseline):**

| Benchmark | Expected direction | Minimum detectable effect | Rationale |
|-----------|-------------------|---------------------------|-----------|
| HotpotQA | fsk >= bok | +1.0 pp | Multi-hop QA has diverse failure modes (retrieval, reasoning, aggregation) |
| HoVer | fsk >= bok | +1.0 pp | Claim verification has heterogeneous failures (evidence matching, label inference) |
| IFBench | fsk >= bok | +0.5 pp | Constraint satisfaction has distinct failure categories per constraint type |
| AIME | fsk ~ bok | 0 pp (no difference expected) | Math reasoning failures are homogeneous (wrong computation, wrong approach) |
| PUPA | fsk >= bok | +0.5 pp | Mixed task types may have heterogeneous failures |
| LiveBench | fsk ~ bok | 0 pp (no difference expected) | Math benchmark, similar to AIME |

**Falsifiability:** If failure_stratified_k shows no improvement (or regression) on HotpotQA and HoVer compared to best_of_k at the same K, the hypothesis is falsified.

### Tertiary hypothesis (H3): Stratification activates more often on heterogeneous benchmarks

**Statement:** The fraction of iterations where stratification actually activates (failing_examples >= K) will be higher on heterogeneous benchmarks (HotpotQA, HoVer) than homogeneous benchmarks (AIME, LiveBench).

**Metric:** `stratification_activation_rate = iterations_with_stratification / total_iterations`

**Success criterion:** Activation rate is > 0.3 on HotpotQA and HoVer. Activation rate is < 0.1 on AIME.

**Diagnostic value:** If activation rate is very low across all benchmarks (< 0.1), the mutation is not getting enough failing examples to stratify, and the minibatch_size may need to be increased (future experiment). This is itself a valuable finding.

### Null result interpretation

A null result (no difference between stratification and standard K-sampling) would indicate one of:
1. The reflection LM is not sensitive to which failing examples it sees -- it abstracts the same lessons regardless of the specific failures shown.
2. The minibatch_size=3 provides too few failing examples for meaningful partitioning.
3. Failure modes on these benchmarks are not as heterogeneous as hypothesized.

Each of these is a publishable finding about GEPA's failure landscape.

---

## 6. Test Strategy

### Unit tests (`tests/test_failure_stratified_k.py`)

**T1: Partition correctness**
- Input: reflective dataset with N failing examples, K partitions requested.
- Assert: each example appears in exactly one partition; union of partitions equals the full failing set; partitions are balanced (sizes differ by at most 1).
- Edge cases: N < K (returns None), N = K (each partition has exactly 1), N = 0 (returns None), N = 1 (returns None).

**T2: Score-based failure identification**
- Input: minibatch scores [1.0, 0.5, 0.0], perfect_score=1.0.
- Assert: examples at indices 1 and 2 are identified as failing; example at index 0 is not.

**T3: Round-robin assignment order**
- Input: 5 failing examples sorted by score ascending, K=3.
- Assert: partition 0 gets examples [0, 3], partition 1 gets [1, 4], partition 2 gets [2]. (Worst failures distributed evenly.)

**T4: Fallback to standard K-sampling**
- Input: reflective dataset with 2 failing examples, K=3.
- Assert: `_partition_reflective_dataset()` returns None; caller uses full dataset for all K.

**T5: Config validation**
- Assert: `use_failure_stratified_k=True` with `mutation_candidates=1` raises ValueError.
- Assert: `use_failure_stratified_k=False` with `mutation_candidates=1` does NOT raise.

**T6: Multi-component reflective dataset**
- Input: reflective dataset with keys ["system_prompt", "reasoning_prompt"], 4 failing examples each, K=2.
- Assert: partitioning is applied per-component with the same partition assignment (example i goes to partition i % K for all components).

### Integration tests

**T7: End-to-end with mock adapter**
- Build a mock GEPAAdapter that returns deterministic scores and trajectories.
- Run `optimize()` with the failure_stratified_k patch for 5 iterations, K=3.
- Assert: the patched proposer calls `propose_new_texts()` 3 times per iteration with different (partitioned) reflective datasets when failing_count >= 3.
- Assert: when failing_count < 3, `propose_new_texts()` is called 3 times with the full dataset (fallback).

**T8: Callback emission correctness (inherited from best_of_k)**
- Assert: `on_evaluation_end` is emitted K times per iteration (or 1 time with K sub-results, depending on best_of_k's fix).
- Assert: `stratification_applied` field is correctly set in metrics.

### Smoke tests

**T9: Dry-run smoke test**
- Run `runner.py --benchmark hotpotqa --seed 42 --subset 5 --max-metric-calls 20 --mutation-candidates 3 --use-failure-stratified-k`.
- Assert: completes without error.
- Assert: results JSON contains `stratification_applied` diagnostic field.
- Assert: rollout count is within expected range (10-20 for budget=20 with K=3).

**T10: Determinism test**
- Run the same smoke test twice with identical seeds.
- Assert: partition assignments are identical across runs.
- Assert: final scores are identical (given deterministic mock or cached LM).

---

## 7. Known Bugs to Fix

### Bug 1: Degenerate with fewer failing examples than K

**Symptom:** If only 1-2 examples fail in the minibatch (common when the candidate is already good), all K candidates receive the same dataset, making stratification identical to standard best_of_k. No error is raised; the mutation silently becomes a no-op.

**Root cause:** The partitioning function does not check whether `len(failing_examples) >= K` before partitioning.

**Fix:** In `_partition_reflective_dataset()`:
```python
if len(failing_examples) < K:
    return None  # Signal fallback to caller
```
The caller (K-loop) checks the return value:
```python
partitions = _partition_reflective_dataset(reflective_dataset, scores, K, perfect_score)
for k in range(K):
    dataset_for_k = partitions[k] if partitions is not None else reflective_dataset
    new_texts = propose_new_texts(candidate, dataset_for_k, components)
    ...
```

**Log message:** `"Stratification fallback: {len(failing)} failing examples < K={K}. Using full dataset for all candidates."`

### Bug 2: Silent no-op at K=1

**Symptom:** If `mutation_candidates=1`, the stratification code path produces a single partition identical to the full dataset. No error, but wasted CPU cycles and misleading diagnostics.

**Root cause:** No validation that K > 1 when `use_failure_stratified_k=True`.

**Fix:** Validate at config construction time:
```python
if config.use_failure_stratified_k and config.mutation_candidates <= 1:
    raise ValueError(
        "failure_stratified_k requires mutation_candidates > 1. "
        "Set mutation_candidates to 3, 5, or 7."
    )
```

### Bug 3: Inherited from best_of_k -- callback undercounting

**Symptom:** In the best_of_k K-loop, K evaluations are performed on the minibatch but only 1 `on_evaluation_end` callback is emitted (for the winning candidate). The MetricsCallback undercounts total evaluations.

**Root cause:** The K-loop in best_of_k calls `adapter.evaluate()` K times but only the winning candidate's results are propagated through the callback system.

**Fix (inherited from best_of_k):** Emit `on_evaluation_end` for each of the K candidates, or emit a single `on_k_candidates_evaluated` event with all K scores. The callback handler in `MetricsCallback` must aggregate correctly. failure_stratified_k inherits this fix automatically since it reuses best_of_k's K-loop.

### Bug 4: Inherited from best_of_k -- no deduplication of identical mutations

**Symptom:** Two of the K candidates may produce identical prompt text, wasting one evaluation call.

**Root cause:** `propose_new_texts()` is called K times with potentially similar reflective datasets (especially in fallback mode), which can produce identical outputs.

**Fix (inherited from best_of_k):** Hash-compare the proposed texts after each K iteration. If a new text matches a previously generated one, skip its evaluation and reuse the prior score. This is less likely to trigger with failure_stratified_k (since each candidate sees different failing examples) but still possible when the reflection LM ignores the specific examples.

### Bug 5: Partition assignment does not account for multi-component alignment

**Symptom:** If the reflective dataset has multiple components (keys), and the failing example indices differ across components, partitioning per-component independently may assign the same example index to different partitions in different components.

**Root cause:** Each component's failing examples are partitioned independently.

**Fix:** Use a single global partition assignment based on the minibatch example indices (not per-component). All components for example `i` go to the same partition `i % K`. The partition function operates on example indices, not on per-component datasets:

```python
# Determine partition assignment from scores (global, not per-component)
failing_indices = [i for i, s in enumerate(scores) if s < perfect_score]
# Sort by score ascending so worst failures are distributed first
failing_indices.sort(key=lambda i: scores[i])
# Round-robin assignment
assignments = {idx: idx_in_sorted % K for idx_in_sorted, idx in enumerate(failing_indices)}

# Apply to each component
for component_name, records in reflective_dataset.items():
    for k in range(K):
        partition_k_records = [records[i] for i in range(len(records)) if assignments.get(i, -1) == k]
        ...
```

---

## 8. Benchmark Selection

### Priority ordering for initial runs

| Priority | Benchmark | Rationale |
|----------|-----------|-----------|
| 1 | **HotpotQA** | Highest expected stratification signal. Multi-hop QA has diverse failure modes: retrieval failures (wrong passage), reasoning failures (wrong inference chain), aggregation failures (wrong answer synthesis). The reflective dataset for each failure type should contain distinct diagnostic information, making partitioning maximally useful. Paper GEPA score: 62.33, leaving 37.67pp of failures to stratify across. |
| 2 | **HoVer** | Second-highest heterogeneity. Claim verification failures split into: evidence insufficiency (not enough supporting facts), label confusion (SUPPORTED vs NOT_SUPPORTED), multi-hop evidence chains. Paper GEPA score: 52.33, leaving 47.67pp of failures. |
| 3 | **IFBench** | Constraint satisfaction has categorically distinct failure modes (format constraints, content constraints, length constraints, etc.). Paper GEPA score: 38.61, with 61.39pp of failures providing rich material for stratification. |
| 4 | **PUPA** | Mixed task types may show heterogeneous failures. Paper GEPA score: 91.85, but only 8.15pp of failures -- very few failing examples to stratify, so the fallback path will dominate. Useful as a diagnostic for activation rate. |
| 5 | **AIME** | Math reasoning failures are expected to be homogeneous (computational errors, wrong approach). This serves as a **negative control** -- stratification should show no benefit. Paper GEPA score: 32.00. |
| 6 | **LiveBench** | Similar to AIME (math domain). Lowest priority because it overlaps with AIME's expected null result. Paper GEPA score: 51.95. |

### Recommended initial run plan

Run HotpotQA and HoVer first (highest expected signal), then IFBench. Use AIME as the negative control. Skip PUPA and LiveBench initially unless the first three show promising results.

---

## 9. Seed Strategy

### Seed count

Use the same seeds as best_of_k for **paired comparison**. This is critical: failure_stratified_k's research question is whether stratification improves over standard K-sampling, so the comparison must be seed-matched.

**Seeds:** `[42, 123, 456]` (3 seeds, matching best_of_k's initial run plan)

### Justification for 3 seeds

From the selection report, best_of_k has d > 1.0 at 3 seeds. failure_stratified_k's incremental signal over best_of_k is expected to be smaller (d = 0.3-0.5), which means 3 seeds are **insufficient for statistical significance** of the stratification increment alone. However:

1. 3 seeds are sufficient to detect whether stratification causes a **regression** (the primary safety check).
2. 3 seeds provide directional evidence that guides whether to invest in more seeds.
3. The paired design (same seeds as best_of_k) reduces variance by controlling for seed-specific effects.

### If directional evidence is positive

Extend to 5 seeds: `[42, 123, 456, 789, 1024]` (matching `DEFAULT_SEEDS` in `experiment.py`). At d = 0.4 and 5 paired seeds, power is approximately 0.50 -- still underpowered for a definitive conclusion, but sufficient for a strong preliminary result.

### If directional evidence is negative or null

Do NOT extend seeds. Report the null result as evidence that failure stratification does not add value over standard K-sampling at our minibatch size. Recommend future experiments with larger minibatch sizes (e.g., 5 or 7) that provide more failing examples to partition.

### Seed matching protocol

For each seed `s` and K value `k`:
- Run `best_of_k(K=k, seed=s)` first (if not already run).
- Run `failure_stratified_k(K=k, seed=s, use_failure_stratified_k=True)` with the same seed.
- Compare results pairwise: `delta[s] = fsk_score[s] - bok_score[s]`.
- Report mean delta, standard deviation of delta, and paired t-test p-value.

---

## 10. Rollout Budget

### Budget per benchmark

Use the paper's GEPA rollout budgets. The K-loop increases per-iteration evaluation cost (K minibatch evaluations instead of 1), but the total rollout budget caps the run regardless. This means failure_stratified_k runs will complete in fewer iterations than vanilla GEPA at the same budget -- identical behavior to best_of_k.

| Benchmark | Paper Budget | Budget per Iteration (K=3) | Approx. Iterations |
|-----------|-------------|---------------------------|---------------------|
| HotpotQA  | 6,871       | 3 (minibatch) + 3*3 (K evals) = 12 | ~572 |
| IFBench   | 3,593       | ~12                       | ~299 |
| HoVer     | 2,426       | ~12                       | ~202 |
| PUPA      | 3,936       | ~12                       | ~328 |
| AIME      | 7,051       | ~12                       | ~587 |
| LiveBench | 1,839       | ~12                       | ~153 |

Note: "Budget per iteration" is approximate. The actual count depends on:
- Whether the candidate is accepted (accepted candidates trigger a full valset evaluation, consuming `|valset|` additional rollouts).
- Whether merge is triggered (consumes additional rollouts).
- Evaluation cache hits (reduce actual evaluations).

### Budget accounting for K evaluations

Each K-loop iteration performs:
1. 1 capture_traces evaluation on minibatch (3 examples) = 3 rollouts
2. K propose_new_texts calls (reflection LM, NOT counted as rollouts)
3. K evaluations of new candidates on minibatch (3 examples each) = 3K rollouts
4. Best candidate (if accepted) evaluated on full valset = |valset| rollouts

Total per accepted iteration: 3 + 3K + |valset|
Total per rejected iteration: 3 + 3K

With K=3: 3 + 9 = 12 rollouts per rejected iteration, 12 + |valset| per accepted iteration.

### Budget fairness

The budget comparison against best_of_k is **fair by construction**: both conditions use the same K value and the same total rollout budget. The only difference is whether the K candidates see partitioned or full reflective datasets. This means any performance difference is attributable to the stratification, not to budget differences.

The budget comparison against vanilla GEPA (K=1) is **NOT the research question** for this mutation. That comparison belongs to best_of_k.

### Smoke test budget

For smoke tests, use `max_metric_calls=50` with `subset=5` to verify the pipeline runs end-to-end without exhausting API credits. This provides approximately 4 iterations at K=3, enough to verify stratification logic activates at least once.

---

## Appendix: Reflective Dataset Structure Reference

The reflective dataset returned by `adapter.make_reflective_dataset()` has the structure:

```python
reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]]
#                    ^component   ^examples    ^per-example record
```

Each per-example record typically contains:
```python
{
    "Inputs": {"question": "...", "context": "..."},
    "Generated Outputs": "...",
    "Feedback": "Score: 0.0. Expected: X, Got: Y."
}
```

The partition function must preserve this structure exactly. Each partition is itself a `Mapping[str, Sequence[Mapping[str, Any]]]` with the same component keys but a subset of the example records.

### Identifying failing examples

The reflective dataset does **not** directly contain numeric scores. Scores come from the `EvaluationBatch.scores` list returned by `adapter.evaluate()`. The correspondence is positional: `reflective_dataset[component][i]` corresponds to `scores[i]` (i.e., the i-th example in the minibatch).

The partition function must receive both the reflective dataset and the scores to identify which examples failed:

```python
def _partition_reflective_dataset(
    reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    scores: list[float],
    K: int,
    perfect_score: float = 1.0,
) -> list[Mapping[str, Sequence[Mapping[str, Any]]]] | None:
    ...
```

### Partition construction

For each partition k, construct a new reflective dataset containing:
- **Failing examples assigned to partition k** (the specialized subset)
- **All passing examples** (shared across all partitions -- the reflection LM needs context about what success looks like)

This means each partition has `len(passing) + len(failing_in_partition_k)` examples, not just `len(failing_in_partition_k)`. The passing examples provide positive context; the failing examples provide the improvement target specific to this candidate.

This design choice is important: if we only showed failing examples, the reflection LM would lack context about what the current prompt does well, potentially proposing changes that regress on passing cases.
