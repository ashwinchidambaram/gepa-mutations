# Final Mutation Selection Report

**Date:** 2026-03-19
**Process:** Cross-team overlap-priority tiebreaking across Team A and Team B consensus rankings

---

## Final Selection (3 Mutations)

| Priority | Mutation | Selection Basis |
|----------|----------|-----------------|
| 1 | **best_of_k** | Tiebreak winner (3rd slot) |
| 2 | **contrastive_reflection** | Auto-included (both teams' top 3) |
| 3 | **failure_stratified_k** | Auto-included (both teams' top 3) |

---

## Auto-Included Mutations

### 1. contrastive_reflection

**Team A rank:** #2 | **Team B rank:** #1 | **Selection:** Auto-include (overlap)

**Rationale:** Both teams independently ranked this in their top 3, making it the strongest consensus pick. It targets the information bottleneck in GEPA's self-reflection mechanism, transforming reflection from open-ended generation into a guided comparative process. This is the only mutation that introduces a genuinely novel algorithmic mechanism not explored in the paper's Section 5 ablations.

- **Expected signal:** +3-5pp on HotpotQA (Team B estimate)
- **Effect size:** d=0.67, requiring ~10 seeds for reliable detection
- **Risk profile:** Calculated gamble. The mechanism is theoretically well-motivated but entirely untested. High information value regardless of outcome -- a null result would indicate that reflection quality is not a binding constraint in GEPA, which is itself a valuable finding.

### 2. failure_stratified_k

**Team A rank:** #3 | **Team B rank:** #3 | **Selection:** Auto-include (overlap)

**Rationale:** Both teams ranked this third, recognizing it as the most scientifically ambitious mutation. It bridges diversity maintenance (niching/fitness-sharing from evolutionary computation) with targeted mutation by stratifying candidate generation based on failure modes. This creates a principled mechanism for GEPA to escape local optima by directing search effort toward underperforming problem categories.

- **Expected signal:** Incremental over best_of_k baseline
- **Effect size:** Incremental signal is likely undetectable in isolation at 3-5 seeds
- **Dependency:** Requires best_of_k infrastructure as a prerequisite. The mutation only makes sense when k>1, since stratification operates over the candidate pool.
- **Risk profile:** High ambition, moderate risk. Even if the stratification signal is not statistically separable from best_of_k alone, the mechanism provides diagnostic information about whether GEPA's failures are clustered by problem type.

---

## Contested 3rd Slot: Tiebreak Analysis

### Candidates

| Mutation | Advocated by | Opposed by |
|----------|-------------|------------|
| best_of_k | Team A (#1) | Team B (infrastructure, not research) |
| stratified_batch | Team B (#2) | Team A (vetoed for low detectability) |

### Tiebreaking Rule Applied

**Rule 1: Empirical evidence > Information value > Implementation cleanliness**

This is decisive. best_of_k has empirical evidence; stratified_batch does not.

- **best_of_k:** +0.300 on AIME in prior runs (v2/v3). Effect size d>1.0. Detectable at just 3 seeds. This is the only mutation in the entire portfolio with a validated positive signal.
- **stratified_batch:** Entirely untested. Effect size estimated at d=0.4-0.8, requiring 12-15 seeds for reliable detection. We do not have the seed budget for this.

Under Rule 1, empirical evidence strictly dominates information value for untested mutations. best_of_k wins without needing to invoke Rules 2 or 3.

### Winner: best_of_k

**Full justification:**

1. **Empirical signal is non-negotiable.** The +0.300 AIME gain with d>1.0 is the single strongest piece of evidence in our mutation portfolio. Running best_of_k first answers the foundational question: does GEPA's 1-of-1 prompt generation design leave performance on the table? Every other mutation is speculative until this is confirmed.

2. **Detectable within our seed budget.** At d>1.0, 3 seeds suffice. This is the only mutation where we can expect a statistically meaningful result from our planned experimental runs. Every other mutation with a detectable signal requires 10+ seeds.

3. **Infrastructure prerequisite for failure_stratified_k.** Since failure_stratified_k (auto-included) requires best_of_k as infrastructure, excluding best_of_k as a standalone research question would mean running it anyway without measuring its isolated contribution. That wastes information.

4. **Team B's objection is addressed by sequencing.** Team B argued best_of_k has "low marginal information value since we already know it works." This conflates preliminary signal with confirmed reproduction. The v2/v3 result was observed but not rigorously reproduced. Running best_of_k as the first experiment (before failure_stratified_k layers on top of it) provides the controlled measurement that separates best_of_k's contribution from failure_stratified_k's increment. This is standard experimental methodology.

---

## Excluded Mutations

### stratified_batch (Runner-up for 3rd slot)

**Team B rank:** #2 | **Team A:** Vetoed

**Why excluded:**

- **Fails the detectability threshold.** d=0.4-0.8 requires 12-15 seeds. At our 3-5 seed budget, any observed effect is indistinguishable from noise. Team A's data scientist was correct to veto this.
- **The "high information value even if null" argument is undermined by low power.** A null result at 3-5 seeds does not diagnose whether fitness noise is a bottleneck; it merely tells us we lacked statistical power. This is a non-result, not a null result.
- **Clean experimental design is necessary but not sufficient.** Same budget, same iterations, same tokens -- these are desirable properties, but they don't compensate for an experiment that cannot produce a conclusive answer at our seed budget.
- **Recommendation:** Revisit stratified_batch if the project scales to 12+ seeds per condition. At that budget, its clean design and orthogonality to Section 5 ablations make it the obvious next candidate.

### structured_dpg (#4)

No gain in prior testing. Would need retest with d=0.67 / 10 seeds. Not competitive with untested mutations that have stronger theoretical motivation.

### lesson_memory (#5)

No gain in prior testing. Detectability classified as undetectable. Not viable at any realistic seed budget.

### phenotypic_diversity (#6)

Never activated in prior runs, meaning the mechanism's trigger conditions may be miscalibrated. Needs diagnostic work before it can be considered as a mutation candidate.

### boltzmann_selection (#7)

No effect observed. Classified as diagnostic, not a performance mutation. Undetectable effect size.

---

## Relationship Between best_of_k and failure_stratified_k

These two mutations have a strict dependency relationship:

```
best_of_k (infrastructure + standalone research question)
    |
    v
failure_stratified_k (layered on top of best_of_k)
```

- **best_of_k** changes GEPA from generating 1 candidate per iteration to generating k candidates and selecting the best. This is both a standalone research question (does k>1 help?) and infrastructure required by failure_stratified_k.
- **failure_stratified_k** modifies *how* the k candidates are generated, stratifying them by failure category rather than generating k i.i.d. samples. It cannot be tested without best_of_k in place.

**Why best_of_k must be a research question, not just infrastructure:**

If we skip measuring best_of_k in isolation and go directly to failure_stratified_k, we cannot attribute any observed gains. A positive result for failure_stratified_k could be entirely due to the k>1 mechanism (best_of_k) with the stratification adding nothing. Only by measuring best_of_k first do we establish the baseline against which failure_stratified_k's incremental contribution can be assessed.

---

## Execution Order and Dependencies

### Phase 1: best_of_k (Run First)

- **Priority:** Highest. Non-negotiable first experiment.
- **Dependencies:** None.
- **Seeds:** 3 minimum (sufficient at d>1.0).
- **Go/no-go gate:** If best_of_k shows no positive signal (contradicting v2/v3 results), investigate before proceeding. If confirmed positive, proceed to Phase 2 and Phase 3.
- **Metrics:** Primary: AIME accuracy. Secondary: HotpotQA F1, all other benchmarks.

### Phase 2: contrastive_reflection (Run Second, Independent)

- **Priority:** High. Independent of best_of_k results.
- **Dependencies:** None. Orthogonal mechanism.
- **Seeds:** Ideally 10 for d=0.67 detection. Run with available budget; accept reduced power if necessary.
- **Metrics:** Primary: HotpotQA F1. Secondary: all benchmarks.
- **Note:** Can run in parallel with Phase 1 if compute budget allows, since it is independent.

### Phase 3: failure_stratified_k (Run Third, Conditional)

- **Priority:** Conditional on Phase 1.
- **Dependencies:** Requires best_of_k infrastructure. Only run if Phase 1 shows positive or neutral signal.
- **Seeds:** Match Phase 1 seed count for direct comparison.
- **Metrics:** Same as Phase 1, with additional diagnostic: per-failure-category accuracy breakdown.
- **Analysis:** Compare against best_of_k baseline (Phase 1), not against vanilla GEPA. The research question is whether stratification adds value over uniform sampling of k candidates.

### Parallel Execution Opportunity

Phases 1 and 2 are independent and can run simultaneously:

```
Time -->

  Phase 1: best_of_k ──────────┐
                                ├──> Phase 3: failure_stratified_k
  Phase 2: contrastive_reflection ──────────> (complete independently)
```

---

## Summary

| Slot | Mutation | Basis | Key Justification |
|------|----------|-------|-------------------|
| 1 | best_of_k | Tiebreak (Rule 1: empirical evidence) | Only mutation with validated signal. d>1.0, 3 seeds. Infrastructure for #3. |
| 2 | contrastive_reflection | Auto-include (overlap) | Consensus #1/#2. Novel mechanism. Highest algorithmic merit. |
| 3 | failure_stratified_k | Auto-include (overlap) | Consensus #3/#3. Most ambitious. Bridges diversity + targeted mutation. |
| -- | stratified_batch | Excluded (runner-up) | Undetectable at our seed budget (needs 12-15 seeds). Revisit if budget scales. |
