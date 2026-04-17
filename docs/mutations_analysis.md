# GEPA Mutations: Methods and Preliminary Observations

## 0. How to Read This Document

**Purpose.** This document provides technical descriptions of eleven algorithmic mutations to GEPA, the baseline prompt optimization framework for this project. The primary goal is to explain what each mutation *is*, how it *works algorithmically*, and precisely how it *differs from GEPA*. This is a methods-first document; empirical comparison is secondary and incomplete.

**Data status.** Experimental coverage is partial. Several mutations have fewer than 5 completed runs; no mutation has full coverage across all models and benchmarks. **No statistical conclusions are drawn.** All data observations are illustrative — preceded explicitly by sample size caveats — and should not be interpreted as definitive performance assessments.

**Notation.**
- `Δtest_score` = mutation test score minus GEPA test score on the same (model, benchmark)
- `rollout_count` = number of times the task LM is evaluated on a training example
- `[n=N]` = number of completed runs underlying an observation
- `—` in tables = no data available
- `⚠` = known stability or implementation concern

---

## 1. GEPA: The Baseline Algorithm

### 1.1 Overview

GEPA (Genetic-Pareto Evolutionary Prompt Automation, ICLR 2026) is an LLM-based prompt optimization framework that combines evolutionary population management with natural language reflection. It treats prompt optimization as an iterative improvement problem: rather than searching a discrete space of prompt templates, it asks a language model to reflect on its own failures and propose targeted improvements.

GEPA is the baseline against which all 11 mutations are defined. Every mutation either modifies one component of the GEPA loop (proposer-replacement class) or replaces the loop entirely (standalone search class).

### 1.2 The 4-Step Loop

Each GEPA iteration executes:

1. **Trajectory Sampling.** Sample a minibatch of `B=3` training examples. Evaluate the current candidate prompt on each example using the task LM. Collect (input, output, label, score) tuples.

2. **Natural Language Reflection.** Pass the collected trajectories — particularly the failures — to a reflection LM. The LM diagnoses error patterns and proposes a new candidate prompt that addresses the identified weaknesses. Reflection is unconstrained: the LM may rewrite, restructure, or extend the prompt.

3. **Pareto Frontier Selection.** Evaluate the proposed candidate on the training examples it has been evaluated on. Maintain a Pareto frontier of prompts: a prompt enters the frontier if it achieves a strictly better score than every current frontier member on at least one example. This preserves diversity — the frontier can contain multiple prompts, each excelling on different subsets.

4. **System-Aware Merge.** Periodically merge two complementary frontier members via a second LLM call, producing a candidate that combines the strengths of both.

### 1.3 Key Mechanisms

**Pareto frontier.** The frontier is the core of GEPA's diversity preservation. It prevents convergence to a single local optimum by maintaining prompts that are each optimal for some subset of the training distribution. Selection is non-dominated: a new prompt is accepted if no existing frontier member dominates it on all evaluated examples.

**Minibatch sampling.** Training examples are served in epoch-shuffled minibatches. Each iteration sees only 3 examples, keeping reflection tractable and enabling fast iteration. The shuffled epoch ensures every example is seen before repeating.

**Reflective proposer.** The reflection prompt includes the current candidate prompt, a subset of failure trajectories, and the task specification. The proposer is unconstrained — it can propose any natural language prompt. This flexibility is both a strength and a risk: proposals can degenerate if the reflective context is too noisy.

### 1.4 Hyperparameters

```
Minibatch size (B):            3
Reflection LM max tokens:      2048
Task LM max tokens:            512  (QA tasks), 4096 (math)
Temperature:                   0.6
Top-P:                         0.95
Top-K:                         20
Rollout budgets (paper):
  hotpotqa:    6,871
  ifbench:     3,593
  hover:       2,426
  pupa:        3,936
  livebench:   1,839
```

### 1.5 Unified Metrics Framework

The following metrics define what **will be compared** across mutations once data coverage is sufficient (≥3 seeds per (model, benchmark) cell):

| Metric | Definition | Source |
|--------|-----------|--------|
| `test_score` | Accuracy on held-out test set | `result.json` |
| `val_score` | Best validation score achieved | `result.json` |
| `rollout_count` | Total task LM evaluations | `metrics.json` |
| `wall_clock_h` | Wall time in hours | `metrics.json` |
| `total_tokens` | Sum of all LM token usage | `metrics.json` |
| `task_input_tokens` | Tokens for task prompts | `metrics.json` |
| `task_output_tokens` | Task LM response tokens | `metrics.json` |
| `reflection_input_tokens` | Tokens for reflection prompts | `metrics.json` |
| `reflection_output_tokens` | Reflection LM response tokens | `metrics.json` |
| `reflection_call_count` | Number of LLM reflection calls | `metrics.json` |
| `score_per_1k_rollouts` | `test_score / rollout_count * 1000` | Derived |
| `prompt_length_tokens` | Final optimized prompt token count | `metrics.json` |

**Delta computation** (for future comparison tables):
```
Δtest_score       = mutation_mean_test - gepa_mean_test
rollout_ratio     = mutation_mean_rollouts / gepa_mean_rollouts
wall_clock_ratio  = mutation_mean_wall_h / gepa_mean_wall_h
spr_ratio         = mutation_spr / gepa_spr   # > 1 means more efficient
```

**Validity requirement:** Comparisons are only reported when both the mutation and GEPA baseline have ≥3 completed seeds at the same (model, benchmark) pair.

---

## 2. Mutation Taxonomy

The 11 mutations fall into three classes based on the architectural layer they modify:

### Proposer-Replacement Mutations
Replace or augment GEPA's reflective proposer while keeping the overall iteration loop, Pareto frontier, and minibatch sampling intact. Same rollout budget as GEPA.

| Mutation | Core change |
|----------|------------|
| `best_of_k` | Proposes K candidates per iteration, selects best |
| `contrastive_reflection` | Injects historical contrastive pairs into reflection prompt |
| `failure_stratified_k` | Best-of-K with partitioned failure coverage per candidate |
| `contrastive_synthesis` | Distills contrastive pairs into abstract principle via extra LLM call |
| `active_minibatch` | Selects high-variance (high-disagreement) examples for each minibatch |

### Standalone Search Algorithms
Replace the GEPA loop entirely with a different search strategy. Typically use far fewer rollouts.

| Mutation | Core change |
|----------|------------|
| `tournament` | 64-candidate single-elimination bracket |
| `iso` | Progressive pruning: 20→10→5→3→1 with inter-round mutation |
| `ant_colony` | Pheromone-based reinforcement over atomic prompt components |
| `synaptic_pruning` | One-shot: generate overspecified prompt, ablate sections, prune |

### Structural / Curriculum Mutations
Keep GEPA as the optimizer but change *how* the training set is presented or *how* the prompt is structured.

| Mutation | Core change |
|----------|------------|
| `ecological_succession` | Curriculum: easy → easy+medium → all examples across 3 phases |
| `modular` | Decompose prompt into 4 modules, optimize each via mini-GEPA, compose |

---

## 3. Individual Mutation Descriptions

---

### 3.1 Best-of-K (`best_of_k`, BoK)

#### 3.1.1 Motivation
Standard GEPA makes a single proposal per iteration. If the reflective LM happens to generate a poor mutation (e.g., an overly specific or hallucinated prompt), that entire iteration is wasted. Best-of-K hedges this risk: generate K independent proposals and select the best, at the cost of K× more LLM evaluation calls per iteration.

#### 3.1.2 Algorithm
At each GEPA iteration, after evaluating the current candidate and constructing the reflective dataset:

1. Run the standard reflection call to produce K=3 independent candidate prompts (via separate LLM calls or temperature-sampled variations).
2. Evaluate each of the K candidates on the **same minibatch** used for the current iteration (no extra training examples beyond the standard B=3).
3. Select the candidate with the **highest score sum** across the minibatch.
4. Proceed with GEPA's standard Pareto frontier update using the winning candidate.
5. Record the winning index (`k0_win_rate`) and whether candidates were unique (deduplication via text hash).

When K=1, the algorithm is identical to vanilla GEPA.

#### 3.1.3 Key Differences from GEPA
- **K extra proposal calls** per iteration (reflection LM called K times, not once).
- **K extra evaluation passes** on the shared minibatch (task LM evaluated on K candidates × B examples, not 1 × B).
- The Pareto frontier update, minibatch sampling, and everything else is unchanged.

#### 3.1.4 Hyperparameters
```
K (mutation_candidates):  3
selection_criterion:      score_sum_max over minibatch
deduplication:            text hash comparison
budget_model:             shared minibatch (all K candidates see same B examples)
```

#### 3.1.5 Computational Cost
- **Extra LLM calls per iteration:** K=3 (one reflection call per candidate)
- **Extra task evaluations per iteration:** (K−1)×B = 2×3 = 6 additional evaluations
- **Total rollout budget:** approximately K× GEPA at the same number of accepted iterations; in practice slightly less because some iterations are not best-of-K (warm-start iterations, convergence iterations)

#### 3.1.6 Preliminary Observations
Based on 34 completed runs across 5 benchmarks (qwen3-27b-awq primary, qwen3-14b and qwen3-1.7b on select benchmarks): On hotpotqa [n=5, qwen3-27b-awq], mean test score is +0.073 above GEPA at approximately the same rollout budget. On hover [n=5], mean test score is −0.153 vs GEPA, suggesting K does not consistently help across benchmarks. `k0_win_rate` (fraction of iterations where the first candidate wins) averages 0.52 for the 27b model, indicating genuine ensemble value; small models (1.7b) show k0_win_rate ≈ 0.89, suggesting first proposals are near-optimal for smaller models.

---

### 3.2 Contrastive Reflection (`contrastive_reflection`, CR)

#### 3.2.1 Motivation
Standard GEPA reflection looks only at the current candidate's failures. Contrastive reflection hypothesizes that more informative signal comes from *comparing* the current candidate with better-performing historical candidates on the same examples. By showing the LLM not just "this prompt failed here" but "this other prompt succeeded where yours failed," the reflective guidance becomes more targeted.

#### 3.2.2 Algorithm
After evaluating the current candidate on a minibatch:

1. **Contrastive mining (CPU-only).** Search the historical candidate pool for any candidate that outperformed the current one on the same training examples by at least `min_score_gap=0.1`. This is an O(n_candidates × n_examples) comparison — no LLM calls.
2. **Snippet extraction.** For each qualifying contrastive pair, extract up to 500 characters of the better-performing candidate's prompt as a "snippet."
3. **Injection.** Append the contrastive snippets as additional entries in the reflective dataset, tagged with the `<side_info>` label. This is rendered into the reflection prompt alongside standard failure trajectories.
4. **Standard reflection.** The LLM proposes a new candidate prompt using the augmented reflective dataset.
5. **Fallback.** If no contrastive pairs are found (gap < threshold, or no candidates yet), the iteration proceeds as standard GEPA with no side_info injection.

#### 3.2.3 Key Differences from GEPA
- CPU-only contrastive search step added between evaluation and proposal.
- Reflection prompt may contain `<side_info>` with high-performing snippets.
- **Zero extra LLM calls per iteration** — all additional processing is CPU.
- Rollout budget and iteration structure otherwise identical to GEPA.

#### 3.2.4 Hyperparameters
```
num_contrastive_pairs:  3
min_score_gap:          0.1   (minimum score difference to qualify)
include_full_text:      False (snippets only, not full prompt text)
max_snippet_length:     500 chars
```

#### 3.2.5 Computational Cost
- **Extra LLM calls per iteration:** 0
- **CPU overhead:** O(n_candidates × n_examples) per iteration — negligible
- **Rollout budget:** Identical to GEPA

#### 3.2.6 Preliminary Observations ⚠
Based on 25 completed runs (of 63 planned) — **60.3% of planned runs stalled without producing output** (gepa_state/ directory exists but no result.json). This is an *implementation stability concern*, not a data gap: all 5 hotpotqa/27b seeds failed to complete. Among completed runs, contrastive activation rate is very low: 0% on hover and livebench (no pairs found), 6–11% on ifbench and pupa. This suggests `min_score_gap=0.1` is too conservative for most benchmarks. Completed runs show modest Δtest of +2–4% on hover and pupa, but the stability failure blocks any broader assessment.

---

### 3.3 Failure-Stratified K (`failure_stratified_k`, FSK)

#### 3.3.1 Motivation
Best-of-K generates K independent proposals from the same reflective context, which may produce similar mutations. Failure-Stratified K introduces deliberate diversity: each of the K candidates is exposed to a *different subset of failing examples*, forcing each proposal to address distinct failure modes.

#### 3.3.2 Algorithm
After evaluating the current candidate on a minibatch:

1. Identify "failing" examples: training examples where the current candidate scores below `perfect_score_threshold=1.0`.
2. Sort failing examples by score ascending (worst failures first).
3. Assign failing examples to K partitions via round-robin: example 0 → partition 0, example 1 → partition 1, …, example K → partition 0, etc.
4. For each of the K candidates: construct a reflective dataset containing **all passing examples + partition k's failing examples only**.
5. Each candidate is proposed by the reflection LM using its unique reflective dataset.
6. Evaluate all K candidates on the shared minibatch; select the highest-scoring.
7. **Fallback:** If fewer than K failing examples exist, or if stratification is disabled, revert to standard best-of-K (all candidates see the same failing examples).

#### 3.3.3 Key Differences from GEPA
- Extends best_of_k by partitioning the reflective dataset across K candidates.
- Each proposal addresses a different subset of failures → deliberate diversity at proposal time.
- K=3 extra LLM calls and K extra evaluation passes per iteration (same as best_of_k).

#### 3.3.4 Hyperparameters
```
K (mutation_candidates):           3
perfect_score_threshold:           1.0
stratification_strategy:           round_robin_on_score_sorted_failures
fallback_when_few_failures:        true (reverts to best_of_k)
```

#### 3.3.5 Computational Cost
- **Extra LLM calls per iteration:** K=3 (same as best_of_k)
- **Extra task evaluations:** (K−1)×B per iteration
- **Rollout budget:** Same as best_of_k

#### 3.3.6 Preliminary Observations
Based on 7 completed runs across 2 benchmarks (qwen3-27b-awq only — sparse coverage). hover [n=3]: mean test ≈ 0.53 vs GEPA ≈ 0.70 (Δ ≈ −0.17). livebench [n=4]: mean test ≈ 0.39 vs GEPA ≈ 0.30 (Δ ≈ +0.09). Data is insufficient to draw conclusions about whether stratification improves over plain best_of_k. Other benchmarks (pupa, ifbench, hotpotqa) have incomplete or zero runs.

---

### 3.4 Synaptic Pruning (`synaptic_pruning`, SP)

#### 3.4.1 Motivation
GEPA's iterative refinement is expensive: each refinement step evaluates on a small minibatch and accumulates changes slowly. Synaptic Pruning takes the opposite approach: generate a maximally detailed prompt upfront (overspecified), then systematically ablate sections to find the minimal subset that preserves performance. The biological analogy is synaptic pruning in neural development — the brain generates more synapses than needed, then prunes unnecessary ones.

#### 3.4.2 Algorithm
The pipeline is **one-shot** (not iterative):

1. **Generate 3 overspecified prompts.** Use the reflection LM with 4 labeled (input, output) domain examples to generate 3 long, detailed candidate prompts (~500–2000 words each). The 4 labeled examples provide task-specific signal critical for domain-appropriate generation.
2. **Select best initial prompt.** Evaluate all 3 on a validation subset (first 40 examples). Select the highest-scoring as the ablation target.
3. **Parse into sections.** Attempt to parse the prompt into sections via: (a) markdown headers (`##`/`###`), (b) bolded paragraph headers (`**Name**`), (c) double-newline splits, (d) entire prompt as one section (fallback). Minimum section size: 80 characters.
4. **Section-level ablation.** For each section individually: remove it, re-evaluate on the validation subset, compute `delta = baseline_score - score_without_section`. Classify:
   - **Load-bearing** (`delta > 0.05`): critical, must keep
   - **Neutral** (`0.01 ≤ delta ≤ 0.05`): minor contribution, keep
   - **Prunable** (`delta < 0.01`): negligible or harmful, can remove
5. **Combined pruning + interaction check.** Remove all prunable sections at once. Re-evaluate. If combined removal causes a score drop > `interaction_threshold=0.03`, add prunable sections back one-by-one (least impactful first) until the drop is acceptable. Safety cap: never prune more than 50% of sections.
6. **Strengthen load-bearing sections.** For each load-bearing section, call the reflection LM to improve clarity and specificity.
7. **Final evaluation.** Evaluate the pruned+strengthened prompt on the full validation and test sets.

#### 3.4.3 Key Differences from GEPA
- **No iterative refinement loop.** This is a one-shot pipeline; there is no concept of "iterations."
- **Ablation-driven rather than reflection-driven.** Sections are selected or pruned based on their individual contribution, not via LLM-generated proposals.
- **Far fewer rollouts.** Ablation uses a 40-example validation subset; full training set evaluation is not used during optimization.
- **Interpretable diagnostics.** Section-level ablation deltas reveal which parts of the prompt matter.

#### 3.4.4 Hyperparameters
```
n_initial_prompts:        3
n_labeled_examples:       4      (for domain-specific generation)
ablation_val_subset:      40
load_bearing_threshold:   0.05
prunable_threshold:       0.01
interaction_threshold:    0.03
min_section_chars:        80
baseline_floor:           0.4    (skip ablation if initial score < 0.4)
max_sections_to_prune:    50%    (safety cap)
```

#### 3.4.5 Computational Cost
- **LLM calls:** 3 (initial generation) + N_sections (ablation, one per section, CPU re-evaluation) + up to N_load_bearing (strengthening). Typically 5–10 LLM calls total.
- **Rollout budget:** ~450 rollouts on hotpotqa (vs. 6,871 for GEPA) — approximately 15× fewer
- **Wall clock:** Typically 0.7–2.1 hours (vs. GEPA's 8–12 hours)

#### 3.4.6 Preliminary Observations
Based on 31 completed runs across 4 benchmarks and 4 models (best coverage of any mutation): hotpotqa/27b [n=5]: mean test ≈ 0.902 with ~451 rollouts, vs GEPA ≈ 0.848 with ~6,871 rollouts — preliminary evidence of quality-efficient tradeoff. pupa/27b [n=5]: mean test ≈ 0.521 vs GEPA ≈ 0.650 (Δ ≈ −0.13), suggesting the one-shot approach struggles on tasks requiring iterative multi-stage refinement. Typical pruning rate: 24–75% of sections pruned depending on prompt structure and task.

---

### 3.5 Tournament (`tournament`, PTS)

#### 3.5.1 Motivation
GEPA refines a single prompt lineage through many iterations. Tournament selection explores a wide population of diverse candidates upfront, then uses head-to-head evaluation to identify the strongest. This trades iteration depth for breadth: rather than making one prompt progressively better, find the best among many diverse candidates in fewer total evaluations.

#### 3.5.2 Algorithm

1. **Candidate pool generation.** Call the reflection LM 4 times using 4 distinct generation strategies:
   - Chain-of-thought / step-by-step reasoning prompts
   - Output format and structure variations
   - Varying detail level (terse to verbose)
   - Error prevention and verification strategies
   Each call generates ~16 diverse candidates. Combined with the seed prompt: 64 total candidates (pool_size = 64, a power of 2 for bracket structure).

2. **Single-elimination bracket.** Run 6 rounds:
   ```
   R0: 32 matchups × 5 examples  = 160 rollouts → 32 winners
   R1: 16 matchups × 7 examples  = 224 rollouts → 16 winners
   R2:  8 matchups × 10 examples = 160 rollouts →  8 winners
   R3:  4 matchups × 15 examples = 120 rollouts →  4 winners
   R4:  2 matchups × 20 examples =  80 rollouts →  2 winners
   R5:  1 matchup  × full valset            →  1 champion
   ```
   In each matchup, the candidate with the higher score sum on the assigned examples advances.

3. **Champion evaluation.** The R5 winner is evaluated on the full test set.

#### 3.5.3 Key Differences from GEPA
- **No iterative refinement.** Candidates are generated once and never improved — only selected or eliminated.
- **Population-based, not lineage-based.** 64 candidates compete simultaneously; GEPA maintains one active candidate at a time.
- **Diversity is upfront.** All exploration happens at generation time (4 strategies); the bracket only selects, not improves.
- **Known limitation:** Writes no intermediate checkpoint files — run progress is invisible to monitoring tools during execution.

#### 3.5.4 Hyperparameters
```
pool_size:              64 (power of 2 required)
n_generation_calls:     4
prompts_per_call:       16
bracket_depth:          6 (log2(64))
examples_per_round:     [5, 7, 10, 15, 20, full valset]
generation_strategies:  [chain_of_thought, format_variation, detail_spectrum, error_prevention]
```

#### 3.5.5 Computational Cost
- **LLM calls:** 4 total (generation only; evaluation is task-LM only)
- **Rollout budget:** ~540–1,504 rollouts (varies by valset size in R5); approximately 4–10× fewer than GEPA
- **Wall clock:** 1–5 hours (dominated by R5 full-valset evaluation)

#### 3.5.6 Preliminary Observations
Based on 30 completed runs across 5 benchmarks and 3 models: hotpotqa/27b [n=5]: mean test ≈ 0.902 using ~1,213 rollouts vs GEPA's ~6,871 (6× fewer). hover/27b [n=4]: mean test ≈ 0.459 vs GEPA ≈ 0.70 (Δ ≈ −0.24). Score spread analysis shows near-zero inter-candidate spread on ifbench (all matchup scores ~0.07), suggesting tournament cannot differentiate candidates on low-signal tasks where even randomly generated candidates perform similarly.

---

### 3.6 Slime Mold (`iso`, SMNO)

#### 3.6.1 Motivation
Tournament eliminates candidates purely by head-to-head comparison with no improvement. Slime Mold adds a failure-informed mutation step between rounds: surviving candidates are improved using failure information from the full round evaluation before the next round begins. This hybrid approach combines population-based exploration with targeted refinement, analogous to biological slime molds that extend pseudopodia toward nutrients and retract from poor paths while adapting their network structure.

#### 3.6.2 Algorithm

1. **Initial generation.** Generate 19 diverse candidate prompts via reflection LM (1 LLM call). Combined with seed = 20 candidates.

2. **Progressive pruning rounds:**
   ```
   R1: 20 candidates × 10 examples = 200 rollouts → keep top 10
       [inter-round mutation: each survivor mutated using R1 failure info]
   R2: 10 candidates × 15 examples = 150 rollouts → keep top 5
       [inter-round mutation]
   R3:  5 candidates × 20 examples = 100 rollouts → keep top 3
       [inter-round mutation]
   R4:  3 candidates × 30 examples =  90 rollouts → keep top 1
   ```
   Total: ~540 rollouts.

3. **Inter-round mutation.** Between each round, the reflection LM examines the surviving candidates and the examples they failed on (sampled from the full training set). It proposes improvements to each survivor independently. This produces a new population for the next round.

4. **Champion evaluation.** The final surviving candidate is evaluated on the full validation and test sets.

#### 3.6.3 Key Differences from GEPA
- Population-based with progressive narrowing (20→10→5→3→1 vs. GEPA's single lineage).
- Failure-aware inter-round mutation replaces GEPA's per-iteration reflection.
- Fixed round structure with predetermined survivor counts.
- ~8–12× fewer rollouts than GEPA.

#### 3.6.4 Hyperparameters
```
init_candidates:          20
rounds:                   4
survivors_per_round:      [10, 5, 3, 1]
examples_per_round:       [10, 15, 20, 30]
failure_sampling_size:    5  (examples used for inter-round mutation signal)
```

#### 3.6.5 Computational Cost
- **LLM calls:** ~22 total (1 initial generation + 4 rounds × ~5 survivors × mutation calls)
- **Rollout budget:** ~540 rollouts total; approximately 8–12× fewer than GEPA
- **Wall clock:** 1–4 hours

#### 3.6.6 Preliminary Observations
Based on 26 completed runs across 5 benchmarks and 2 models (good coverage): hotpotqa/27b [n=5]: mean test ≈ 0.897 using ~801 rollouts (11× fewer than GEPA). pupa/27b [n=10]: mean test ≈ 0.712. hover/27b [n=5]: mean test ≈ 0.491 vs GEPA ≈ 0.70 (Δ ≈ −0.21), consistent with the hover weakness seen in tournament. Score variance within rounds shrinks as expected: spread 0.6–1.0 in R1 narrows to 0.9–1.0 in R4 on hotpotqa, confirming progressive quality concentration.

---

### 3.7 Ant Colony (`ant_colony`, ACPCO)

#### 3.7.1 Motivation
Rather than treating a prompt as a monolithic text to optimize, Ant Colony decomposes it into reusable atomic components and searches over the combinatorial space of component compositions. Inspired by Ant Colony Optimization (ACO) — where ants reinforce successful paths via pheromone trails — this method reinforces components that contribute to high-scoring combinations.

#### 3.7.2 Algorithm

1. **Component library generation.** Generate ~50 components across 5 predefined categories via LLM (one call per category):
   - `task_framing` (10 components): role, scope, objective definitions
   - `reasoning_strategy` (10): logical process, decision rules
   - `format_instructions` (10): output structure, style requirements
   - `error_prevention` (10): validation, safety constraints
   - `domain_specific` (10): task-specific knowledge or heuristics
   Initial pheromone = 1.0 per component.

2. **Pheromone-based search loop** (50 rounds):
   - Each round: 3 "ants" independently sample 8–12 components weighted by current pheromone levels (sampling without replacement within a round).
   - Each ant evaluates its component combination on 10 training examples.
   - Pheromone update: components in above-median-scoring ants gain `score − median_score` pheromone. All components lose 10% pheromone (evaporation). Floor at 1e−6.
   - Total: 50 × 3 × 10 = 1,500 planned rollouts.

3. **Polish phase.** Select top-15 components by final pheromone. Call reflection LM once to compose them into a coherent prompt. Falls back to raw concatenation if LLM polish fails.

4. **Final evaluation.** Evaluate polished prompt on full validation and test sets.

#### 3.7.3 Key Differences from GEPA
- **Decomposes the prompt into components** rather than optimizing holistic prompt text.
- **Pheromone reinforcement** drives component selection — no LLM reflection per iteration.
- **No iterative text refinement.** Component vocabularies are generated once; the search is over compositions.
- Provides component-level interpretability: which categories dominate the final pheromone rankings.

#### 3.7.4 Hyperparameters
```
n_components:         50  (5 categories × 10 each)
n_ants:               3
n_components_per_ant: 10  (range: 8–12, randomized)
n_rounds:             50
evaporation_rate:     0.1  (10% decay per round)
top_k_polish:         15   (top-pheromone components for final assembly)
n_eval_per_round:     10   (examples per ant per round)
```

#### 3.7.5 Computational Cost
- **LLM calls:** ~6 total (5 category generation + 1 polish)
- **Rollout budget:** ~1,500 planned; early stopping if val_score = 1.0
- **Wall clock:** 1–2 hours

#### 3.7.6 Preliminary Observations
Based on 8 completed runs across 2 benchmarks (very sparse): hotpotqa/27b [n=1, seed 42]: test 0.923 at 65 rollouts (early stop; val=1.0). This single result is striking but **entirely unconfirmed** — it should not be interpreted as representative until ≥3 seeds are available. ifbench/27b [n=2]: mean test ≈ 0.074; ifbench/1.7b [n=5]: mean test ≈ 0.075. The component decomposition approach appears poorly matched to ifbench's instruction-following requirements, likely because the 5 fixed categories do not capture the nuanced constraint types in that benchmark.

---

### 3.8 Active Minibatch (`active_minibatch`, AMS)

#### 3.8.1 Motivation
GEPA samples minibatches uniformly (epoch-shuffled). Active Minibatch Selection applies active learning principles: prefer examples where the model is most uncertain or where different candidate prompts disagree most. These high-disagreement examples are hypothesized to carry the most signal for reflective improvement.

#### 3.8.2 Algorithm
Replaces GEPA's `EpochShuffledBatchSampler` with `ActiveMinibatchSampler`:

1. **Warmup phase** (iterations 1–10): epoch-shuffled sampling, identical to GEPA. Builds initial score history across examples.

2. **Active phase** (iteration 11+): construct each minibatch as:
   - **70% active selection:** rank all training examples by their per-example score variance (std dev of scores across recent candidate evaluations). Select the highest-variance examples. Unseen examples (no score history) are ranked highest to ensure early coverage.
   - **30% random fallback:** randomly sample remaining examples to prevent over-concentration on a fixed hard subset.

3. Variance is computed in CPU (no LLM calls). All other GEPA components — proposer, Pareto frontier, merge — are unchanged.

#### 3.8.3 Key Differences from GEPA
- Changes **which training examples** are selected per iteration.
- Everything else (proposer, Pareto selection, minibatch size) is identical to GEPA.
- Zero extra LLM calls; zero extra rollouts in expectation.

#### 3.8.4 Hyperparameters
```
warmup_iterations:   10
active_fraction:     0.70
random_fraction:     0.30
minibatch_size:      3  (same as GEPA)
variance_metric:     std dev of scores across recent evaluations
```

#### 3.8.5 Computational Cost
- **Extra LLM calls per iteration:** 0
- **CPU overhead:** O(n_examples) per iteration for variance ranking — negligible
- **Rollout budget:** Identical to GEPA

#### 3.8.6 Preliminary Observations
Based on 4 completed runs across 2 benchmarks (very sparse): hotpotqa/27b [n=1]: test 0.837 vs GEPA ≈ 0.850 using only 52 rollouts (early convergence). ifbench/1.7b [n=2]: mean test ≈ 0.072 vs GEPA ≈ 0.069 (+0.003). Notably, observed disagreement scores are uniformly low across tasks (mean 0.003–0.031 range), suggesting the variance signal is weak in practice — candidates converge quickly and fail to produce high-disagreement examples for active selection to exploit. This may indicate the warmup period needs extending, or that a different divergence metric is needed.

---

### 3.9 Contrastive Synthesis (`contrastive_synthesis`, CSR)

#### 3.9.1 Motivation
Contrastive reflection injects raw snippets from high-performing candidates — but these snippets may be noisy or task-specific. Contrastive Synthesis adds a distillation step: instead of injecting raw text, it calls the reflection LM to synthesize the contrastive pairs into a single abstract improvement principle. The hypothesis is that abstract principles ("use explicit constraints") transfer better across examples than specific prompt snippets.

#### 3.9.2 Algorithm
Extends contrastive_reflection with a synthesis step:

1. **Contrastive mining (CPU-only).** Identical to contrastive_reflection: find candidates that outperformed the current one by ≥`min_score_gap=0.1` on specific examples.
2. **Synthesis call (1 extra LLM call).** Pass the contrastive pairs to the reflection LM with the prompt: *"Distill the KEY PRINCIPLE that explains WHY the better prompt worked on these examples. State it as a single, actionable instruction (1–2 sentences)."* This call uses ~500 tokens.
3. **Principle injection.** Inject the synthesized principle as structured `side_info` (tagged `synthesis_principle`), rather than raw snippets, into the reflection prompt.
4. **Standard reflection.** The LLM proposes a new candidate using the augmented context.
5. **Fallback.** If no contrastive pairs are found or synthesis returns empty, proceed as standard GEPA.

#### 3.9.3 Key Differences from GEPA
- Adds contrastive mining (CPU) + 1 synthesis LLM call per iteration.
- Versus contrastive_reflection: injects a distilled principle rather than raw snippets.
- Versus GEPA: reflection prompt enriched with abstract side_info when synthesis is activated.

#### 3.9.4 Hyperparameters
```
num_contrastive_pairs:  3
min_score_gap:          0.1
max_snippet_length:     500 chars
synthesis_tokens:       ~500
extra_llm_calls:        1 per iteration (synthesis call)
```

#### 3.9.5 Computational Cost
- **Extra LLM calls per iteration:** 1 (synthesis, ~500 tokens)
- **Rollout budget:** Same as GEPA
- **Token overhead:** ~500 tokens × N_active_iterations (where synthesis activates); typically 2% of total reflection tokens

#### 3.9.6 Preliminary Observations
Based on 3 completed runs across 2 benchmarks (very sparse): ifbench/1.7b [n=2]: mean test ≈ 0.104. Synthesis activates in approximately 10% of iterations (when contrastive pairs exist). Sample synthesized principles are coherent — e.g., "Use precise, task-specific instructions that clearly define the expected output and the types of entities to extract" — suggesting the synthesis call produces meaningful abstractions. The 90% fallback rate (no pairs found in most iterations) mirrors the same activation issue seen in contrastive_reflection, likely inheriting the same `min_score_gap=0.1` conservatism.

---

### 3.10 Ecological Succession (`ecological_succession`, ESO)

#### 3.10.1 Motivation
GEPA trains on all examples simultaneously from the start. Ecological Succession applies curriculum learning: expose the optimizer to easy examples first, let it converge to a good general structure, then progressively introduce harder examples for refinement. The biological metaphor is ecological succession — ecosystems develop from pioneer species (generalist, hardy) through intermediate communities to mature, complex ecosystems.

#### 3.10.2 Algorithm

**Phase 0: Difficulty Estimation (overhead)**
Evaluate the seed prompt on every training example 3 times with stochastic sampling (temperature > 0). Compute mean score per example. Examples with high mean score are "easy" (the seed already handles them); low mean score are "hard."
- Partition: easy (top 20% by score), medium (next 30%), hard (bottom 50%)
- Rollout overhead: 3 × N_train (not counted in the GEPA budget; e.g., ~450 extra for N=150)

**Phase 1: Pioneer (15% of GEPA rollout budget)**
Run GEPA on easy examples only. Best prompt from Phase 1 seeds Phase 2.

**Phase 2: Shrub (30% of GEPA rollout budget)**
Run GEPA on easy + medium examples, warm-started from Phase 1 best. Best prompt seeds Phase 3.

**Phase 3: Forest (55% of GEPA rollout budget)**
Run GEPA on all examples, warm-started from Phase 2 best. Final prompt evaluated on test set.

#### 3.10.3 Key Differences from GEPA
- **Same proposer, Pareto selection, and iteration structure** as GEPA — but training set changes per phase.
- Phases share budget (total = GEPA budget + difficulty estimation overhead).
- Warm-start seeding across phases: each phase inherits the best prompt from the previous.
- Difficulty estimation adds a fixed rollout cost before optimization begins.

#### 3.10.4 Hyperparameters
```
pioneer_budget_fraction:    0.15
shrub_budget_fraction:      0.30
forest_budget_fraction:     0.55
difficulty_estimation_passes: 3  (per training example)
easy_percentile:            0.20
medium_percentile:          0.30  (easy + medium = top 50%)
```

#### 3.10.5 Computational Cost
- **Extra LLM calls per iteration:** 0 (identical to GEPA within each phase)
- **Extra rollout overhead:** 3 × N_train for difficulty estimation
- **Total rollouts:** GEPA budget + 3×N_train (e.g., 3,593 + 450 = 4,043 for ifbench)

#### 3.10.6 Preliminary Observations
Based on 9 completed runs across 2 benchmarks (sparse): hotpotqa/27b [n=1]: test 0.837 with 68 total rollouts including difficulty overhead. ifbench/27b [n=4]: highly seed-dependent — seed 42 achieved test=1.0 (suspected seed-specific alignment with curriculum ordering); seeds 456, 789, 1024 averaged test ≈ 0.067 (−0.053 vs GEPA). ifbench/1.7b [n=4]: mean test ≈ 0.072 (−0.018 vs GEPA). Results suggest curriculum outcome is sensitive to whether the difficulty estimation correctly partitions the specific seed's training examples — a reliability concern for the current 3-pass stochastic estimation approach.

---

### 3.11 Modular Decomposition (`modular`, PDMO)

#### 3.11.1 Motivation
Prompts are complex artifacts with multiple interacting concerns: task framing, reasoning strategy, output format, and error handling. Modular Decomposition hypothesizes that optimizing these concerns independently — each with dedicated budget — produces more targeted improvements than optimizing the full prompt jointly. A final composition step reassembles the modules.

#### 3.11.2 Algorithm

**Phase 1: Decompose.**
Call the reflection LM once to decompose the seed prompt into 4 semantic modules:
- `task_framing`: role, scope, objectives
- `reasoning_strategy`: logical processes, decision rules
- `format_constraints`: output structure, style
- `error_prevention`: validation, safety

**Phase 2: Module Optimization (sequential mini-GEPAs).**
For each of the 4 modules in sequence, run a mini-GEPA with budget/5 rollouts. The mini-GEPA optimizes only the module text while holding other modules fixed. The best module prompt from each run seeds the next module's optimization.

**Phase 3: Smooth Composition.**
Call the reflection LM once to compose all 4 optimized modules into a coherent unified prompt — smoothing away inconsistencies or redundancies introduced by independent optimization.

**Phase 4: Joint Refinement.**
Run a final GEPA pass (budget/5 rollouts) on the smoothed composed prompt, allowing module interactions to be refined together.

Total budget: 4×(budget/5) + budget/5 = budget (same as GEPA).

#### 3.11.3 Key Differences from GEPA
- Introduces a decompose-then-compose structure around the GEPA optimizer.
- Each module optimization is a mini-GEPA with reduced budget — same iteration mechanics.
- Two fixed LLM calls outside the optimization loop (decompose, compose).
- The independence assumption: modules are optimized without seeing each other's concurrent changes.

#### 3.11.4 Hyperparameters
```
n_modules:          4
modules:            [task_framing, reasoning_strategy, format_constraints, error_prevention]
per_module_budget:  budget / 5
final_joint_budget: budget / 5
decompose_calls:    1  (Phase 1)
compose_calls:      1  (Phase 3)
```

#### 3.11.5 Computational Cost
- **Extra LLM calls per iteration:** 0 (same as GEPA within each phase)
- **Fixed overhead:** 2 extra LLM calls per run (decompose + compose)
- **Rollout budget:** Identical to GEPA by construction

#### 3.11.6 Preliminary Observations
Based on 6 completed runs across 2 benchmarks (sparse): hotpotqa/27b [n=1]: test 0.830 vs GEPA ≈ 0.848 (Δ ≈ −0.018). ifbench/27b [n=2]: mean test ≈ 0.068 vs GEPA ≈ 0.12 (Δ ≈ −0.052). The Phase 3 smoothing step degraded performance in 5 of 6 runs (average Δ ≈ −0.005 after smoothing), with the Phase 4 joint refinement recovering 70–150% of the loss depending on model size. This pattern suggests the independence assumption in Phase 2 causes modules to diverge in ways that composition cannot fully reconcile, particularly on constraint-heavy tasks where modules tightly interact.

---

## 4. Unified Comparison Framework

### 4.1 Metrics Schema

When data coverage is sufficient, each mutation will be compared against GEPA using the schema defined in Section 1.5. All comparisons are **within (model, benchmark)** — cross-benchmark aggregation is not valid due to incomparable score distributions.

### 4.2 Data Requirements for Valid Comparison

A (mutation, model, benchmark) cell is considered **valid for comparison** when:
- ≥3 completed seeds exist for both the mutation AND GEPA at that (model, benchmark)
- No ≥50% stall rate (e.g., contrastive_reflection requires stability fix first)
- `result.json` is present for every counted seed

### 4.3 Current Data Coverage

| Mutation | Best-covered | Minimum seeds needed | Status |
|----------|-------------|---------------------|--------|
| `best_of_k` | 27b: 5 benchmarks × 5 seeds | More model sizes | Partially ready |
| `contrastive_reflection` | 27b: hover, livebench, pupa (5 seeds) | Fix 60% stall rate first | ⚠ Blocked |
| `failure_stratified_k` | 27b: hover (n=3), livebench (n=4) | More benchmarks + models | Insufficient |
| `synaptic_pruning` | 27b+14b: 4 benchmarks × 5 seeds | Nearly ready | Nearly ready |
| `tournament` | 27b+14b+8b: 5 benchmarks × 5 seeds | Ready for 27b | Ready |
| `iso` | 27b+14b: 5 benchmarks × 5 seeds | Ready for 27b | Ready |
| `ant_colony` | 27b: hotpotqa (n=1), ifbench (n=2) | ≥3 seeds on hotpotqa | Insufficient |
| `active_minibatch` | 27b: hotpotqa (n=1); 1.7b: ifbench (n=2) | More seeds on all | Insufficient |
| `contrastive_synthesis` | 1.7b: ifbench (n=2) | More benchmarks + seeds | Insufficient |
| `ecological_succession` | 27b+1.7b: ifbench (~4 each) | Fix seed sensitivity | Insufficient |
| `modular` | 27b+1.7b: ifbench (~3 total) | More seeds + benchmarks | Insufficient |

### 4.4 Known Gaps and What Is Needed

**Structural gaps (affecting multiple mutations):**
- hover and livebench are under-represented for all proposer-replacement mutations
- pupa and hotpotqa are missing for ecological_succession, contrastive_synthesis, modular, active_minibatch

**Implementation concerns:**
- `contrastive_reflection`: 60% stall rate — requires debugging before any valid comparison
- `contrastive_reflection` / `contrastive_synthesis`: `min_score_gap=0.1` activation threshold appears too conservative; consider reducing to 0.02–0.05

**Single-seed hotpotqa results** (ant_colony: 0.923, active_minibatch: 0.837, ecological_succession: 0.837, modular: 0.830, synaptic_pruning: 0.902): all require ≥3 additional seeds before these numbers can be reported with any confidence.

---

## 5. Summary Table

| Mutation | Class | Core Mechanism | Extra LLM Calls/iter | Rollout Budget vs. GEPA | Implementation Status |
|----------|-------|---------------|---------------------|------------------------|-----------------------|
| `best_of_k` | Proposer-replacement | K=3 independent proposals, select best | K=3 | ~K× (same iterations) | Stable |
| `contrastive_reflection` | Proposer-replacement | CPU contrastive mining → snippets in side_info | 0 | Same | ⚠ 60% stall rate |
| `failure_stratified_k` | Proposer-replacement | K proposals, partitioned failure coverage | K=3 | ~K× | Stable |
| `contrastive_synthesis` | Proposer-replacement | CPU contrastive mining → LLM distills principle | 1 (synthesis) | Same | Stable |
| `active_minibatch` | Proposer-replacement | High-variance example selection for minibatch | 0 | Same | Stable |
| `tournament` | Standalone search | 64-candidate single-elimination bracket | 4 (generation only) | ~1/5–1/10× | Stable; no checkpointing |
| `iso` | Standalone search | 20→1 progressive pruning + inter-round mutation | ~22 total | ~1/8–1/12× | Stable |
| `ant_colony` | Standalone search | Pheromone-reinforced component selection | ~6 total | ~1500 rollouts | Stable |
| `synaptic_pruning` | Standalone search | One-shot generate-ablate-prune pipeline | ~5–10 total | ~1/7–1/15× | Stable |
| `ecological_succession` | Curriculum | Easy→medium→all phase curriculum | Same as GEPA | GEPA + 3×N_train overhead | Stable; seed-sensitive |
| `modular` | Structural | Decompose prompt → per-module mini-GEPA → compose | 2 fixed (decompose, compose) | Same as GEPA | Stable |
