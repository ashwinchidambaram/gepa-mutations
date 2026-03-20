# Phase 2 Mutation Ideas -- Canonical Write-Up

Eight mutation experiments targeting different components of GEPA's optimization loop.
Each mutation modifies one or more aspects of the `optimize()` pipeline to test a specific
hypothesis about prompt evolution dynamics.

This document is the canonical reference produced by synthesizing three independent analyses
(Code Explorer, Paper Researcher, AI Research Team) against the v0.1.1 GEPA codebase.

---

## Summary Table

| # | Canonical Name              | Source | Priority | Prior Result    | Mechanism Target         | Status     |
|---|----------------------------|--------|----------|-----------------|--------------------------|------------|
| 1 | best_of_k                  | v2     | 1/8      | +0.300 (AIME)   | Mutation quality          | Validated  |
| 2 | contrastive_reflection     | v3     | 2/8      | Untested        | Reflection signal         | Untested   |
| 3 | stratified_batch           | v3     | 3/8      | Untested        | Training distribution     | Untested   |
| 4 | structured_dpg             | v2     | 4/8      | 0.100 (no gain) | Reflection decomposition  | Needs retest |
| 5 | lesson_memory              | v2     | 5/8      | 0.100 (no gain) | Cross-iteration learning  | Needs retest |
| 6 | phenotypic_diversity       | v2     | 6/8      | 0.100 (no gain) | Population diversity      | Needs retest |
| 7 | boltzmann_selection        | v2     | 7/8      | 0.100 (no gain) | Candidate selection       | Diagnostic |
| 8 | failure_stratified_k       | v3     | 8/8      | Untested        | K-candidate specialization| Untested   |

**Key findings from synthesis:**
- Agent C listed `contrastive_reflection` twice (as #4 and #7); these are the same mutation.
  The correct count is 8 unique mutations.
- All v2 results used gpt-oss-20b on AIME with budget=50, seed=0. Only `best_of_k` (K=3)
  produced a measurable gain (0.100 -> 0.400). All others produced 1 candidate matching
  baseline. These tests were heavily budget-constrained and should not be treated as definitive.
- v3 mutations (#2 contrastive_reflection, #3 stratified_batch, #8 failure_stratified_k)
  have never been tested and carry the highest information value.
- The binding constraint for budget=50 runs is **mutation quality** (number of accepted
  candidates). Mutations targeting selection/diversity (#6, #7) require larger Pareto
  fronts that only materialize with longer runs or stronger models.

---

## 1. Best-of-K Mutation Sampling (`best_of_k`)

**Source:** v2
**Priority:** 1/8

**Description:** Instead of proposing a single mutated prompt per iteration, generate K
independent mutations from the same parent candidate and reflective dataset, evaluate all
K on the same minibatch, and keep only the best-scoring one. This directly increases the
expected quality of each accepted mutation at the cost of K times the reflection + evaluation
budget per iteration.

**Hypothesis:** Generating K=3 mutation candidates per iteration will increase the number
of accepted candidates by at least 2x compared to K=1, translating to +3-10 percentage
points on benchmarks where the baseline produces fewer than 5 accepted candidates in a
full run (AIME, IFBench). The effect will be strongest on tasks where the reflection LM
produces high-variance but occasionally good mutations.

**Parameters and Sweep Values:**
- `mutation_candidates`: [1, 3, 5, 7] (default: 1, paper default: 1)

**Paper Baseline Comparison:** Compare against GEPA scores from Table 1. K=1 should
reproduce baseline exactly; K>1 consumes budget faster but should produce higher peak
scores within the same wallclock time.

**Prior Results (v2/v3):** K=3 on AIME (budget=50, gpt-oss-20b): 0.400 (4/10) vs
baseline 0.100 (1/10). The only mutation with a measurable gain in v2 testing.

**Implementation Notes:**
- GEPA components modified: `ReflectiveMutationProposer.propose()` (K-loop around
  `propose_new_texts()` + minibatch eval), `api.py` (new parameter)
- New classes/methods: None; extends existing `propose()` with K-loop
- Known bugs in v2/v3 code:
  - Callback undercounting: K evaluations fire but only 1 `on_evaluation_end` callback
    is emitted. Fix: emit K evaluation events or a single event with K sub-results.
  - No deduplication of identical mutations across K candidates. Fix: hash-compare
    proposed texts and skip duplicates to save budget.
- Integration point: Inside `ReflectiveMutationProposer.propose()`, after building
  the reflective dataset but before returning the `CandidateProposal`. The K-loop
  calls `propose_new_texts()` K times, evaluates each on the same minibatch, and
  returns only the best-scoring proposal.

**MutationConfig Compatibility:**
- Existing fields that apply: all current fields (K=1 reproduces baseline)
- New fields needed:
  - `mutation_candidates: int = 1`

---

## 2. Contrastive Reflection (`contrastive_reflection`)

**Source:** v3
**Priority:** 2/8

**Description:** After evaluating the current candidate on a minibatch, search the existing
candidate pool for candidates that solved examples the current candidate failed on. Inject
a short snippet (up to 300 characters) of the successful candidate's prompt text into the
reflection prompt, giving the reflection LM a concrete signal about what distinguishes
success from failure on specific examples.

**Hypothesis:** Contrastive snippets will produce more targeted mutations on tasks with
diverse failure modes, increasing accepted-candidate rate by 30-50% compared to baseline
reflection. Expected gain: +1-3 percentage points on HotpotQA, HoVer, IFBench where
failure patterns are heterogeneous. Effect will be minimal on AIME/math where failures
are more uniform.

**Parameters and Sweep Values:**
- `use_contrastive_reflection`: [True, False] (default: False)
- `contrastive_snippet_length`: [150, 300, 500] (default: 300)

**Paper Baseline Comparison:** Compare against GEPA scores from Table 1. The mutation
only activates when the Pareto front contains 2+ candidates with complementary success
patterns, so early iterations may behave identically to baseline.

**Prior Results (v2/v3):** Never tested standalone. No empirical data available.

**Implementation Notes:**
- GEPA components modified: `reflective_mutation.py` (new `_build_contrastive_text()`
  method), `instruction_proposal.py` (accepts contrastive text in prompt), `api.py`
  (new parameter)
- New classes/methods: `_build_contrastive_text()` on `ReflectiveMutationProposer`
- Known bugs in v2/v3 code:
  - **Val/train ID mismatch (critical):** The contrastive search looks up val subscores
    (`state.prog_candidate_val_subscores`) using train-set IDs from the minibatch.
    Train and val sets use different ID spaces, so the lookup will miss all matches.
    Fix: either search using val IDs mapped from train examples, or build a separate
    contrastive index on train-set evaluations.
  - Only the first component's snippet is shown even in multi-component candidates.
    Fix: include snippets for all components being updated.
- Integration point: In `ReflectiveMutationProposer.propose()`, after `capture_traces`
  eval and before calling `propose_new_texts()`. The contrastive text is prepended to
  the reflective dataset or injected via a modified prompt template.

**MutationConfig Compatibility:**
- Existing fields that apply: `reflection_prompt_template` (contrastive text may be
  injected via template modification)
- New fields needed:
  - `use_contrastive_reflection: bool = False`
  - `contrastive_snippet_length: int = 300`

---

## 3. Difficulty-Stratified Batch Sampling (`stratified_batch`)

**Source:** v3
**Priority:** 3/8

**Description:** Replace the default `EpochShuffledBatchSampler` with a
`DifficultyStratifiedBatchSampler` that partitions training examples into easy, medium,
and hard strata based on the current best candidate's validation scores. Each minibatch
is composed by drawing from all three strata, ensuring the reflection LM sees a balanced
mix of difficulty levels rather than a random sample that may be skewed toward easy
examples.

**Hypothesis:** Stratified sampling will produce mutations that generalize better across
difficulty levels, reducing the variance of validation scores by 20-40% compared to
epoch-shuffled sampling. Expected gain: +1-2 percentage points on benchmarks with
bimodal difficulty distributions (HotpotQA, HoVer). Minimal effect on uniformly
difficult benchmarks (AIME).

**Parameters and Sweep Values:**
- `batch_sampler`: ["epoch_shuffled", "difficulty_stratified"] (default: "epoch_shuffled")
- `stratified_easy_threshold`: [0.6, 0.7, 0.8] (default: 0.7)
- `stratified_hard_threshold`: [0.2, 0.3, 0.4] (default: 0.3)

**Paper Baseline Comparison:** Compare against GEPA scores from Table 1 with
epoch_shuffled sampler (paper default). The mutation is most informative on benchmarks
with at least 30 training examples to ensure meaningful stratification.

**Prior Results (v2/v3):** Never tested standalone. No empirical data available.

**Implementation Notes:**
- GEPA components modified: `batch_sampler.py` (new `DifficultyStratifiedBatchSampler`
  class), `api.py` (extended `batch_sampler` string factory)
- New classes/methods: `DifficultyStratifiedBatchSampler(BatchSampler)` -- drop-in
  replacement implementing the `BatchSampler` protocol
- Known bugs in v2/v3 code:
  - **Train/val ID mismatch (critical):** Uses `state.prog_candidate_val_subscores` to
    compute difficulty but receives the train `DataLoader`. If train and val use different
    ID spaces (which they do in most benchmarks), the difficulty lookup fails and the
    sampler degrades to random sampling silently. Fix: build difficulty scores from
    train-set evaluation history, or map val scores to train IDs via content matching.
  - No epoch tracking: unlike `EpochShuffledBatchSampler`, does not maintain epoch
    state, so coverage guarantees are lost. Fix: add epoch cycling within each stratum.
- Integration point: Drop-in replacement for `batch_sampler` parameter in `optimize()`.
  The `api.py` factory maps the string `"difficulty_stratified"` to the new class.

**MutationConfig Compatibility:**
- Existing fields that apply: `reflection_minibatch_size` (determines total batch size,
  divided across strata)
- New fields needed:
  - `batch_sampler: str = "epoch_shuffled"` (already exists as optimize() param but
    not in MutationConfig; must be added)
  - `stratified_easy_threshold: float = 0.7`
  - `stratified_hard_threshold: float = 0.3`

---

## 4. Structured DPG Reflection (`structured_dpg`)

**Source:** v2
**Priority:** 4/8

**Description:** Replace GEPA's single-pass reflection (one LM call that produces the
new prompt) with a three-phase Diagnose-Prescribe-Generate (DPG) pipeline. Phase 1
(Diagnose) identifies specific failure patterns in the reflective dataset. Phase 2
(Prescribe) proposes concrete changes to address each failure. Phase 3 (Generate)
produces the new prompt text incorporating the prescribed changes. Each phase is a
separate LM call.

**Hypothesis:** Decomposing reflection into 3 sequential LM calls will produce more
targeted mutations on tasks with structured failure modes, increasing accepted-candidate
rate by 25-50% compared to single-pass reflection. Expected gain: +1-3 percentage points
on IFBench (constraint failures), HoVer (label matching), and AIME (reasoning errors).
The effect will be model-dependent: stronger models benefit more from structured prompting
while weaker models may produce vague diagnoses that compound errors across phases.

**Parameters and Sweep Values:**
- `reflection_strategy`: ["default", "structured"] (default: "default")

**Paper Baseline Comparison:** Compare against GEPA scores from Table 1. Structured DPG
uses 3x the reflection LM tokens per iteration, so the fair comparison holds total
reflection token budget constant by reducing the number of iterations proportionally.

**Prior Results (v2/v3):** AIME (budget=50, gpt-oss-20b): 0.100 (1/10), matching
baseline. Only 1 candidate was found. The structured path produced vague diagnoses like
"the model needs to think harder" rather than actionable prescriptions. This is likely
a model capability issue -- stronger reflection LMs may benefit more.

**Implementation Notes:**
- GEPA components modified: `reflective_mutation.py` (strategy dispatch in
  `propose_new_texts()`), `instruction_proposal.py` (new `run_structured()` method),
  `api.py` (new parameter)
- New classes/methods: `run_structured()` (3-phase DPG pipeline),
  `_format_samples_static()` (text-only sample formatter for structured path)
- Known bugs in v2/v3 code:
  - No Image handling in the structured path: if the reflective dataset contains
    `Image` objects, `_format_samples_static()` will fail or produce placeholder text.
    Fix: use the same `format_samples()` renderer from `InstructionProposalSignature`.
  - No contrastive injection in structured path: if `use_contrastive_reflection=True`
    is set alongside `reflection_strategy="structured"`, the contrastive text is
    silently dropped. Fix: inject contrastive text into the Diagnose phase.
- Integration point: In `ReflectiveMutationProposer.propose_new_texts()`, a strategy
  dispatch selects between the default `InstructionProposalSignature.run()` and the new
  `run_structured()` based on the `reflection_strategy` parameter.

**MutationConfig Compatibility:**
- Existing fields that apply: `reflection_prompt_template` (used for the Generate
  phase; Diagnose and Prescribe use hardcoded prompts)
- New fields needed:
  - `reflection_strategy: str = "default"`

---

## 5. Cross-Iteration Lesson Memory (`lesson_memory`)

**Source:** v2
**Priority:** 5/8

**Description:** Maintain a rolling window buffer of "lessons" learned from previous
mutation attempts. Each lesson records what was tried, whether it improved scores, and
a brief summary of the change. Before each reflection, the buffer contents are rendered
and injected into the reflection prompt, giving the LM context about what has already
been attempted and what worked or failed.

**Hypothesis:** Lesson memory will reduce redundant mutations (attempting changes similar
to previously rejected ones) and amplify successful patterns, increasing the diversity
of accepted candidates by 20-40%. Expected gain: +1-3 percentage points on longer runs
(budget >= 200) where the optimization has time to accumulate meaningful lessons. Effect
will be negligible on short runs (budget <= 50) where the buffer never fills.

**Parameters and Sweep Values:**
- `use_lesson_memory`: [True, False] (default: False)
- `lesson_window_size`: [3, 5, 10] (default: 5)
- `lesson_max_rejected`: [1, 2, 3] (default: 2)

**Paper Baseline Comparison:** Compare against GEPA scores from Table 1. Lesson memory
adds a fixed token overhead per reflection call (proportional to window size), so the
comparison should control for total reflection tokens.

**Prior Results (v2/v3):** AIME (budget=50, gpt-oss-20b): 0.100 (1/10), matching
baseline. The budget was likely too short for the lesson buffer to accumulate useful
signal. Retest with budget >= 200 and a benchmark with more training examples.

**Implementation Notes:**
- GEPA components modified: NEW `lesson_memory.py` module, `api.py` (new parameter),
  `reflective_mutation.py` (records lessons, injects into reflection), `instruction_proposal.py`
  (accepts `<lessons>` placeholder or prepend)
- New classes/methods: `Lesson` (dataclass: iteration, change_summary, accepted,
  score_delta), `LessonMemory` (deque-based rolling window, render method)
- Known bugs in v2/v3 code:
  - **"Accepted" flag mismatch:** The lesson's `accepted` flag is computed at the
    proposer level (new subsample score > old subsample score) but the engine may still
    reject the candidate after full val-set evaluation. This means some "accepted"
    lessons in the buffer actually correspond to rejected candidates. Fix: record the
    lesson after the engine's final accept/reject decision via a callback.
- Integration point: `LessonMemory` is instantiated in `optimize()` and passed to
  `ReflectiveMutationProposer`. After each mutation attempt, a lesson is recorded.
  Before each `propose_new_texts()` call, rendered lessons are injected into the
  reflective dataset or prompt template.

**MutationConfig Compatibility:**
- Existing fields that apply: `reflection_prompt_template` (lessons may be injected
  via a `<lessons>` placeholder)
- New fields needed:
  - `use_lesson_memory: bool = False`
  - `lesson_window_size: int = 5`
  - `lesson_max_rejected: int = 2`

---

## 6. Phenotypic Diversity Pressure (`phenotypic_diversity`)

**Source:** v2
**Priority:** 6/8

**Description:** After evaluating a new candidate on the full validation set, check
whether its per-example success pattern (binary vector of which examples it solves)
overlaps excessively with any existing candidate on the Pareto front. If the overlap
exceeds a threshold, the candidate is flagged as a "phenotypic duplicate" -- it solves
essentially the same subset of problems as an existing candidate and thus adds no
diversity to the front.

This mutation also includes a companion `DiverseParetoSelector` in `candidate_selector.py`
that biases parent selection toward candidates whose success patterns are least covered
by recent mutations.

**Hypothesis:** Phenotypic deduplication will increase the effective diversity of the
Pareto front by blocking redundant candidates, leading to more diverse parent selection
and higher-quality mutations. Expected gain: +0-2 percentage points, but primarily
visible on longer runs (budget >= 500) where the Pareto front grows large enough for
duplicates to appear. Effect is front-size-dependent.

**Parameters and Sweep Values:**
- `phenotypic_dedup_threshold`: [0.7, 0.8, 0.9, None] (default: None = disabled)
- `diversity_decay`: [0.2, 0.3, 0.5] (default: 0.3, for DiverseParetoSelector)
- `diversity_window`: [2, 3, 5] (default: 3, for DiverseParetoSelector)

**Paper Baseline Comparison:** Compare against GEPA scores from Table 1. Phenotypic
diversity is most informative on benchmarks with large validation sets (HotpotQA: 100+,
HoVer: 100+) where the success vectors have meaningful structure.

**Prior Results (v2/v3):** AIME (budget=50, gpt-oss-20b): 0.100 (1/10), matching
baseline. Only 1 candidate was produced, so the dedup logic never activated.
Inconclusive -- the mutation needs a run that produces multiple candidates.

**Implementation Notes:**
- GEPA components modified: `engine.py` (new `_is_phenotypic_duplicate()` method after
  val-set eval), `candidate_selector.py` (new `DiverseParetoSelector`), `api.py`
  (new parameters)
- New classes/methods: `_is_phenotypic_duplicate()` on `GEPAEngine`,
  `DiverseParetoSelector(CandidateSelector)` (in `candidate_selector.py`)
- Known bugs in v2/v3 code:
  - **Not actually blocking (critical):** The `_is_phenotypic_duplicate()` method sets
    a trace flag but does NOT prevent the candidate from being added to the state.
    It is informational only. Fix: add a conditional branch in `_run_full_eval_and_add()`
    that skips `state.update_state_with_new_program()` when the duplicate flag is set.
  - **Asymmetric overlap:** Only checks whether the new candidate's successes are a
    subset of an existing candidate's, not vice versa. A candidate that solves a strict
    superset would not be flagged. This may be intentional (supersets are strictly
    better) but should be documented.
- Integration point: In `GEPAEngine._run_full_eval_and_add()`, after `_evaluate_on_valset()`
  returns scores but before `state.update_state_with_new_program()`. The duplicate check
  compares the binary success vector (score >= threshold) against all existing candidates.

**MutationConfig Compatibility:**
- Existing fields that apply: `candidate_selection_strategy` (set to `"diverse_pareto"`
  for the companion selector)
- New fields needed:
  - `phenotypic_dedup_threshold: float | None = None`
  - `diversity_decay: float = 0.3`
  - `diversity_window: int = 3`

---

## 7. Boltzmann Candidate Selection (`boltzmann_selection`)

**Source:** v2
**Priority:** 7/8

**Description:** Replace GEPA's Pareto-based candidate selection with a Boltzmann
(softmax) selector. Instead of sampling uniformly from the Pareto front, candidates are
selected with probability proportional to exp(score / temperature). Lower temperatures
concentrate selection on the best candidates; higher temperatures spread selection more
evenly. This provides a smooth interpolation between greedy (temperature -> 0) and
uniform (temperature -> infinity) selection.

**Hypothesis:** Boltzmann selection with temperature=0.5 will slightly improve convergence
speed over Pareto selection by concentrating mutations on higher-scoring parents while
maintaining some exploration. Expected gain: +0-1 percentage points. This is primarily
a diagnostic mutation -- it tests whether parent selection strategy matters at all when
mutation quality is the binding constraint.

**Parameters and Sweep Values:**
- `candidate_selection_strategy`: ["pareto", "boltzmann"] (default: "pareto")
- `boltzmann_temperature`: [0.1, 0.3, 0.5, 1.0] (default: 0.5)

**Paper Baseline Comparison:** Compare against GEPA scores from Table 1 with Pareto
selection. The mutation is only informative when the candidate pool has 3+ candidates,
so it should be tested on longer runs or combined with `best_of_k`.

**Prior Results (v2/v3):** AIME (budget=50, gpt-oss-20b): 0.100 (1/10), matching
baseline. 3 candidates were produced (same as baseline), suggesting selection strategy
had no effect when mutation quality is the bottleneck.

**Implementation Notes:**
- GEPA components modified: `candidate_selector.py` (new `BoltzmannCandidateSelector`
  class), `api.py` (extended selector factory)
- New classes/methods: `BoltzmannCandidateSelector(CandidateSelector)` with
  temperature-scaled softmax over `state.program_full_scores_val_set`
- Known bugs in v2/v3 code:
  - Temperature is hardcoded at 0.5 in the `api.py` factory. Fix: add
    `boltzmann_temperature` parameter to `optimize()` and pass through to the factory.
  - The `DiverseParetoSelector` (from mutation #6) was added in the same diff as
    `"diverse_pareto"` -- these are separate mutations sharing a file.
- Implementation is numerically stable (log-sum-exp trick).
- Integration point: Drop-in replacement for the `candidate_selection_strategy`
  parameter. The factory in `api.py` maps `"boltzmann"` to
  `BoltzmannCandidateSelector(temperature=0.5, rng=rng)`.

**MutationConfig Compatibility:**
- Existing fields that apply: `candidate_selection_strategy` (set to `"boltzmann"`)
- New fields needed:
  - `boltzmann_temperature: float = 0.5`

---

## 8. Failure-Stratified K (`failure_stratified_k`)

**Source:** v3
**Priority:** 8/8

**Description:** A refinement of `best_of_k` (mutation #1). Instead of generating K
independent mutations from the same reflective dataset, partition the reflective dataset's
failing examples across the K candidates so each candidate focuses on a different subset
of failures. This aims to produce K diverse mutations that each specialize in fixing a
different failure mode, rather than K similar mutations that address the same obvious
failure.

**Hypothesis:** Failure-stratified K will produce more diverse accepted candidates than
standard best-of-K, with complementary success patterns that improve the Pareto front
more efficiently. Expected gain: +0.5-2 percentage points over best_of_k on benchmarks
with heterogeneous failure modes (HotpotQA, HoVer, IFBench). Effect is minimal on
benchmarks with homogeneous failures (AIME).

**Parameters and Sweep Values:**
- `use_failure_stratified_k`: [True, False] (default: False)
- `mutation_candidates`: [3, 5, 7] (default: 1; must be > 1 for stratification to activate)

**Paper Baseline Comparison:** Compare against both baseline GEPA and `best_of_k` with
the same K. The mutation tests whether specialization of K candidates outperforms
independent sampling of K candidates.

**Prior Results (v2/v3):** Never tested standalone. No empirical data available.

**Implementation Notes:**
- GEPA components modified: `reflective_mutation.py` (new
  `_stratify_reflective_dataset()` method), `api.py` (new parameter)
- New classes/methods: `_stratify_reflective_dataset()` on `ReflectiveMutationProposer`
  -- partitions the failing examples into K groups for K iterations of the K-loop
- Known bugs in v2/v3 code:
  - **Degenerate with 1 failing example:** If only 1 example failed in the minibatch,
    all K candidates receive the same single-example dataset, making stratification
    identical to standard best-of-K. Fix: fall back to standard K-sampling when
    failing examples < K.
  - **Silent no-op at K=1:** If `mutation_candidates=1`, the stratification code path
    is entered but produces a single partition identical to the full dataset. Fix:
    either skip stratification when K=1 or validate K>1 at config time.
  - Requires `mutation_candidates > 1` (from mutation #1) to function.
- Integration point: Inside the K-loop in `ReflectiveMutationProposer.propose()`,
  before each call to `propose_new_texts()`. The reflective dataset is partitioned
  so candidate k receives failing examples in partition k.

**MutationConfig Compatibility:**
- Existing fields that apply: `mutation_candidates` (from mutation #1)
- New fields needed:
  - `use_failure_stratified_k: bool = False`

---

## Cross-Check Notes and Inconsistencies Resolved

### Naming Resolutions

| Agent A Name                       | Agent C Name             | Canonical Name            | Rationale                          |
|-----------------------------------|--------------------------|---------------------------|------------------------------------|
| Best-of-K Mutation Sampling       | best_of_k                | `best_of_k`               | Concise, matches API param         |
| Cross-Iteration Lesson Memory     | lesson_memory            | `lesson_memory`            | Shorter, clear                     |
| Phenotypic Diversity Pressure     | phenotypic_diversity     | `phenotypic_diversity`     | Matches the mechanism              |
| Boltzmann Candidate Selection     | boltzmann_selection      | `boltzmann_selection`      | Matches API string                 |
| Structured DPG Reflection         | structured_dpg           | `structured_dpg`           | DPG is the key distinguisher       |
| Difficulty-Stratified Batch       | stratified_batch         | `stratified_batch`         | Shorter, clear                     |
| Contrastive Reflection            | contrastive_reflection   | `contrastive_reflection`   | Direct, unambiguous                |
| Failure-Stratified K / Diverse K  | failure_stratified_k     | `failure_stratified_k`     | Distinguishes from generic K       |

### Agent C Duplicate Error

Agent C listed `contrastive_reflection` as both #4 and #7, giving it two priority
rankings. This is a single mutation. The correct canonical list has 8 unique mutations.
Agent C's `boltzmann_selection` was omitted from their numbering as a result. The
priority ranking in this document resolves the conflict.

### Priority Ranking Disagreement

Agent B ranked `failure_stratified_k` at #2 ("best-formed hypothesis") while Agent C
ranked it at #8. The synthesis ranks it at #8 because: (a) it depends on `best_of_k`
(#1) working first, (b) it adds complexity without independent empirical evidence,
(c) it has a degenerate failure mode with few failing examples. It remains a valuable
refinement to test after `best_of_k` is validated on more benchmarks.

Agent B ranked `boltzmann_selection` at #7 and Agent C omitted it from ranking.
Both agree it is low-leverage. Placed at #7 as a diagnostic mutation.

### v2/v3 Relationship

Agent A confirmed: v3 is a strict byte-identical superset of v2 for shared files.
v3 adds mutations #2 (`contrastive_reflection`), #3 (`stratified_batch`), and
#8 (`failure_stratified_k`). All v2 mutations (#1, #4, #5, #6, #7) are present
in v3 unchanged.

---

## MutationConfig Gap Assessment

### Current MutationConfig Fields

```python
@dataclass
class MutationConfig:
    mutation_name: str
    description: str = ""
    benchmark: str = "hotpotqa"
    seed: int = 42
    subset: int | None = None
    candidate_selection_strategy: str | Any = "pareto"
    frontier_type: str = "instance"
    module_selector: str = "round_robin"
    use_merge: bool = True
    max_merge_invocations: int = 5
    merge_val_overlap_floor: int = 5
    reflection_minibatch_size: int = 3
    reflection_prompt_template: str | dict[str, str] | None = None
    max_metric_calls: int | None = None
```

### Fields That Need To Be Added

```python
# --- Mutation #1: best_of_k ---
mutation_candidates: int = 1

# --- Mutation #2: contrastive_reflection ---
use_contrastive_reflection: bool = False
contrastive_snippet_length: int = 300

# --- Mutation #3: stratified_batch ---
batch_sampler: str = "epoch_shuffled"          # NOTE: already an optimize() param
stratified_easy_threshold: float = 0.7         #       but missing from MutationConfig
stratified_hard_threshold: float = 0.3

# --- Mutation #4: structured_dpg ---
reflection_strategy: str = "default"

# --- Mutation #5: lesson_memory ---
use_lesson_memory: bool = False
lesson_window_size: int = 5
lesson_max_rejected: int = 2

# --- Mutation #6: phenotypic_diversity ---
phenotypic_dedup_threshold: float | None = None
diversity_decay: float = 0.3                   # For DiverseParetoSelector
diversity_window: int = 3                      # For DiverseParetoSelector

# --- Mutation #7: boltzmann_selection ---
boltzmann_temperature: float = 0.5

# --- Mutation #8: failure_stratified_k ---
use_failure_stratified_k: bool = False
```

### Integration Notes for MutationConfig

1. **`batch_sampler` gap:** The `optimize()` API already accepts `batch_sampler` as a
   parameter, but `MutationConfig` does not expose it. The `run_mutation()` function in
   `base.py` does not pass it through to `optimize()`. This must be fixed for mutation #3.

2. **`boltzmann_temperature` threading:** The current `api.py` factory hardcodes
   temperature=0.5 for the `"boltzmann"` selector. The temperature parameter must be
   threaded from `MutationConfig` through `run_mutation()` to the factory. The cleanest
   approach: pass a pre-built `BoltzmannCandidateSelector(temperature=X)` instance via
   `candidate_selection_strategy` (which already accepts `str | Any`).

3. **`diversity_decay` / `diversity_window` threading:** Same pattern as boltzmann.
   Pass a pre-built `DiverseParetoSelector(decay=X, window=Y)` via
   `candidate_selection_strategy`.

4. **None-defaults preserve baseline:** All new fields default to values that reproduce
   the paper baseline exactly (mutation disabled). This ensures that a `MutationConfig`
   with only `mutation_name` and `benchmark` set will behave identically to Phase 1
   reproduction runs.

5. **`run_mutation()` pass-through:** The `run_mutation()` function in `base.py` must
   be updated to forward the new MutationConfig fields to `optimize()`. Fields that
   modify GEPA internals (mutations #1, #2, #4, #5, #6, #8) require patches to the
   GEPA source or wrapper classes; they cannot be configured via `optimize()` kwargs
   alone. Only mutations #3 and #7 are pure `optimize()` parameter changes.

### Recommended Implementation Order

1. `best_of_k` -- validated, highest expected impact, single integration point
2. `contrastive_reflection` -- untested but strong theoretical basis, moderate complexity
3. `stratified_batch` -- clean drop-in replacement, fix train/val ID bug first
4. `structured_dpg` -- retest with stronger model before investing in bug fixes
5. `lesson_memory` -- retest with larger budget before investing in bug fixes
6. `phenotypic_diversity` -- fix the "not actually blocking" bug, then retest
7. `boltzmann_selection` -- diagnostic, lowest implementation effort
8. `failure_stratified_k` -- implement after `best_of_k` is validated on more benchmarks
