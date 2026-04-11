# Mutation Plan: contrastive_reflection

## Mutation Summary

After evaluating the current candidate on a minibatch, search the existing candidate pool for candidates that solved examples the current candidate failed on. Inject a short snippet (up to 300 characters) of the successful candidate's prompt text into the reflection prompt, giving the reflection LM a concrete signal about what distinguishes success from failure.

**Prior result**: Never tested standalone. No empirical data available.

---

## 1. Implementation Approach

### Step-by-step code changes

**Step 1: Create `ContrastiveReflectionProposer` subclass**

Create `src/gepa_mutations/experiments/contrastive_reflection/proposer.py` containing a subclass of `ReflectiveMutationProposer` that overrides `propose()` to inject contrastive snippets between the reflective dataset construction and the `propose_new_texts()` call.

The override keeps all of the parent's logic intact (candidate selection, minibatch sampling, evaluation, cache updates, callback notifications, skip-perfect logic) and only inserts the contrastive search + injection logic at exactly one point: after `make_reflective_dataset()` returns and before `propose_new_texts()` is called (between steps 2 and 3 in the original flow at lines 265-297 of `reflective_mutation.py`).

**Step 2: Implement contrastive search function**

Create a standalone function `find_contrastive_candidates()` in `src/gepa_mutations/experiments/contrastive_reflection/contrastive_search.py`:

```python
def find_contrastive_candidates(
    state: GEPAState,
    current_candidate_idx: int,
    train_scores: dict[DataId, float],  # scores from current eval, keyed by TRAIN IDs
    failing_threshold: float = 0.5,
) -> dict[str, str]:
    """
    For each train example where current candidate scored below threshold,
    search OTHER candidates' TRAIN-SET evaluations (NOT val subscores)
    for candidates that scored higher on that same example.

    Returns: dict mapping component_name -> snippet (up to contrastive_snippet_length chars)
    from the best contrastive candidate.
    """
```

This function:
1. Identifies failing train examples from `train_scores` (the scores returned by `adapter.evaluate()` on the minibatch, keyed by `subsample_ids` which are train-set IDs).
2. Searches the **evaluation cache** (`state.evaluation_cache`) for other candidates that scored higher on the same train-set example IDs. Does NOT use `state.prog_candidate_val_subscores` (which uses val-set IDs -- see Bug #1 in Section 7).
3. Ranks contrastive candidates by how many of the current candidate's failures they solve.
4. Extracts a snippet from the top contrastive candidate's prompt text for **each** component being updated (see Bug #2 in Section 7).

**Step 3: Implement snippet injection**

Create `inject_contrastive_snippets()` in `src/gepa_mutations/experiments/contrastive_reflection/injection.py`:

Two injection strategies (selected via config):

- **Strategy A (`side_info`)**: Append a contrastive block to the `<side_info>` section of the reflective dataset. This works by adding an extra entry to the reflective dataset sequence for each component, formatted as:
  ```
  ## Contrastive Signal
  Another candidate solved examples that the current candidate failed on.
  The key difference in its instructions was:
  "{snippet}"
  Consider what makes this approach successful and incorporate relevant elements.
  ```

- **Strategy B (`template`)**: Use a custom `reflection_prompt_template` that includes a `<contrastive>` placeholder in addition to `<curr_param>` and `<side_info>`. Requires passing the custom template to `InstructionProposalSignature`. Note: the `validate_prompt_template()` method only checks for `<curr_param>` and `<side_info>`, so a `<contrastive>` placeholder will not be rejected but also will not be auto-replaced by GEPA -- our code must handle the replacement before passing the template.

Default strategy: **A (`side_info`)**, because it requires no changes to InstructionProposalSignature and works with the existing prompt template validation.

**Step 4: Wire into the propose() override**

In `ContrastiveReflectionProposer.propose()`:

```
# ... (all parent logic through make_reflective_dataset) ...

# === CONTRASTIVE INJECTION POINT ===
if self.use_contrastive_reflection:
    # Build train_scores dict from current eval
    train_scores = dict(zip(subsample_ids, eval_curr.scores))

    contrastive_snippets = find_contrastive_candidates(
        state=state,
        current_candidate_idx=curr_prog_id,
        train_scores=train_scores,
        failing_threshold=0.5,
    )

    if contrastive_snippets:
        reflective_dataset = inject_contrastive_snippets(
            reflective_dataset=reflective_dataset,
            contrastive_snippets=contrastive_snippets,
            components_to_update=predictor_names_to_update,
            max_snippet_length=self.contrastive_snippet_length,
        )

# ... (continue with propose_new_texts) ...
```

**Step 5: Build a contrastive train-score index**

Since `state.prog_candidate_val_subscores` only stores val-set scores and the evaluation cache is keyed by candidate hash (not candidate index), we need an auxiliary data structure.

Create `ContrastiveTrainIndex` in `contrastive_search.py`:
- On each iteration, after the current candidate is evaluated on the minibatch, record `{candidate_idx: {train_id: score}}` in the index.
- The index is maintained in the proposer instance and passed to `find_contrastive_candidates()`.
- This avoids the val/train ID mismatch bug entirely by building a purpose-built index from train evaluations.

**Step 6: Create experiment runner config**

Create `src/gepa_mutations/experiments/contrastive_reflection/config.py` with a Pydantic model for experiment-specific settings, plus a factory function that constructs the `ContrastiveReflectionProposer` with the right parameters.

---

## 2. GEPA Components to Modify

### Components we DO NOT modify (patches avoided)

| File | Reason |
|------|--------|
| `gepa/src/gepa/proposer/reflective_mutation/reflective_mutation.py` | We subclass, not patch |
| `gepa/src/gepa/strategies/instruction_proposal.py` | The `<side_info>` injection strategy uses the existing template without modification |
| `gepa/src/gepa/core/state.py` | We read state fields; never write custom fields onto GEPAState |

All changes live outside the `gepa/` submodule. The official GEPA source remains untouched.

### Components we extend (clean extensions)

| New File | Purpose |
|----------|---------|
| `src/gepa_mutations/experiments/contrastive_reflection/proposer.py` | `ContrastiveReflectionProposer` subclass of `ReflectiveMutationProposer` |
| `src/gepa_mutations/experiments/contrastive_reflection/contrastive_search.py` | `find_contrastive_candidates()` + `ContrastiveTrainIndex` |
| `src/gepa_mutations/experiments/contrastive_reflection/injection.py` | `inject_contrastive_snippets()` with side_info strategy |
| `src/gepa_mutations/experiments/contrastive_reflection/config.py` | `ContrastiveReflectionConfig` pydantic model + factory |
| `src/gepa_mutations/experiments/contrastive_reflection/__init__.py` | Package exports |

### GEPA interfaces we depend on (read-only)

| Interface | What we read | Risk |
|-----------|-------------|------|
| `GEPAState.program_candidates` | List of candidate dicts to extract prompt text snippets | Stable (core state) |
| `GEPAState.evaluation_cache` | Used if available as fallback for train-score lookups | May be None |
| `GEPAState.prog_candidate_val_subscores` | NOT used for contrastive search (wrong ID space), only for diagnostics | N/A |
| `ReflectiveMutationProposer.propose()` | Subclassed; entire method body replicated with injection point added | Version-sensitive: must track upstream changes to this method |
| `InstructionProposalSignature.prompt_renderer()` | Relies on `<side_info>` placeholder being rendered as markdown | Stable |
| `adapter.make_reflective_dataset()` return type | `Mapping[str, Sequence[Mapping[str, Any]]]` -- we append entries to the sequence | Stable |

---

## 3. New MutationConfig Fields

These fields are defined in `ContrastiveReflectionConfig`, not added to the upstream `MutationConfig`:

```python
from pydantic import BaseModel, Field

class ContrastiveReflectionConfig(BaseModel):
    """Configuration for the contrastive_reflection mutation."""

    # Core toggle
    use_contrastive_reflection: bool = Field(
        default=False,
        description="Whether to inject contrastive snippets into reflection prompts."
    )

    # Snippet parameters
    contrastive_snippet_length: int = Field(
        default=300,
        ge=50,
        le=1000,
        description="Maximum character length of contrastive snippet injected into reflection."
    )

    # Search parameters
    failing_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Score threshold below which a train example is considered 'failed'. "
            "Only failed examples trigger contrastive search."
        )
    )

    min_contrastive_score_gap: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum score difference between the contrastive candidate and current candidate "
            "on a failing example for the contrastive candidate to qualify."
        )
    )

    injection_strategy: str = Field(
        default="side_info",
        pattern="^(side_info|template)$",
        description="How to inject the contrastive snippet: 'side_info' appends to reflective dataset, "
                    "'template' uses a custom prompt template with <contrastive> placeholder."
    )

    # Ablation control
    contrastive_source: str = Field(
        default="best_solver",
        pattern="^(best_solver|random_solver|highest_overall)$",
        description=(
            "Which contrastive candidate to select: "
            "'best_solver' = candidate that solved the most failures, "
            "'random_solver' = random candidate that solved at least one failure, "
            "'highest_overall' = candidate with highest overall val score."
        )
    )
```

**Validation rules**:
- `contrastive_snippet_length` must be >= 50 (snippets shorter than this carry no meaningful signal).
- `contrastive_snippet_length` must be <= 1000 (longer snippets risk dominating the reflection prompt).
- If `use_contrastive_reflection` is False, all other fields are ignored (the proposer falls back to vanilla GEPA behavior).
- `injection_strategy="template"` requires a custom `reflection_prompt_template` to be provided; the config validator will raise an error if it is missing.

---

## 4. Parameter Sweep Values

### Primary sweep (2 x 3 = 6 configurations per benchmark)

| Parameter | Values | Justification |
|-----------|--------|---------------|
| `use_contrastive_reflection` | `[True, False]` | The False condition is the vanilla GEPA control; True enables the mutation |
| `contrastive_snippet_length` | `[150, 300, 500]` | 150 = minimal signal (1-2 sentence fragment), 300 = default (full strategy sentence), 500 = extended context. Paper's Qwen3-8B context is 16384 tokens, so even 500 chars (~125 tokens) is <1% of context budget |

### Secondary sweep (run only if primary sweep shows positive signal)

| Parameter | Values | Justification |
|-----------|--------|---------------|
| `failing_threshold` | `[0.3, 0.5, 0.7]` | 0.3 = only clear failures, 0.5 = default binary, 0.7 = includes partial successes as "failures" |
| `min_contrastive_score_gap` | `[0.1, 0.3, 0.5]` | Controls how different the contrastive candidate must be |
| `contrastive_source` | `[best_solver, random_solver, highest_overall]` | Ablation: is the contrastive signal useful because of its content (best_solver), or does any alternative text help (random_solver)? |

### Fixed parameters (match paper defaults)

| Parameter | Value | Source |
|-----------|-------|--------|
| `temperature` | 0.6 | Paper Table 7 |
| `top_p` | 0.95 | Paper Table 7 |
| `top_k` | 20 | Paper Table 7 |
| `minibatch_size` | 3 | Paper Section 4.1 |
| `module_selection` | round_robin | Paper Section 3.2 |
| `candidate_selection_strategy` | pareto | Paper Section 3.3 |

---

## 5. Hypotheses and Success Criteria

### Primary Hypothesis (H1)

**Statement**: Contrastive reflection with snippet_length=300 will improve GEPA's final test-set score by at least +1.5 points (absolute) on multi-hop reasoning benchmarks (HotpotQA, HoVer) compared to vanilla GEPA, at the same rollout budget.

**Rationale**: Multi-hop reasoning tasks have the most diverse failure modes, so contrastive signals from candidates that solve different sub-problems should be most informative. The +1.5 threshold is approximately one standard deviation of inter-seed variance observed in similar GEPA runs.

**Falsification**: If contrastive_reflection=True scores within +/-1.0 of vanilla GEPA across 3 seeds on both HotpotQA and HoVer, H1 is rejected.

### Secondary Hypothesis (H2)

**Statement**: Shorter snippets (150 chars) will underperform the default (300 chars) because they lose strategy context, while longer snippets (500 chars) will perform comparably to 300 chars because the additional text adds redundancy but not new signal.

**Falsification**: If 150-char snippets match or exceed 300-char performance on 2+ benchmarks, H2 is rejected.

### Tertiary Hypothesis (H3)

**Statement**: The contrastive signal's value comes from its content, not merely from adding diversity. The `best_solver` source will outperform `random_solver` by at least +1.0 point.

**Falsification**: If `random_solver` matches `best_solver` within +/-0.5 points, the improvement is due to diversity injection, not contrastive content.

### Per-benchmark success criteria

| Benchmark | Vanilla GEPA baseline | Target (contrastive, 300 chars) | Minimum for "positive signal" |
|-----------|----------------------|--------------------------------|------------------------------|
| HotpotQA  | 62.33 | >= 63.83 (+1.5) | >= 62.83 (+0.5) |
| HoVer     | 52.33 | >= 53.83 (+1.5) | >= 52.83 (+0.5) |
| IFBench   | 38.61 | >= 39.61 (+1.0) | >= 38.61 (+0.0, no regression) |
| PUPA      | 91.85 | >= 92.35 (+0.5) | >= 91.35 (-0.5, no large regression) |

### Efficiency hypothesis (H4)

**Statement**: Contrastive reflection will NOT increase rollout budget by more than 5%, because the contrastive search is a CPU-only lookup (no additional LLM calls). The only additional LLM cost is the extra tokens in the reflection prompt (~300 chars = ~75 tokens per component).

**Falsification**: If rollout count increases by >10% compared to vanilla GEPA, investigate whether the contrastive signal is causing more aggressive exploration.

---

## 6. Test Strategy

### Unit Tests

File: `tests/experiments/contrastive_reflection/test_contrastive_search.py`

1. **test_find_contrastive_no_failures**: All train scores above threshold -> returns empty dict. Verifies early exit.
2. **test_find_contrastive_no_other_candidates**: Only one candidate in pool -> returns empty dict. Verifies graceful handling.
3. **test_find_contrastive_basic_match**: Two candidates, candidate B solved example that candidate A failed -> returns snippet from B. Verifies core logic.
4. **test_find_contrastive_snippet_truncation**: Contrastive candidate prompt is 1000 chars, snippet_length=300 -> returned snippet is exactly 300 chars, truncated at a word boundary.
5. **test_find_contrastive_respects_score_gap**: Contrastive candidate scored only marginally better (gap < min_contrastive_score_gap) -> returns empty dict.
6. **test_contrastive_train_index_accumulation**: Index correctly accumulates scores across multiple iterations.
7. **test_contrastive_train_index_does_not_use_val_ids**: Index stores train IDs, not val IDs. Verifies the critical bug fix.

File: `tests/experiments/contrastive_reflection/test_injection.py`

8. **test_inject_side_info_strategy**: Contrastive snippet is appended as a new entry in the reflective dataset with correct formatting.
9. **test_inject_empty_snippets**: Empty snippets dict -> reflective dataset unchanged.
10. **test_inject_multi_component**: Snippets for two components -> both components' reflective datasets get their respective snippets.
11. **test_snippet_html_escaping**: Snippet containing markdown-special characters is escaped properly.

File: `tests/experiments/contrastive_reflection/test_config.py`

12. **test_config_defaults**: Default config has use_contrastive_reflection=False, snippet_length=300.
13. **test_config_validation_snippet_length_bounds**: snippet_length < 50 or > 1000 raises ValidationError.
14. **test_config_template_strategy_requires_template**: injection_strategy="template" without a template raises error.

### Integration Tests

File: `tests/experiments/contrastive_reflection/test_proposer_integration.py`

15. **test_proposer_vanilla_passthrough**: With use_contrastive_reflection=False, proposer behaves identically to vanilla ReflectiveMutationProposer. Mock adapter, verify same call sequence.
16. **test_proposer_contrastive_injection**: With use_contrastive_reflection=True, verify that propose_new_texts receives a reflective_dataset with the contrastive entry appended.
17. **test_proposer_no_contrastive_when_all_pass**: All minibatch examples pass -> no contrastive search, no injection. Identical to vanilla.
18. **test_proposer_callback_notifications_preserved**: All GEPA callback events (CandidateSelected, MinibatchSampled, etc.) fire in the correct order even with contrastive injection enabled.

### Smoke Tests

File: `scripts/smoke_test_contrastive_reflection.sh`

19. **Smoke test**: Run 3 iterations of contrastive_reflection on HotpotQA with minibatch_size=3, snippet_length=300. Verify:
    - No crashes
    - At least one contrastive snippet was injected (check logs)
    - Rollout count is within expected range
    - Output files (candidates.json, run_log.json) are written correctly
    - Runtime < 10 minutes (for 3 iterations)

---

## 7. Known Bugs to Fix

### Bug #1: Val/train ID mismatch (CRITICAL)

**Description**: The naive contrastive search would look up `state.prog_candidate_val_subscores` using train-set IDs from the minibatch (`subsample_ids`). Train and val sets use different ID spaces (train IDs come from `batch_sampler.next_minibatch_ids()` applied to `self.trainset`; val IDs come from the validation set used during `update_state_with_new_program()`). The lookup will return no matches because no val-set key matches any train-set ID.

**Fix**: Do NOT use `state.prog_candidate_val_subscores` for contrastive search at all. Instead, maintain a separate `ContrastiveTrainIndex` that records `{candidate_idx: {train_id: score}}` from each iteration's train-set evaluation. This index is populated inside `ContrastiveReflectionProposer.propose()` immediately after the `adapter.evaluate()` call returns `eval_curr.scores`:

```python
# After eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
self._contrastive_train_index.record(
    candidate_idx=curr_prog_id,
    train_scores=dict(zip(subsample_ids, eval_curr.scores)),
)
```

The contrastive search then queries this index exclusively. Since different iterations may evaluate different minibatch subsets, the index tracks a sparse mapping -- a contrastive candidate qualifies only if it has been evaluated on at least one of the current candidate's failing examples.

**Verification**: Unit test `test_contrastive_train_index_does_not_use_val_ids` explicitly constructs a state with val IDs that differ from train IDs and confirms that the contrastive search still finds matches via the train index.

### Bug #2: Single-component snippet in multi-component candidates (MODERATE)

**Description**: If the candidate has multiple components (e.g., `{"system_prompt": "...", "user_template": "..."}`), the naive implementation extracts a snippet from only the first component of the contrastive candidate and injects it into all components' reflection prompts. This means the user_template's reflection prompt sees a system_prompt snippet, which may be irrelevant or confusing.

**Fix**: `find_contrastive_candidates()` returns `dict[str, str]` mapping component_name to its respective snippet. `inject_contrastive_snippets()` then injects each component's snippet into only that component's reflective dataset entry:

```python
def find_contrastive_candidates(..., components_to_update: list[str]) -> dict[str, str]:
    # ... find best contrastive candidate ...
    snippets = {}
    for component in components_to_update:
        if component in contrastive_candidate:
            text = contrastive_candidate[component]
            snippets[component] = text[:max_snippet_length]  # truncate at word boundary
    return snippets
```

**Verification**: Unit test `test_inject_multi_component` creates a two-component candidate and verifies each component's reflective dataset gets its own snippet.

### Bug #3: Empty candidate pool early in optimization (LOW)

**Description**: In the first iteration (i=0), the candidate pool contains only the seed candidate. If the seed candidate fails on a minibatch example, the contrastive search will find zero alternative candidates because the seed is the only entry in the pool. The search returns an empty dict and no injection occurs -- this is correct behavior, but the first several iterations may never see contrastive signal if minibatches are small.

**Fix**: This is expected behavior, not a bug. Document it. The contrastive signal is designed to become useful after 3-5 iterations when the candidate pool has accumulated enough diversity. Add a log message when contrastive search returns empty due to insufficient pool size:

```python
if len(self._contrastive_train_index) < 2:
    self.logger.log(f"Iteration {i}: Contrastive search skipped (pool size < 2)")
```

---

## 8. Benchmark Selection

### Tier 1: Run first (development + primary evaluation)

**HotpotQA** -- Run first for these reasons:
1. Multi-hop QA has diverse failure modes (entity linking, reasoning chain, evidence retrieval), making it the best candidate for contrastive signal to help.
2. Paper baseline is 62.33 with 6871 rollouts -- moderate budget, not the most expensive.
3. Well-studied benchmark with stable evaluation metrics (F1-based).
4. Large train set provides sufficient minibatch diversity for contrastive search to find matches.

### Tier 2: Run second (validation of generalization)

**HoVer** -- Second priority:
1. Also multi-hop reasoning (multi-hop fact verification), so H1 predicts improvement here too.
2. Lowest rollout budget (2426), making it the cheapest to run.
3. If contrastive_reflection helps on HotpotQA but not HoVer (or vice versa), that reveals whether the signal is task-specific.

### Tier 3: Run if Tiers 1-2 show positive signal

**IFBench** -- Instruction following is structurally different from multi-hop reasoning. If contrastive reflection helps here too, the mutation is more general than hypothesized.

**PUPA** -- High baseline (91.85) means limited room for improvement, but also a good test for regression.

### Tier 4: Run last (expensive, niche)

**AIME-2025** -- Highest rollout budget (7051) and math reasoning may not benefit from contrastive text snippets (math solutions require exact reasoning steps, not prompt phrasing).

**LiveBench-Math** -- Lowest rollout budget (1839) but also lowest expected benefit for contrastive reflection in math domains.

### Justification for ordering

The ordering prioritizes benchmarks where the mutation is most likely to show an effect (multi-hop reasoning with diverse failure modes) at moderate cost, before spending budget on benchmarks where the expected effect is smaller or the cost is higher.

---

## 9. Seed Strategy

### Number of seeds: 3

Seeds: `[42, 137, 2025]`

**Justification**:
- 3 seeds is the minimum needed to compute a meaningful standard deviation and distinguish signal from noise.
- The paper does not report per-seed variance for Qwen3-8B, so we need at least 3 seeds to estimate it ourselves.
- More seeds would be better statistically but the rollout budget makes >3 expensive (see Section 10).
- Seeds 42 and 137 are conventional; 2025 matches the paper's publication year and ensures we are not accidentally using a "lucky" seed.

### What the seed controls

- Minibatch sampling order (via `BatchSampler`)
- Candidate selection tiebreaking (via `CandidateSelector`)
- LLM generation randomness (temperature=0.6, controlled by the API-level seed if available)

### Seed application

The seed is set via `random.seed(s)` and `numpy.random.seed(s)` at experiment start. The LLM seed (if supported by OpenRouter) is set per-call. Each (seed, config) pair is a separate experiment run with its own `run_dir`.

### Total runs

- Primary sweep: 6 configs x 3 seeds = 18 runs per benchmark
- With 2 Tier-1/2 benchmarks: 36 runs
- Full primary sweep across 4 benchmarks: 72 runs

---

## 10. Rollout Budget

### Per-benchmark rollout budgets

Match the paper's GEPA rollout budgets exactly, so results are directly comparable:

| Benchmark | Paper GEPA Budget | Our Budget | Rationale |
|-----------|------------------|------------|-----------|
| HotpotQA  | 6871 | 6871 | Exact match for direct comparison |
| HoVer     | 2426 | 2426 | Exact match |
| IFBench   | 3593 | 3593 | Exact match |
| PUPA      | 3936 | 3936 | Exact match |
| AIME-2025 | 7051 | 7051 | Exact match |
| LiveBench | 1839 | 1839 | Exact match |

### Additional rollouts from contrastive reflection

The contrastive search itself requires **zero additional LLM evaluations**. It is a CPU-only lookup over the `ContrastiveTrainIndex`. The only additional LLM cost is the extra ~75 tokens per component in the reflection prompt (the contrastive snippet), which increases the reflection LLM call cost marginally but does not count as an additional "rollout" in GEPA's budget accounting.

Therefore, the total rollout budget is identical between the vanilla control and the contrastive_reflection treatment. This is a key advantage of this mutation: it adds information to the reflection step without consuming evaluation budget.

### Budget enforcement

GEPA's `state.total_num_evals` tracks the cumulative evaluation count. The optimization loop terminates when `total_num_evals >= max_metric_calls`. We set `max_metric_calls` to the paper budget for each benchmark.

### Cost estimate (Tier 1 + Tier 2, primary sweep)

Per run (HotpotQA, 6871 rollouts):
- Evaluation LLM calls: ~6871 (Qwen3-8B via OpenRouter)
- Reflection LLM calls: ~6871 / 3 (minibatch_size) * 1 (one reflection per iteration) = ~2290
- Total LLM calls per run: ~9161
- At ~$0.20/M input tokens + $0.60/M output tokens (OpenRouter Qwen3-8B): ~$0.50-1.00 per run

Total for Tier 1 + Tier 2 primary sweep (36 runs):
- HotpotQA: 18 runs x ~$0.75 = ~$13.50
- HoVer: 18 runs x ~$0.26 (2426/6871 ratio) = ~$4.75
- **Estimated total: ~$18.25**

### Early stopping

If a run's score plateaus for 20 consecutive iterations (no improvement in `program_full_scores_val_set`), the run is stopped early and the budget savings are noted. This matches GEPA's default behavior and does not need special handling.
