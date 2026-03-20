# Smoke Test Issues Log

Tracking issues encountered during benchmark smoke tests (subset=5, 50 rollouts).

---

## Issue 1: Test set evaluation is the bottleneck (not optimization)

**Observed**: AIME test eval (150 dspy ChainOfThought calls) took ~3+ hours. HoVer test eval (300 direct LLM calls) is taking 1+ hour. The optimization phase itself finishes in 15-30 minutes for QA benchmarks.

**Root cause**: `_evaluate_on_test()` in `experiment.py` evaluates examples sequentially — one LLM call at a time. No parallelization.

**Impact**: For full experiments with 300+ test examples, test evaluation will dominate wall-clock time even though it's just one pass with no iteration.

**Proposed fix**: Parallelize test evaluation using `concurrent.futures.ThreadPoolExecutor` or `litellm.batch_completion`. For dspy benchmarks (AIME, LiveBench), use `dspy.Evaluate` with `num_threads` parameter. Low priority for smoke tests but important for full runs.

**Needs user permission?**: No — code change only.

---

## Issue 2: PUPA and LiveBench dataset identifiers were wrong (FIXED)

**Observed**: PUPA tried `liyucheng/pupa`, LiveBench tried `livebench/livebench`. Both 404'd on HuggingFace.

**Fix applied**: PUPA → `Columbia-NLP/PUPA` config `pupa_tnb`. LiveBench → `livebench/math`, question in `turns[0]`, answer in `ground_truth`. Committed in `0cfccb3`.

---

## Issue 3: PUPA smoke test score extremely low (4.21% vs paper 91.85%)

**Observed**: PUPA scored 4.21% on the smoke test. Paper GEPA scores 91.85%.

**Root cause**: PUPA is a privacy-preserving query redaction task (`user_query` → `redacted_query`). The QA adapter's `_score()` method uses simple answer containment (`answer.lower() in response.lower()`), which is wrong for this task. A redacted query won't be "contained" in the model's response — it needs a different scoring approach (e.g., checking that PII was properly redacted, or comparing the structure of the redaction).

**Impact**: PUPA results are invalid until the evaluator is fixed. Other benchmarks unaffected.

**Proposed fix**: Implement a PUPA-specific adapter that scores based on PII redaction quality. Need to research how the GEPA paper evaluates PUPA — check `gepa/tests/test_pareto_frontier_types/` for clues.

**Needs user permission?**: No — code change only. But need to research the correct scoring method first.

---

## Issue 4: No progress logging during test set evaluation

**Observed**: Test evaluation runs silently for hours (150-300 examples). No per-example logging. Process appears stuck but is actually progressing.

**Root cause**: `_evaluate_on_test()` and `_evaluate_dspy()` / `_evaluate_qa()` in `experiment.py` have no progress output. Each example is evaluated sequentially with no indication of progress.

**Impact**: Impossible to distinguish a stuck process from a slow one. Causes unnecessary concern during monitoring.

**Proposed fix**: Add progress logging every 10 examples in `_evaluate_dspy()` and `_evaluate_qa()`:
```python
if (i + 1) % 10 == 0:
    console.print(f"  Test eval: {i+1}/{len(testset)} ({correct}/{i+1} correct)")
```

**Needs user permission?**: No — code change only. **FIXED** — added every-10-examples logging in experiment.py.

---

## Issue 5: dspy ChainOfThought calls hang indefinitely (no timeout)

**Observed**: LiveBench smoke test hung at example ~120/148 twice (first run hung for 3+ hours, retry also hung at same point). Process at 0% CPU, no active TCP connections, 16MB RSS.

**Root cause**: `dspy.ChainOfThought` calls to OpenRouter via litellm have no timeout configured. If a single API call hangs, the entire test evaluation blocks forever.

**Impact**: BLOCKER for unattended runs. Any dspy benchmark (AIME, LiveBench) can hang.

**Proposed fix**:
1. Add `timeout` param to dspy LM config: `dspy.LM(..., timeout=120)`
2. Add `timeout` to custom LM litellm calls: `litellm.completion(..., timeout=120)`
3. Wrap individual test eval calls in try/except with per-call timeout

**Needs user permission?**: No — code change only. Should fix before more runs.

---

## Issue 6: LiveBench evaluator uses wrong scoring (AIME int-parse for non-integer answers)

**Observed**: LiveBench test eval: 87+ errors out of 100 examples. Error: `invalid literal for int() with base 10: '-\frac{16x}{3}'`, `'B'`, `'3,14,2,7,...'`.

**Root cause**: LiveBench mapped to `AIMEAdapter` in `evaluators.py:get_adapter()` which uses `_math_metric()` → `int(prediction.answer)`. LiveBench-Math answers include LaTeX, letters, comma-separated sequences — not simple integers.

**Impact**: LiveBench scores meaningless (~3% due to parse failures). Other benchmarks unaffected.

**Proposed fix**: Create LiveBench-specific evaluator with string comparison instead of int parsing. Research paper's LiveBench scoring method.

**Needs user permission?**: No — code change only.

---

## Issue 7: Contrastive reflection finds 0 pairs on small subsets

**Observed**: During smoke tests (subset=5), contrastive_reflection found 0 contrastive pairs on most iterations for most benchmarks. Only PUPA generated meaningful pairs.

**Root cause**: With subset=5, the seed prompt often scores perfectly on all 5 training examples, so there are no failures to contrast against. The `min_score_gap=0.1` threshold further filters out weak contrasts.

**Impact**: The contrastive mutation effectively degrades to vanilla GEPA on small subsets. Full runs with larger subsets (≥20) needed for meaningful comparison.

**Proposed fix**: None — expected behavior. Document that contrastive_reflection requires larger subsets.

---

## Issue 8: AIME smoke tests show DSPy JSON mode / thinking mode incompatibility

**Observed**: AIME runs produce warnings about `enable_thinking` + `JSON mode` incompatibility on OpenRouter (Alibaba backend). 8-14 test eval errors per run from structured output format issues.

**Root cause**: Qwen3-8B on OpenRouter has a `thinking` mode that conflicts with dspy's JSON-structured output requests. Framework recovers gracefully (scores 0 for failed examples).

**Impact**: Reduces effective test set size by ~5-10%. Not a blocker but inflates error rates.

**Proposed fix**: Consider adding `extra_body={"enable_thinking": false}` to the dspy LM config.

---

## Issue 9: Missing pupa/gepa and livebench/gepa baseline results

**Observed**: After clearing stale old-format results (2026-03-20), gepa baseline results for PUPA and LiveBench were not re-run. Mutation results exist but have no baseline.

**Root cause**: Stale results deleted during cleanup; mutation smoke test agents only ran mutations, not baselines.

**Proposed fix**: Re-run `uv run gepa-mutations run pupa --subset 5 --seed 42 --max-metric-calls 50` and same for livebench.

---

## Issue 10: BestOfKMetricsCallback required direct wiring fix

**Observed**: `BestOfKMetricsCallback.on_candidate_accepted()` was dead code — `CandidateAcceptedEvent` doesn't carry proposal metadata.

**Fix applied**: Added `record_iteration()` method called directly by `BestOfKProposer`. Committed in `632d87e`.

---

## Issue 11: IFBench evaluator is broken — substring containment, not structural compliance (CRITICAL)

**Observed**: IFBench scores 100% (300/300) across all methods. Paper GEPA gets 38.61%.

**Root cause**: `IFBenchAdapter._score()` in `evaluators.py` checks `constraint.lower() in response.lower()`. IFBench constraints are structural requirements ("at least 3 paragraphs", "include the word 'ocean'"), not literal substrings. Almost any response contains short constraint words as substrings.

**Impact**: IFBench results are meaningless. BLOCKER for full IFBench experiments.

**Proposed fix**: Implement proper IFEval-style programmatic constraint checking (paragraph counting, word presence, format verification).

---

## Issue 12: HoVer evaluator is broken — substring match on "supported" in reasoning (CRITICAL)

**Observed**: HoVer scores 95.67% across all methods. Paper GEPA gets 52.33%.

**Root cause**: `HoVerAdapter._score()` checks `expected.lower() in response.lower()`. Labels are "supported" / "not_supported". Since the model is prompted to reason about whether claims are SUPPORTED or NOT_SUPPORTED, the word "supported" appears in almost every response's reasoning chain — even when arguing for NOT_SUPPORTED.

**Impact**: HoVer results are meaningless. BLOCKER for full HoVer experiments.

**Proposed fix**: Extract the verdict label from the response (last occurrence of SUPPORTED/NOT_SUPPORTED, or structured output) rather than substring matching the full reasoning chain.

---

## Issue 13: AIME test set has duplicated questions — 30 questions tiled 5x = 150 (CRITICAL)

**Observed**: AIME test scores show a perfectly repeating pattern with period 30 across all 150 test examples. Scores at positions 30-59 are identical to 60-89, 90-119, and 120-149.

**Root cause**: The AIME test set loader likely repeats the 30 AIME-2025 problems 5 times (matching the paper's "5 runs per question" protocol, but implemented as dataset duplication rather than repeated evaluation).

**Impact**: Inflates apparent N (150 vs actual 30 unique questions). AIME scores biased and confidence intervals invalid. Paper reports 32% on the same 30 questions.

**Proposed fix**: Verify AIME dataset loader. If the 5x repetition is intentional (per-question variance), aggregate scores should average across the 5 runs per question, not treat them as independent examples.

---

## Issue 14: Contrastive reflection catastrophic regression on AIME (-20.7pp)

**Observed**: contrastive_reflection scored 46.67% on AIME vs baseline 67.33% (-20.7pp). The evolved prompt is identical to the seed prompt on multiple benchmarks, suggesting the contrastive mechanism is not producing useful mutations.

**Root cause**: With subset=5, the contrastive index rarely accumulates enough data to find meaningful contrastive pairs (min_score_gap=0.1). On most iterations, 0 contrastive pairs are found, and the mutation degrades to vanilla GEPA. The AIME regression may be stochastic eval noise on the seed prompt, not caused by contrastive injection.

**Impact**: Contrastive reflection needs larger subsets (≥20) to be meaningful. AIME result should be monitored on full runs.

**Proposed fix**: No code fix needed. Monitor AIME closely in first full-scale seed. If >10pp below baseline after full run, investigate.

---

