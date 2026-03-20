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

