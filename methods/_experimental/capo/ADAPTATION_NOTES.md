# CAPO Adaptation Notes

This document explains how CAPO was adapted to run on our benchmark system, what changed from the
original implementation, and why the core algorithm is still faithful to the paper.

**Original paper**: [CAPO: Cost-Aware Prompt Optimization (arXiv 2504.16005)](https://arxiv.org/abs/2504.16005)  
**Original code**: https://github.com/finitearth/capo (archived) / https://github.com/finitearth/promptolution (active fork)

---

## What CAPO Is

CAPO is an evolutionary prompt optimizer with three core ideas:

1. **Population-based search**: Maintains a population of candidate prompts, evolving them via
   LLM-driven crossover (merge two parents) and mutation (rephrase one parent).

2. **Statistical racing**: Rather than evaluating every candidate on the full training set, CAPO
   evaluates in blocks and eliminates underperforming candidates early using a paired t-test
   (α=0.2). The paper reports ~44% average reduction in evaluations with negligible quality loss.

3. **Length penalty**: Scores are adjusted as `adjusted = accuracy - γ·rel_token_length` (γ=0.05)
   to discourage prompt bloat, since longer prompts cost more at inference time.

---

## What We Changed and Why

### 1. Evaluation interface (necessary)

**Original**: CAPO uses its own `Predictor` and `DataProcessor` classes to wrap HuggingFace
datasets and call models via a custom LLM abstraction.

**Ours**: We call `adapter.evaluate(examples, candidate)` from our existing
`gepa_mutations.benchmarks.evaluators` layer, which already handles all five of our benchmarks
(HotpotQA, IFBench, HoVer, PUPA, AIME) with their respective scoring logic.

**Why it still matches**: Racing and selection operate on scores returned by the evaluator. As
long as scores are in [0, 1] per-example (which ours are), the statistical racing logic is
identical. We don't change when or how candidates are eliminated — only what calls the LLM.

### 2. No few-shot example optimization (deliberate simplification)

**Original**: CAPO co-optimizes instruction text *and* few-shot examples in the prompt. The
crossover operator merges both the instruction and the pool of demonstrations from two parents.
Few-shot examples account for ~66% of prompt length in the paper's experiments, so the length
penalty primarily targets them.

**Ours**: Our benchmarks are zero-shot — the prompt is a system instruction only, with no
in-context demonstrations. We optimize the instruction text only. The length penalty still
applies to the instruction itself.

**Why it still matches**: The evolutionary operators (crossover, mutation) and the racing
elimination logic are unchanged. Removing few-shot optimization simplifies CAPO to its
instruction-only variant, which is still a valid configuration. The length penalty is less
impactful without demonstrations but is retained for consistency with the paper.

### 3. Meta-LLM is our existing `reflection_lm` (consistent with other mutations)

**Original**: CAPO uses a configurable `meta_llm` separate from the task LLM. In experiments
this is typically the same model family as the task LLM.

**Ours**: We pass `build_reflection_lm(settings)` as the meta-LLM, which is the same model used
for mutation/reflection across all other methods in this project. This keeps the comparison
controlled — all methods get the same reflection budget from the same model.

### 4. Token tracking via `TrackedLM` (additive, no algorithmic change)

**Original**: CAPO reports cost in dollars and input token counts per experiment but does not
expose per-call token tracking at the optimization loop level.

**Ours**: We wrap both the task LM and reflection LM with `TrackedLM` from
`gepa_mutations.metrics.collector`, which counts tokens per call and accumulates them into
`MetricsCollector`. This is purely additive instrumentation — the underlying LM calls are
identical.

### 5. Result storage via `save_result()` (additive)

**Original**: CAPO writes results to its own directory structure.

**Ours**: Results are written to `runs/<model_tag>/<benchmark>/capo/<seed>/` via
`save_result()`, consistent with all other methods. Fields in `result.json` and `metrics.json`
match the unified schema defined in `src/gepa_mutations/metrics/collector.py`.

---

## Hyperparameters

All core hyperparameters are taken directly from the paper defaults:

| Parameter | Value | Source |
|---|---|---|
| `population_size` | 10 | Paper §4.1 |
| `racing_block_size` | 30 | Paper §4.2 |
| `racing_alpha` | 0.2 | Paper §4.2 |
| `length_penalty_gamma` | 0.05 | Paper §3.3 |
| Rollout budget | 5000 | Matched to GEPA paper budget for fair comparison |

---

## What Is Not Reproduced

- **Few-shot optimization**: As described above — not applicable to our zero-shot benchmarks.
- **CAPO's original benchmark suite** (SST-5, AG News, Subj, COPA): We test exclusively on our
  five benchmarks. This is intentional — we are not reproducing CAPO's paper results, we are
  evaluating CAPO as a baseline on our task distribution.
- **Multi-model racing**: The paper tests on Llama-3.3-70B, Qwen2.5-32B, and Mistral-Small-24B.
  We test on Qwen3 1.7B–27B as served by our vLLM cluster.

---

## Claim: Core Algorithm Is Faithful

The two algorithmic contributions of CAPO that differentiate it from vanilla evolutionary search
are **statistical racing** and **length-penalized selection**. Both are implemented exactly as
described in the paper:

- Racing uses a paired t-test with the paper's α=0.2 threshold, block size 30, and eliminates
  any candidate for which significantly more than one competitor is better.
- Length penalty uses γ=0.05 applied to relative token length vs. the seed prompt.

The evolutionary operators (crossover = LLM-merge two parents, mutation = LLM-rephrase one
parent) follow the paper's description. Random parent selection (not fitness-proportional) is
used, as specified in §3.2.
