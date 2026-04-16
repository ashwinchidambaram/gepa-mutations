# Experiment 04 — Pilot: GEPA vs Slime Mold (HotpotQA, Qwen3-8B)

**Status:** PENDING (pre-registered, awaiting pod provisioning)
**Date:** 2026-04-16
**Purpose:** Establish a head-to-head reference baseline for GEPA vs the personality
variant of Slime Mold on a single benchmark + model. This number becomes the
forward-looking baseline that all future inductive-discovery variants are
measured against.

---

## What we're testing

Two prompt-optimization algorithms running on the **same seed prompt** with the
**same rollout budget** on the **same model**:

- **`gepa`** — reflective Pareto-frontier prompt search (Agrawal et al. 2025).
  Wraps `gepa.api.optimize` with the `DefaultAdapter` for single-prompt
  optimization. Already wired in `src/gepa_mutations/runner/experiment.py`.
- **`slime_mold`** — personality-strategy variant (Analytical/Creative/Minimal/
  Expert), the most-studied reference point from prior experiments. This is
  the default `--strategy-mode personality --mutation-mode blind --refresh-mode
  none` configuration of `methods/slime_mold/slime_mold/runner.py`.

We are **not** including the inductive variants in this pilot. Those are still
being prompt-engineered and would muddy a head-to-head against GEPA.

## Configuration

| Setting | Value |
|---|---|
| Benchmark | HotpotQA |
| Model (task LM) | Qwen3-8B bf16 (vLLM on A40) |
| Model (reflection LM) | Qwen3-8B bf16 (same) — flagged as deviation from GEPA paper, which uses GPT-5 |
| Seed prompt | `"You are a helpful assistant."` (matches `gepa/tests/test_pareto_frontier_types/...:98`) |
| Rollout budget | 6,871 (paper budget for HotpotQA from `PAPER_ROLLOUTS["gepa"]` — Table 1 of GEPA paper) |
| Seeds | 555, 999, 1337 (deliberately outside the prior sweep range of 42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768 to prevent any tuning leak) |
| Total runs | 6 (3 seeds × 2 methods) |
| Hardware | RunPod A40 (single GPU) |
| Estimated cost | $10–15 (~$0.47/hr × ~3hr per run × 6 runs, sequential within a single pod) |

## Fairness conditions

1. **Same seed prompt:** both methods read `BENCHMARK_SEED_PROMPTS["hotpotqa"]`,
   which is the GEPA-canonical minimal seed.
2. **Same rollout budget:** both methods default to
   `PAPER_ROLLOUTS["gepa"]["hotpotqa"] = 6871` when `--max-metric-calls` is not
   passed.
3. **Same model and task LM config:** both use the vLLM endpoint at
   `$GEPA_BASE_URL` with paper hyperparameters (`temp=0.6, top_p=0.95, top_k=20`).
4. **Same reflection LM:** Qwen3-8B for both. We are NOT giving GEPA a stronger
   reflection LM. This is a known deviation from the GEPA paper (which uses
   GPT-5 as reflection LM) and is the same handicap Slime Mold runs under.
5. **Same seeds:** 555, 999, 1337 across both methods. Deterministic on the
   same seed, so head-to-head comparisons are paired.

## Pre-registered decision criteria

These are written **before** results are in. We commit to interpreting the data
according to this matrix, regardless of how the pilot turns out, to avoid
post-hoc rationalization.

Let `Δ = mean(Slime Mold test) − mean(GEPA test)` (3 seeds each).

| Outcome | Δ | Decision |
|---|---|---|
| **Slime Mold clearly wins** | Δ ≥ +5% absolute, with comparable or lower std | Proceed to full sweep across 4 benchmarks × 5 seeds × 2 methods. Headline framing: "Slime Mold beats GEPA on single-prompt optimization at matched budget." |
| **Comparable / cost advantage** | +2% ≤ Δ < +5% | Proceed to full sweep but **temper claims**. Headline: "comparable performance with cost advantage" — NOT "beats." |
| **Inconclusive / GEPA wins** | Δ < +2% (including any negative Δ) | Do **NOT** claim we beat GEPA. Reframe experiment 03 as a pure Slime-Mold ablation study (inductive vs personality vs prescribed8). |
| **GEPA fails to converge** | GEPA's mean test score ≤ seed test score, or its trajectory shows no improvement across rollouts | This is a finding in itself — GEPA's single-prompt baseline behavior on Qwen3-8B is weak (the paper never tested this). Report Slime Mold standalone. Optionally re-run with a stronger reflection LM as a follow-up if budget allows. |

**Variance check:** if either method's std exceeds 0.05 (5 absolute points)
across the 3 seeds, flag the result as too noisy to draw conclusions and
recommend adding 2 more seeds before deciding.

## Open questions to surface during execution (do NOT silently resolve)

If during the pilot you observe any of the following, **stop and report**:

- GEPA's `DefaultAdapter` produces prompts in a format our eval harness can't
  evaluate
- Either method's actual rollout consumption diverges from its `max_metric_calls`
  budget by >10%, breaking the matched-budget premise
- Qwen3-8B as reflection LM produces obviously degenerate output (empty
  reflections, `<think>` artifacts, repeated text). If so, both methods are
  affected and the comparison is still fair, but the absolute numbers should
  not be treated as canonical
- Any individual run takes >4 wall-clock hours, indicating a stall (the
  2-hour-per-run wall-clock timeout from Phase 6 should kick in first)
- Schema validation fails on any of the 6 result.json files

## Hard pause point

After the 6 runs complete and `analyze.py` produces the summary, **STOP**.
Do not proceed to the full sweep without an explicit go from the user, even if
the pilot results look favorable. The pilot's job is to inform that decision,
not to make it.

## How this becomes a forward-looking baseline

The 6 result directories produced by this pilot are the **canonical
single-prompt Qwen3-8B baseline numbers** for HotpotQA going forward. Any
future experiment that claims an improvement on HotpotQA should compare
against the GEPA + Slime Mold means recorded here, not against earlier numbers
from experiment 02 (which used the old task-descriptive seed prompts and is
therefore not on the same comparison surface).

Specifically:
- `gepa` mean test score on the 3 pilot seeds = the forward GEPA reference
- `slime_mold` mean test score on the 3 pilot seeds = the forward Slime Mold
  reference

These are recorded in `analyze_summary.json` after the pilot completes.

## Reproduction command

After pod is provisioned with vLLM serving Qwen3-8B on port 8125:

```bash
# From repo root on the pod
.venv/bin/python scripts/run_all_local.py \
    --workers 2 \
    --seeds 555,999,1337 \
    --method gepa slime_mold \
    --benchmark hotpotqa \
    --runs-dir experiments/04-pilot-gepa-comparison-2026-04/runs
```

`--workers 2` keeps both methods sharing the single A40 without OOM. Sequential
seeds within each method, both methods racing.

## Analysis

After all 6 runs complete:

```bash
.venv/bin/python experiments/04-pilot-gepa-comparison-2026-04/analyze.py \
    --runs-dir experiments/04-pilot-gepa-comparison-2026-04/runs/qwen3-8b/hotpotqa
```

Produces:
- `analyze_summary.json` — machine-readable means, stds, best-of-3, gap
- `final_prompts.md` — qualitative dump of all 6 final prompts for inspection
- `convergence.png` — rollouts-vs-best-score curve per method
- stdout summary in the format pre-registered in this README

## Files in this directory

- `README.md` — this file (pre-registered decision criteria)
- `runs/` — the 6 result directories (populated during pilot)
- `plots/` — analysis outputs
- `logs/` — orchestrator + vLLM logs
- `analyze.py` — analysis script
- `analyze_summary.json` — generated by analyze.py
- `final_prompts.md` — generated by analyze.py
