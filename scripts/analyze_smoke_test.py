#!/usr/bin/env python3
"""Analyze smoke test results for a given model and run.

Checks three dimensions:
  1. Operational: did the run complete with valid outputs?
  2. Data collection: are all required metric fields present and populated?
  3. GEPA methodology parity: is the config consistent with paper settings?

Usage:
    python scripts/analyze_smoke_test.py --model-tag qwen3-8b
    python scripts/analyze_smoke_test.py --model-tag qwen3-8b --benchmark hotpotqa --method active_minibatch --seed 42
    python scripts/analyze_smoke_test.py --model-tag qwen3-8b --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# GEPA paper methodology reference values (for parity checks)
# ---------------------------------------------------------------------------

GEPA_SEED_PROMPTS = {
    "hotpotqa": (
        "Answer the question by reasoning step by step through the provided context. "
        "Give a concise, factual answer."
    ),
    "ifbench": (
        "Follow the instructions carefully, ensuring you satisfy all the given constraints. "
        "Provide a complete response that meets every requirement."
    ),
    "hover": (
        "Determine whether the given claim is SUPPORTED or NOT_SUPPORTED based on the "
        "provided evidence. Reason through the evidence step by step."
    ),
    "pupa": (
        "Rewrite the user's query to remove all personally identifiable information (PII) "
        "such as names, addresses, phone numbers, and emails. Replace PII with generic "
        "placeholders while preserving the query's meaning and intent."
    ),
    "livebench": (
        "Solve the math problem step by step. Provide the final answer in the exact "
        "format requested (number, expression, comma-separated list, etc.)."
    ),
}

GEPA_PAPER_TEMPERATURE = 0.6
GEPA_PAPER_TOP_P = 0.95
GEPA_PAPER_TOP_K = 20

# Expected score range for any working prompt (outside = malfunction)
SCORE_MIN = 0.0   # 0.0 exactly = model outputting nothing or always wrong
SCORE_MAX = 1.0
SCORE_SUSPICIOUSLY_LOW = 0.02   # Below this on hotpotqa/hover = model likely broken

# Reasonable tokens-per-rollout range
TOKENS_PER_ROLLOUT_MIN = 100
TOKENS_PER_ROLLOUT_MAX = 3000

REQUIRED_METRIC_FIELDS = [
    "rollout_count",
    "reflection_call_count",
    "task_input_tokens",
    "task_output_tokens",
    "total_tokens",
    "wall_clock_seconds",
    "model",
    "model_tag",
    "benchmark",
    "seed",
    "method",
]

REQUIRED_RESULT_FIELDS = [
    "test_score",
    "val_score",
    "best_prompt",
    "rollout_count",
    "wall_clock_seconds",
    "benchmark",
    "method",
    "seed",
]


# ---------------------------------------------------------------------------
# Check runners
# ---------------------------------------------------------------------------

def check_operational(result: dict, config: dict, verbose: bool) -> list[str]:
    """Check that the run completed and produced valid outputs."""
    issues = []

    # Required result fields
    for f in REQUIRED_RESULT_FIELDS:
        if f not in result:
            issues.append(f"result.json missing field: {f!r}")

    test_score = result.get("test_score", None)
    val_score = result.get("val_score", None)

    if test_score is None:
        issues.append("test_score is None")
    elif test_score == 0.0:
        issues.append(f"test_score is exactly 0.0 — model may be non-functional or all answers wrong")
    elif test_score < SCORE_SUSPICIOUSLY_LOW:
        issues.append(f"test_score={test_score:.4f} is suspiciously low (< {SCORE_SUSPICIOUSLY_LOW}) — check model output quality")

    if val_score is None:
        issues.append("val_score is None")

    best_prompt = result.get("best_prompt", {})
    if not best_prompt:
        issues.append("best_prompt is empty")
    elif isinstance(best_prompt, dict):
        sp = best_prompt.get("system_prompt", "")
        if not sp or len(sp) < 10:
            issues.append(f"best_prompt.system_prompt too short ({len(sp)} chars) — optimization may have failed")

    rollout_count = result.get("rollout_count", 0)
    if rollout_count == 0:
        issues.append("rollout_count is 0 — no rollouts executed")
    elif rollout_count < 5:
        issues.append(f"rollout_count={rollout_count} very low — budget may have been exhausted immediately")

    if verbose and not issues:
        print(f"    test_score={test_score:.4f}, val_score={val_score:.4f}, rollouts={rollout_count}")
        sp = (best_prompt or {}).get("system_prompt", "")
        print(f"    best_prompt preview: {sp[:120]!r}...")

    return issues


def check_data_collection(metrics: dict, result: dict, verbose: bool) -> list[str]:
    """Check that all required metric fields are present and populated."""
    issues = []

    # Required fields
    for f in REQUIRED_METRIC_FIELDS:
        val = metrics.get(f)
        if val is None:
            issues.append(f"metrics.json missing field: {f!r}")
        elif val == "" or (isinstance(val, (int, float)) and val == 0 and f not in
                           ["reflection_call_count", "reflection_input_tokens",
                            "reflection_output_tokens", "seed"]):
            # reflection fields can legitimately be 0 for non-reflective methods
            if f in ["task_input_tokens", "task_output_tokens", "total_tokens",
                     "rollout_count", "wall_clock_seconds"]:
                issues.append(f"metrics.json field {f!r} is 0 — token tracking may not be working")

    # Token sanity check
    rollout_count = metrics.get("rollout_count") or result.get("rollout_count", 0)
    total_tokens = metrics.get("total_tokens", 0)
    if rollout_count > 0 and total_tokens > 0:
        tok_per_rollout = total_tokens / rollout_count
        if tok_per_rollout < TOKENS_PER_ROLLOUT_MIN:
            issues.append(
                f"tokens/rollout={tok_per_rollout:.0f} suspiciously low "
                f"(total={total_tokens}, rollouts={rollout_count}) — responses may be truncated or empty"
            )
        elif tok_per_rollout > TOKENS_PER_ROLLOUT_MAX:
            issues.append(
                f"tokens/rollout={tok_per_rollout:.0f} suspiciously high "
                f"(total={total_tokens}, rollouts={rollout_count}) — thinking mode may be ON or max_tokens too high"
            )
    elif rollout_count > 0 and total_tokens == 0:
        issues.append("total_tokens=0 with rollout_count>0 — TrackedLM is not capturing tokens")

    # Check model and model_tag
    model = metrics.get("model", "")
    model_tag = metrics.get("model_tag", "")
    if not model:
        issues.append("metrics.json 'model' field is empty — model ID not recorded")
    if not model_tag:
        issues.append("metrics.json 'model_tag' field is empty — model_tag not recorded")

    # Check tokens are split into input/output (not all in one bucket)
    inp = metrics.get("task_input_tokens", 0)
    out = metrics.get("task_output_tokens", 0)
    if inp + out > 0 and out == 0:
        issues.append("task_output_tokens=0 — completion tokens not captured (LM._last_usage may be None)")
    if inp + out > 0 and inp == 0:
        issues.append("task_input_tokens=0 — prompt tokens not captured")

    if verbose and not issues:
        print(f"    total_tokens={total_tokens}, tok/rollout={total_tokens/max(rollout_count,1):.0f}")
        print(f"    task_in={inp}, task_out={out}")
        print(f"    model={model!r}, model_tag={model_tag!r}")

    return issues


def check_gepa_parity(result: dict, config: dict, metrics: dict, verbose: bool) -> list[str]:
    """Check that the run is in parity with GEPA paper methodology."""
    issues = []
    benchmark = result.get("benchmark", config.get("benchmark", ""))

    # 1. Temperature and sampling params in config
    temp = config.get("temperature", config.get("gepa_temperature"))
    if temp is not None and abs(temp - GEPA_PAPER_TEMPERATURE) > 0.01:
        issues.append(f"temperature={temp} != paper default {GEPA_PAPER_TEMPERATURE}")

    top_p = config.get("top_p", config.get("gepa_top_p"))
    if top_p is not None and abs(top_p - GEPA_PAPER_TOP_P) > 0.01:
        issues.append(f"top_p={top_p} != paper default {GEPA_PAPER_TOP_P}")

    # 2. Seed prompt — the optimized prompt should not be identical to the seed prompt
    # (if it is, optimization didn't do anything), but it should START from a reasonable place
    best_prompt = result.get("best_prompt", {})
    if isinstance(best_prompt, dict):
        sp = best_prompt.get("system_prompt", "")
        seed_prompt = GEPA_SEED_PROMPTS.get(benchmark, "")
        rollout_count = result.get("rollout_count", metrics.get("rollout_count", 0))
        if seed_prompt and sp == seed_prompt:
            if rollout_count <= 30:
                # Expected for smoke tests with very few rollouts — not a blocking issue
                print("    note: best_prompt unchanged from seed (expected with ≤20 rollouts)")
            else:
                issues.append(
                    "best_prompt is identical to seed prompt — optimization made no changes"
                )

    # 3. Check model field is recorded
    model = config.get("model", metrics.get("model", ""))
    if not model:
        issues.append("model ID not recorded in config or metrics")

    # 4. Check method is recorded correctly
    method_in_result = result.get("method", "")
    method_in_metrics = metrics.get("method", "")
    if method_in_metrics and method_in_result and method_in_result != method_in_metrics:
        issues.append(
            f"method mismatch: result.json says {method_in_result!r}, "
            f"metrics.json says {method_in_metrics!r}"
        )

    # 5. Benchmark-specific checks
    if benchmark == "hotpotqa":
        test_score = result.get("test_score", 0)
        if isinstance(test_score, (int, float)) and 0 < test_score < SCORE_SUSPICIOUSLY_LOW:
            issues.append(
                f"hotpotqa test_score={test_score:.4f} is extremely low — "
                "verify model can do multi-hop reasoning at all"
            )

    if verbose and not issues:
        print(f"    temperature={temp}, top_p={top_p}")
        print(f"    model={model!r}, benchmark={benchmark!r}")
        print(f"    method={method_in_result!r}")

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_run(
    run_dir: Path,
    verbose: bool = False,
) -> bool:
    """Analyze a single run directory. Returns True if all checks pass."""
    result_path = run_dir / "result.json"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"

    if not result_path.exists():
        print(f"  [SKIP] No result.json at {run_dir}")
        return False

    result = json.loads(result_path.read_text())
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    all_issues: list[str] = []

    print(f"\n  [1] Operational checks...")
    op_issues = check_operational(result, config, verbose)
    all_issues.extend(op_issues)
    for iss in op_issues:
        print(f"      ⚠  {iss}")
    if not op_issues:
        print(f"      ✓  PASS")

    print(f"  [2] Data collection checks...")
    dc_issues = check_data_collection(metrics, result, verbose)
    all_issues.extend(dc_issues)
    for iss in dc_issues:
        print(f"      ⚠  {iss}")
    if not dc_issues:
        print(f"      ✓  PASS")

    print(f"  [3] GEPA methodology parity checks...")
    gp_issues = check_gepa_parity(result, config, metrics, verbose)
    all_issues.extend(gp_issues)
    for iss in gp_issues:
        print(f"      ⚠  {iss}")
    if not gp_issues:
        print(f"      ✓  PASS")

    return len(all_issues) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze smoke test results")
    parser.add_argument("--model-tag", required=True, help="Model tag (e.g. qwen3-8b)")
    parser.add_argument("--benchmark", default="hotpotqa", help="Benchmark (default: hotpotqa)")
    parser.add_argument("--method", default="active_minibatch", help="Method (default: active_minibatch)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument("--all-runs", action="store_true",
                        help="Analyze all runs under runs/{model_tag}/ instead of one specific run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print passing values too")
    args = parser.parse_args()

    runs_root = Path("runs") / args.model_tag
    if not runs_root.exists():
        print(f"No runs found at {runs_root}")
        sys.exit(1)

    total = 0
    passed = 0

    if args.all_runs:
        # Walk all completed runs under this model tag
        for bench_dir in sorted(runs_root.iterdir()):
            if not bench_dir.is_dir():
                continue
            for method_dir in sorted(bench_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                for seed_dir in sorted(method_dir.iterdir()):
                    if not seed_dir.is_dir() or not (seed_dir / "result.json").exists():
                        continue
                    label = f"{bench_dir.name}/{method_dir.name}/seed={seed_dir.name}"
                    print(f"\n{'─'*60}")
                    print(f"Analyzing: {label}")
                    ok = analyze_run(seed_dir, verbose=args.verbose)
                    total += 1
                    if ok:
                        passed += 1
    else:
        run_dir = runs_root / args.benchmark / args.method / str(args.seed)
        print(f"Analyzing: {args.model_tag}/{args.benchmark}/{args.method}/seed={args.seed}")
        ok = analyze_run(run_dir, verbose=args.verbose)
        total = 1
        passed = 1 if ok else 0

    print(f"\n{'='*60}")
    if passed == total:
        print(f"✅  ALL CHECKS PASSED ({passed}/{total} runs)")
    else:
        print(f"❌  {total - passed}/{total} runs have issues — review warnings above")
        sys.exit(1)


if __name__ == "__main__":
    main()
