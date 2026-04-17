#!/usr/bin/env python3
"""Validate sweep results — checks all result files for required fields and consistency.

Usage:
    python scripts/validate_sweep.py                    # validate all models
    python scripts/validate_sweep.py --model qwen3-8b   # validate one model
"""
import argparse
import json
import sys
from pathlib import Path

RUNS_DIR = Path("runs")

REQUIRED_RESULT_FIELDS = [
    "test_score", "val_score", "train_score",
    "seed_prompt_test_score", "seed_prompt_val_score",
    "best_prompt", "all_candidates", "rollout_count",
    "test_example_scores", "test_example_ids",
]

REQUIRED_METRICS_FIELDS = [
    "val_score_trajectory", "prompt_length_trajectory",
    "prompt_char_delta", "prompt_levenshtein_ratio",
    "task_error_count", "reflection_error_count",
    "val_example_scores", "train_example_scores",
    "total_tokens",
]

STANDALONE_METHODS = ["synaptic_pruning", "iso", "tournament"]

def validate_experiment(exp_dir: Path) -> list[str]:
    """Validate a single experiment directory. Returns list of error strings."""
    errors = []

    # Check result.json
    result_path = exp_dir / "result.json"
    if not result_path.exists():
        return [f"MISSING result.json"]

    try:
        result = json.loads(result_path.read_text())
    except json.JSONDecodeError:
        return [f"CORRUPT result.json — invalid JSON"]

    if result_path.stat().st_size < 100:
        errors.append(f"SUSPICIOUS result.json — only {result_path.stat().st_size} bytes")

    for field in REQUIRED_RESULT_FIELDS:
        if field not in result:
            errors.append(f"MISSING field in result.json: {field}")

    test_score = result.get("test_score")
    if test_score is not None and test_score == 0.0:
        errors.append(f"ZERO test_score — likely vLLM error")

    # Check metrics.json
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            errors.append(f"CORRUPT metrics.json")
            metrics = {}

        for field in REQUIRED_METRICS_FIELDS:
            if field not in metrics:
                errors.append(f"MISSING field in metrics.json: {field}")

        # Check trajectory starts at rollout=0
        traj = metrics.get("val_score_trajectory", [])
        if traj and traj[0][0] != 0:
            errors.append(f"val_score_trajectory does not start at rollout=0 (starts at {traj[0][0]})")

        # Check trajectory density for standalone methods
        method = result.get("method", "")
        if method in STANDALONE_METHODS and len(traj) < 3:
            errors.append(f"SPARSE trajectory for {method}: only {len(traj)} points (expected >=3)")

        # Check prompt_length_trajectory
        plt = metrics.get("prompt_length_trajectory", [])
        if len(plt) < 2:
            errors.append(f"prompt_length_trajectory has only {len(plt)} points (expected >=2)")

        # Check stage_timings for standalone methods
        if method in STANDALONE_METHODS:
            st = metrics.get("stage_timings", metrics.get("method_specific", {}).get("stage_timings"))
            if not st:
                errors.append(f"MISSING stage_timings for standalone method {method}")

        # Check non-zero tokens
        if metrics.get("total_tokens", 0) == 0:
            errors.append(f"ZERO total_tokens — token tracking may be broken")
    else:
        errors.append(f"MISSING metrics.json")

    # Check test_outputs.json
    outputs_path = exp_dir / "test_outputs.json"
    if not outputs_path.exists():
        errors.append(f"MISSING test_outputs.json")
    else:
        try:
            outputs = json.loads(outputs_path.read_text())
            if not isinstance(outputs, list) or len(outputs) == 0:
                errors.append(f"EMPTY test_outputs.json")
        except json.JSONDecodeError:
            errors.append(f"CORRUPT test_outputs.json")

    # Check config.json
    config_path = exp_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            for field in ["train_size", "val_size", "test_size"]:
                if field not in config:
                    errors.append(f"MISSING field in config.json: {field}")
        except json.JSONDecodeError:
            errors.append(f"CORRUPT config.json")
    else:
        errors.append(f"MISSING config.json")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate sweep results")
    parser.add_argument("--model", help="Validate only this model tag (e.g. qwen3-8b)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all experiments, not just failures")
    args = parser.parse_args()

    if args.model:
        model_dirs = [RUNS_DIR / args.model]
    else:
        model_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("qwen3")]

    total = 0
    passed = 0
    failed = 0
    all_errors = []

    for model_dir in sorted(model_dirs):
        env_path = model_dir / "environment.json"
        if not env_path.exists():
            print(f"WARNING: {model_dir.name} missing environment.json")

        for exp_dir in sorted(model_dir.rglob("*")):
            if not exp_dir.is_dir():
                continue
            if not (exp_dir / "result.json").exists() and not any(exp_dir.iterdir()):
                continue
            # Only validate leaf directories (those with result.json or that should have it)
            if exp_dir.name.isdigit():  # seed directory
                total += 1
                errors = validate_experiment(exp_dir)
                if errors:
                    failed += 1
                    rel = exp_dir.relative_to(RUNS_DIR)
                    all_errors.append((str(rel), errors))
                    if args.verbose:
                        print(f"FAIL {rel}")
                        for e in errors:
                            print(f"  x {e}")
                else:
                    passed += 1
                    if args.verbose:
                        print(f"PASS {exp_dir.relative_to(RUNS_DIR)}")

    print(f"\n{'='*60}")
    print(f"Validation: {passed} passed, {failed} failed, {total} total")

    if all_errors:
        print(f"\nFailures:")
        for path, errors in all_errors:
            print(f"  {path}:")
            for e in errors:
                print(f"    x {e}")
        sys.exit(1)
    else:
        print("All experiments PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
