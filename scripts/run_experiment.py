#!/usr/bin/env python3
"""Unified experiment entry point for all methods.

Dispatches to GEPA, ISO variants, MIPROv2, and random search via a single CLI.
All methods share the same contract: accept benchmark, seed, budget, reflection model.

Usage:
    python scripts/run_experiment.py \
        --method iso_refresh --benchmark hotpotqa \
        --task-model qwen3-8b --task-url http://localhost:8125/v1 \
        --budget-rollouts 1000 --seed 42 \
        --output-dir runs/phase3/hotpotqa/iso_refresh/42
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Method dispatch table: method_name → (module, function, extra_kwargs)
METHOD_DISPATCH = {
    # GEPA
    "gepa": {
        "module": "gepa_mutations.base",
        "function": "run_mutation",
        "kwargs": {},
    },
    # ISO variants
    "iso_personality": {
        "module": "iso.runner",
        "function": "run_iso",
        "kwargs": {"strategy_mode": "personality", "mutation_mode": "blind"},
    },
    "iso_personality_crosspollin": {
        "module": "iso.runner",
        "function": "run_iso",
        "kwargs": {"strategy_mode": "personality", "mutation_mode": "crosspollin"},
    },
    "iso_prescribed8": {
        "module": "iso.runner",
        "function": "run_iso",
        "kwargs": {"strategy_mode": "prescribed8", "mutation_mode": "blind"},
    },
    "iso_base": {
        "module": "iso.runner",
        "function": "run_iso",
        "kwargs": {"strategy_mode": "inductive", "k": 5, "mutation_mode": "blind"},
    },
    "iso_crosspollin": {
        "module": "iso.runner",
        "function": "run_iso",
        "kwargs": {"strategy_mode": "inductive", "k": 5, "mutation_mode": "crosspollin"},
    },
    "iso_refresh": {
        "module": "iso.runner",
        "function": "run_iso",
        "kwargs": {"strategy_mode": "inductive", "k": 5, "mutation_mode": "blind", "refresh_mode": "expand"},
    },
    # MIPROv2
    "miprov2": {
        "module": "miprov2.runner",
        "function": "run_miprov2",
        "kwargs": {},
    },
    # Random search (sanity floor)
    "random_search": {
        "module": "random_search.runner",
        "function": "run_random_search",
        "kwargs": {},
    },
}

VALID_METHODS = sorted(METHOD_DISPATCH.keys())
VALID_BENCHMARKS = ["hotpotqa", "hover", "pupa", "ifbench", "livebench", "aime"]


def _write_manifest(output_dir: Path, args: argparse.Namespace, wall_clock: float):
    """Write run manifest with git SHA, environment, config."""
    import subprocess

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_sha = "unknown"

    manifest = {
        "git_sha": git_sha,
        "method": args.method,
        "benchmark": args.benchmark,
        "seed": args.seed,
        "budget_rollouts": args.budget_rollouts,
        "task_model": args.task_model,
        "reflection_model": args.reflection_model,
        "wall_clock_seconds": round(wall_clock, 2),
        "python_version": sys.version,
        "timestamp": time.time(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Run a prompt optimization experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method", required=True, choices=VALID_METHODS,
        help=f"Optimization method: {', '.join(VALID_METHODS)}",
    )
    parser.add_argument("--benchmark", required=True, choices=VALID_BENCHMARKS)
    parser.add_argument("--task-model", default=None, help="Task model ID (e.g., qwen3-8b)")
    parser.add_argument("--task-url", default=None, help="Task model endpoint URL")
    parser.add_argument("--reflection-model", default=None, help="Reflection model ID (for external analyzer)")
    parser.add_argument("--reflection-url", default=None, help="Reflection model endpoint URL")
    parser.add_argument("--budget-rollouts", type=int, default=None, help="Rollout budget (task-model eval calls)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=None, help="Limit train/val to N examples")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")

    args = parser.parse_args()

    # Configure environment from CLI flags
    if args.task_model:
        os.environ["GEPA_MODEL"] = args.task_model
    if args.task_url:
        os.environ["GEPA_BASE_URL"] = args.task_url
    if args.reflection_model:
        os.environ["REFLECTION_MODEL"] = args.reflection_model
    if args.reflection_url:
        os.environ["REFLECTION_BASE_URL"] = args.reflection_url
    if args.output_dir:
        os.environ["RUNS_DIR"] = args.output_dir

    # Import and dispatch
    dispatch = METHOD_DISPATCH[args.method]
    module = __import__(dispatch["module"], fromlist=[dispatch["function"]])
    run_fn = getattr(module, dispatch["function"])

    # Build kwargs
    kwargs = {
        "benchmark": args.benchmark,
        "seed": args.seed,
        "max_metric_calls": args.budget_rollouts,
        **dispatch["kwargs"],
    }
    if args.subset:
        kwargs["subset"] = args.subset

    # For GEPA, the interface is different (uses MutationConfig)
    if args.method == "gepa":
        from gepa_mutations.base import MutationConfig
        config = MutationConfig(
            mutation_name="gepa",
            benchmark=args.benchmark,
            seed=args.seed,
            subset=args.subset,
            max_metric_calls=args.budget_rollouts,
        )
        kwargs = {"config": config}

    start = time.time()
    result = run_fn(**kwargs)
    wall_clock = time.time() - start

    # Write manifest
    if args.output_dir:
        _write_manifest(Path(args.output_dir), args, wall_clock)

    print(f"\nResult: test={result.test_score:.4f} val={result.val_score:.4f} "
          f"rollouts={result.rollout_count} wall_clock={wall_clock:.0f}s")


if __name__ == "__main__":
    main()
