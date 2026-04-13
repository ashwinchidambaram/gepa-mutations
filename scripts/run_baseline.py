#!/usr/bin/env python3
"""Baseline (no-optimization) evaluation for each (model, benchmark, seed).

Evaluates the seed prompt directly on test and val sets — no optimization loop.
This gives a floor to measure how much each optimization method improves.

Usage:
    # Single run
    python scripts/run_baseline.py --model Qwen/Qwen3-8B --base-url http://10.0.10.58:8125/v1 --benchmark hotpotqa --seed 42

    # All benchmarks, all seeds (defaults)
    python scripts/run_baseline.py --model Qwen/Qwen3-8B --base-url http://10.0.10.58:8125/v1

    # Dry run — preview what would execute
    python scripts/run_baseline.py --model Qwen/Qwen3-8B --base-url http://10.0.10.58:8125/v1 --dry-run

Environment variables (override CLI args):
    GEPA_MODEL      Model name (e.g. Qwen/Qwen3-8B)
    GEPA_BASE_URL   vLLM endpoint (e.g. http://10.0.10.58:8125/v1)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re as _re
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from gepa_mutations.base import evaluate_on_test
from gepa_mutations.benchmarks.loader import load_benchmark
from gepa_mutations.config import Settings
from gepa_mutations.runner.experiment import BENCHMARK_SEED_PROMPTS
from gepa_mutations.storage.local import save_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ALL_BENCHMARKS = ["hotpotqa", "pupa", "ifbench"]
ALL_SEEDS = [42, 123, 456, 789, 1024]


# ---------------------------------------------------------------------------
# Model tag derivation (mirrors run_all_local.py)
# ---------------------------------------------------------------------------

def _env_model_tag() -> str:
    """Derive a stable run-directory tag from GEPA_MODEL environment variable."""
    raw = os.environ.get("GEPA_MODEL", "").lower()
    m = _re.sub(r"-\d+bit$", "", raw)

    if "qwen" in m:
        if "27b" in m:  return "qwen3-27b-awq"
        if "32b" in m:  return "qwen3-32b"
        if "14b" in m:  return "qwen3-14b"
        if "8b"  in m:  return "qwen3-8b"
        if "4b"  in m:  return "qwen3-4b"
        if "1.7b" in m: return "qwen3-1.7b"
        if "0.6b" in m: return "qwen3-0.6b"
    if "gemma" in m:
        if "27b" in m:  return "gemma3-27b"
        if "12b" in m:  return "gemma3-12b"
        if "4b"  in m:  return "gemma3-4b"
        if "1b"  in m:  return "gemma3-1b"
    if "llama" in m:
        if "70b" in m:  return "llama3-70b"
        if "8b"  in m:  return "llama3-8b"
        if "3b"  in m:  return "llama3-3b"
        if "1b"  in m:  return "llama3-1b"

    return ""


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_baseline(benchmark: str, seed: int, model_tag: str, settings: Settings) -> dict | None:
    """Evaluate seed prompt on one (benchmark, seed) pair. Returns result dict or None if skipped."""
    # Result path — check skip-if-done
    if model_tag:
        result_path = Path(f"runs/{model_tag}/{benchmark}/baseline/{seed}/result.json")
    else:
        result_path = Path(f"runs/{benchmark}/baseline/{seed}/result.json")

    if result_path.exists():
        log.info("SKIP %s seed=%d — already done: %s", benchmark, seed, result_path)
        return None

    log.info("START %s seed=%d (model_tag=%s)", benchmark, seed, model_tag)
    t0 = time.time()

    # Load benchmark (data splits are seed-independent)
    data = load_benchmark(benchmark, seed=0)

    # Seed prompt — no optimization
    prompt_text = BENCHMARK_SEED_PROMPTS[benchmark]
    best_prompt = {"system_prompt": prompt_text}

    # Evaluate on test set
    log.info("  evaluating on test set (%d examples)...", len(data.test))
    test_result = evaluate_on_test(benchmark, best_prompt, data.test, settings)

    # Evaluate on val set for val_score
    log.info("  evaluating on val set (%d examples)...", len(data.val))
    val_result = evaluate_on_test(benchmark, best_prompt, data.val, settings)

    elapsed = time.time() - t0

    # Build result dict matching ExperimentResult.to_dict() format
    result_data = {
        "benchmark": benchmark,
        "method": "baseline",
        "seed": seed,
        "test_score": test_result.score,
        "val_score": val_result.score,
        "best_prompt": best_prompt,
        "rollout_count": 0,
        "wall_clock_seconds": elapsed,
        "num_candidates": 0,
        "test_example_scores": test_result.example_scores,
        "test_example_ids": test_result.example_ids,
    }

    # Save using existing persistence
    config_data = {
        "model": os.environ.get("GEPA_MODEL", ""),
        "base_url": os.environ.get("GEPA_BASE_URL", ""),
        "method": "baseline",
        "benchmark": benchmark,
        "seed": seed,
    }
    save_result(
        benchmark=benchmark,
        seed=seed,
        result_data=result_data,
        config_data=config_data,
        method="baseline",
        model_tag=model_tag,
    )

    log.info(
        "DONE  %s seed=%d — test_score=%.4f val_score=%.4f (%.1fs)",
        benchmark, seed, test_result.score, val_result.score, elapsed,
    )
    return result_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline (no-optimization) evaluation")
    parser.add_argument("--model", default=os.environ.get("GEPA_MODEL", ""),
                        help="Model name (e.g. Qwen/Qwen3-8B). Also reads GEPA_MODEL env var.")
    parser.add_argument("--base-url", default=os.environ.get("GEPA_BASE_URL", ""),
                        help="vLLM endpoint URL. Also reads GEPA_BASE_URL env var.")
    parser.add_argument("--benchmark", default="all",
                        help="Benchmark name or 'all' (default: all). Choices: " + ", ".join(ALL_BENCHMARKS))
    parser.add_argument("--seed", default=",".join(str(s) for s in ALL_SEEDS),
                        help="Comma-separated seeds (default: 42,123,456,789,1024)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned runs without executing")
    args = parser.parse_args()

    # Propagate to env so Settings() picks them up via pydantic-settings
    if args.model:
        os.environ["GEPA_MODEL"] = args.model
    if args.base_url:
        os.environ["GEPA_BASE_URL"] = args.base_url
        if not os.environ.get("API_BASE_URL"):
            os.environ["API_BASE_URL"] = args.base_url

    model_tag = _env_model_tag()
    if not model_tag:
        log.warning("Could not derive model_tag from GEPA_MODEL=%r — results will save without model prefix",
                     os.environ.get("GEPA_MODEL", ""))

    # Parse benchmarks and seeds
    benchmarks = ALL_BENCHMARKS if args.benchmark == "all" else [args.benchmark]
    seeds = [int(s.strip()) for s in args.seed.split(",")]

    # Build run list
    runs = [(bm, seed) for bm in benchmarks for seed in seeds]

    if args.dry_run:
        print(f"\nDry run: {len(runs)} baseline evaluations (model_tag={model_tag!r})\n")
        for bm, seed in runs:
            if model_tag:
                p = f"runs/{model_tag}/{bm}/baseline/{seed}/result.json"
            else:
                p = f"runs/{bm}/baseline/{seed}/result.json"
            done = Path(p).exists()
            status = "DONE" if done else "TODO"
            print(f"  [{status}] {bm} seed={seed} → {p}")
        return

    # Run sequentially
    settings = Settings()
    log.info("Starting baseline evaluation: %d runs, model=%s, base_url=%s",
             len(runs), os.environ.get("GEPA_MODEL", "?"), os.environ.get("GEPA_BASE_URL", "?"))

    completed = 0
    skipped = 0
    for bm, seed in runs:
        result = run_baseline(bm, seed, model_tag, settings)
        if result is None:
            skipped += 1
        else:
            completed += 1

    log.info("Baseline evaluation complete: %d completed, %d skipped", completed, skipped)


if __name__ == "__main__":
    main()
