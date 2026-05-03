"""Run SCALPEL optimization on the raycluster.

Uses Qwen3.5-27B (alias openai/gpt-oss-120b) via LiteLLM as both task
LM and reflection LM, matching the GEPA paper's self-reflection setup.
Per addendum Q4, thinking is OFF for both.

Usage (on gho-vm-2):
    uv run python scripts/raycluster/run_scalpel.py
    uv run python scripts/raycluster/run_scalpel.py --benchmark hotpotqa
    uv run python scripts/raycluster/run_scalpel.py --benchmark hotpotqa --seeds 42
    uv run python scripts/raycluster/run_scalpel.py --benchmark hotpotqa --max-iters 5  # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from rich.console import Console

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from config import (  # noqa: E402
    BENCHMARK_MAX_TOKENS,
    BENCHMARKS,
    INFERENCE_BASE_URL,
    INFRA_TAG,
    MAX_TOKENS_QA,
    MAX_TOKENS_REFLECT,
    MODEL_FULL_NAME,
    MODEL_TAG,
    SEEDS,
)

from gepa_mutations.benchmarks.evaluators import get_adapter  # noqa: E402
from gepa_mutations.benchmarks.loader import load_benchmark  # noqa: E402
from scalpel.edits.span_index import materialize  # noqa: E402
from scalpel.llm.client import LiteLLMClient  # noqa: E402
from scalpel.optimizer import SCALPEL  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()

# Per-benchmark suggested max iterations under the matched-rollouts regime.
# Each SCALPEL iteration uses ~226 rollouts (per addendum cost table), so:
#   GEPA budget / 226 ~= max_iters.
# These are upper bounds; we typically run 30-60 in practice.
MAX_ITERS_PER_BENCHMARK = {
    "hotpotqa": 30,    # GEPA: 6,871 rollouts -> ~30 SCALPEL iters
    "hover":    11,    # GEPA: 2,426 -> ~11
    "pupa":     17,    # GEPA: 3,936 -> ~17
    "ifbench":  16,    # GEPA: 3,593 -> ~16
    "livebench": 8,    # GEPA: 1,839 -> ~8
    "aime":     31,    # GEPA: 7,051 -> ~31 (Q10 sanity probe only)
}

SEED_PROMPT = "You are a helpful assistant."
BENCHMARK_SEED_PROMPTS = {
    "aime": (
        "You are a helpful assistant. You are given a question and you need to answer "
        "it. The answer should be given at the end of your response in exactly the "
        "format '### <final answer>'"
    ),
}


def get_seed_prompt(benchmark: str) -> str:
    return BENCHMARK_SEED_PROMPTS.get(benchmark, SEED_PROMPT)


def _make_metric(benchmark_name: str):
    """Build a (gold, pred) -> float metric using the benchmark adapter's _score."""
    adapter = get_adapter(benchmark_name, task_lm=None)

    def metric(gold, pred):
        # gold may be a dspy.Example, dict, or string with an `.answer` attr.
        example = gold
        if isinstance(gold, dict):
            example = type("Ex", (), {"answer": gold.get("answer", "")})()
        elif isinstance(gold, str):
            example = type("Ex", (), {"answer": gold})()
        elif not hasattr(gold, "answer"):
            example = type("Ex", (), {"answer": str(gold)})()
        response = pred if isinstance(pred, str) else str(getattr(pred, "answer", pred))
        score, _ = adapter._score(example, response)
        return float(score)

    return metric


def _make_feedback(benchmark_name: str):
    """Build a (gold, pred, trace=None) -> str feedback fn from adapter._score."""
    adapter = get_adapter(benchmark_name, task_lm=None)

    def feedback(gold, pred, trace=None):  # noqa: ARG001
        example = gold
        if isinstance(gold, dict):
            example = type("Ex", (), {"answer": gold.get("answer", "")})()
        elif isinstance(gold, str):
            example = type("Ex", (), {"answer": gold})()
        elif not hasattr(gold, "answer"):
            example = type("Ex", (), {"answer": str(gold)})()
        response = pred if isinstance(pred, str) else str(getattr(pred, "answer", pred))
        _, fb = adapter._score(example, response)
        return str(fb)

    return feedback


def run_scalpel_single(
    benchmark_name: str,
    seed: int,
    runs_dir: Path,
    max_iters: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Run SCALPEL on a single (benchmark, seed). Mirrors run_gepa_single."""
    result_path = runs_dir / benchmark_name / "scalpel" / str(seed) / "result.json"
    if result_path.exists():
        console.print(
            f"\n[dim]SCALPEL: {benchmark_name} / seed={seed} - result exists, skipping[/dim]"
        )
        with open(result_path) as f:
            return json.load(f)

    console.print(f"\n[bold]SCALPEL: {benchmark_name} / seed={seed}[/bold]")

    # Load data (seed=0 for data shuffling, like run_gepa.py)
    data = load_benchmark(benchmark_name, seed=0)
    trainset, valset, testset = data.train, data.val, data.test
    console.print(f"  Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")

    if dry_run:
        console.print("  [yellow]DRY RUN - instantiating SCALPEL but not optimizing[/yellow]")
        max_tokens = BENCHMARK_MAX_TOKENS.get(benchmark_name, MAX_TOKENS_QA)
        task_lm = LiteLLMClient(max_tokens_task=max_tokens)
        reflect_lm = task_lm  # same model
        _ = SCALPEL(task_lm=task_lm, reflect_lm=reflect_lm, max_iters=1, seed=seed)
        console.print("  Dry run OK")
        return {"dry_run": True, "benchmark": benchmark_name, "seed": seed}

    # Build LMs (single shared client for task+reflection, per addendum Q8)
    max_tokens = BENCHMARK_MAX_TOKENS.get(benchmark_name, MAX_TOKENS_QA)
    task_lm = LiteLLMClient(max_tokens_task=max_tokens, max_tokens_reflect=MAX_TOKENS_REFLECT)
    reflect_lm = task_lm  # self-reflection per GEPA paper

    metric = _make_metric(benchmark_name)
    feedback = _make_feedback(benchmark_name)

    # Configure optimizer
    iters = max_iters if max_iters is not None else MAX_ITERS_PER_BENCHMARK.get(benchmark_name, 30)
    console.print(f"  Max iters: {iters}")

    optimizer = SCALPEL(
        task_lm=task_lm,
        reflect_lm=reflect_lm,
        max_iters=iters,
        seed=seed,
    )

    seed_prompt_str = get_seed_prompt(benchmark_name)
    student = {"default": seed_prompt_str}

    # Compile
    t0 = time.time()
    try:
        optimized = optimizer.compile(
            student=student,
            trainset=trainset,
            valset=valset,
            metric=metric,
            feedback=feedback,
        )
    except Exception as e:
        console.print(f"  [bold red]SCALPEL optimization failed: {e}[/bold red]")
        error_result = {
            "metadata": {
                "model_tag": MODEL_TAG,
                "model_name": MODEL_FULL_NAME,
                "infra": INFRA_TAG,
                "inference_endpoint": INFERENCE_BASE_URL,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "benchmark": benchmark_name,
                "method": "scalpel",
                "error": str(e),
            },
            "test_score": 0.0,
            "wall_clock_seconds": time.time() - t0,
        }
        out_dir = runs_dir / benchmark_name / "scalpel" / str(seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "result.json", "w") as f:
            json.dump(error_result, f, indent=2)
        return error_result

    # Materialize the best prompts back into a flat dict
    best_prompt_dict = {mod: materialize(p) for mod, p in optimized.items()}
    best_prompt_for_eval = {
        "system_prompt": best_prompt_dict.get("default", SEED_PROMPT)
    }
    val_score = optimizer.best_candidate.pareto_score if optimizer.best_candidate else 0.0
    console.print(f"  Optimization complete: val_score={val_score:.4f}")

    # Evaluate best prompt on test set (mirror run_gepa.py:269-279)
    console.print(f"  Evaluating on test set ({len(testset)} examples)...")
    test_lm = LiteLLMClient(max_tokens_task=max_tokens)
    test_adapter = get_adapter(benchmark_name, task_lm=test_lm)
    test_batch_result = test_adapter.evaluate(testset, best_prompt_for_eval, capture_traces=False)
    test_score = (
        sum(test_batch_result.scores) / len(test_batch_result.scores)
        if test_batch_result.scores
        else 0.0
    )

    wall_clock = time.time() - t0
    console.print(
        f"  [bold green]Test score: {test_score:.4f}[/bold green] (val: {val_score:.4f})"
    )
    console.print(f"  Wall clock: {wall_clock:.1f}s")
    console.print(f"  Iterations: {len(optimizer.iteration_logs)}")
    console.print(f"  Candidates explored: {len(optimizer.candidates)}")

    # Build result.json (mirror run_gepa.py:291-320)
    git_sha = ""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        pass

    # Surrogate Brier (if surrogate exists and was used)
    brier = None
    try:
        if hasattr(optimizer.surrogate, "monitor"):
            brier = float(optimizer.surrogate.monitor.brier())
    except Exception:
        brier = None

    def _candidate_default_text(c) -> str:
        prompt_obj = c.prompts.get("default")
        if prompt_obj is None and c.prompts:
            prompt_obj = c.prompts[next(iter(c.prompts))]
        return materialize(prompt_obj) if prompt_obj is not None else ""

    result_dict = {
        "metadata": {
            "model_tag": MODEL_TAG,
            "model_name": MODEL_FULL_NAME,
            "infra": INFRA_TAG,
            "inference_endpoint": INFERENCE_BASE_URL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha,
            "seed": seed,
            "benchmark": benchmark_name,
            "method": "scalpel",
            "seed_prompt": seed_prompt_str,
            "max_iters": iters,
        },
        "test_score": test_score,
        "val_score": val_score,
        "best_prompt": best_prompt_for_eval,
        "all_candidates": [
            {
                "prompt": {"system_prompt": _candidate_default_text(c)},
                "score": c.pareto_score,
            }
            for c in optimizer.candidates[:20]
        ],
        "test_example_scores": list(test_batch_result.scores),
        "wall_clock_seconds": wall_clock,
        "scalpel_specific": {
            "iterations": len(optimizer.iteration_logs),
            "candidates_explored": len(optimizer.candidates),
            "iteration_logs": [log.model_dump(mode="json") for log in optimizer.iteration_logs],
            "cluster_k_history": [log.cluster_k for log in optimizer.iteration_logs],
            "lesson_book_size_at_end": len(optimizer.lesson_book.lessons),
            "surrogate_brier": brier,
            "surrogate_enabled": bool(getattr(optimizer.surrogate, "enabled", False)),
        },
    }

    last_usage = task_lm._last_usage
    total_input = last_usage.prompt_tokens if last_usage else 0
    total_output = last_usage.completion_tokens if last_usage else 0
    metrics = {
        "total_tokens": total_input + total_output,
        "task_tokens_input": total_input,
        "task_tokens_output": total_output,
        # NOTE: we share one LiteLLMClient for task+reflect calls, so the breakdown
        # is approximate. Future: separate clients with separate counters.
    }

    out_dir = runs_dir / benchmark_name / "scalpel" / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump(result_dict, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    console.print(f"  Saved to {out_dir.relative_to(PROJECT_ROOT)}")
    return result_dict


def main():
    parser = argparse.ArgumentParser(description="Run SCALPEL optimization on raycluster")
    parser.add_argument("--benchmark", nargs="+", default=BENCHMARKS, help="Benchmarks to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Random seeds")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Override runs directory")
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Override per-benchmark default max iterations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify wiring without making real LM calls",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    runs_dir = args.runs_dir or (PROJECT_ROOT / "runs" / MODEL_TAG)
    runs_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]" + "=" * 60)
    console.print("[bold]Raycluster SCALPEL Optimization")
    console.print(f"[bold]Model: {MODEL_FULL_NAME} ({MODEL_TAG})")
    console.print(f"[bold]Endpoint: {INFERENCE_BASE_URL}")
    console.print(f"[bold]Benchmarks: {args.benchmark}")
    console.print(f"[bold]Seeds: {args.seeds}")
    if args.max_iters is not None:
        console.print(f"[bold]Max iters override: {args.max_iters}")
    if args.dry_run:
        console.print("[bold yellow]DRY RUN MODE")
    console.print("[bold]" + "=" * 60)

    # Quick connectivity check (skip in dry-run)
    if not args.dry_run:
        try:
            resp = requests.get(f"{INFERENCE_BASE_URL}/models", timeout=10)
            resp.raise_for_status()
            console.print("[green]API reachable[/green]\n")
        except Exception as e:
            console.print(f"[bold red]API not reachable: {e}[/bold red]")
            sys.exit(1)

    # Run all benchmark x seed combinations
    all_results = []
    total_t0 = time.time()

    for benchmark in args.benchmark:
        for seed in args.seeds:
            result = run_scalpel_single(
                benchmark, seed, runs_dir,
                max_iters=args.max_iters,
                dry_run=args.dry_run,
            )
            all_results.append(result)

    # Final summary
    total_time = time.time() - total_t0
    console.print(f"\n[bold]{'=' * 60}")
    console.print(
        f"[bold]SCALPEL COMPLETE - {len(all_results)} runs in "
        f"{total_time:.0f}s ({total_time/3600:.1f}h)"
    )
    console.print(f"[bold]{'=' * 60}\n")

    if not args.dry_run:
        console.print("[bold]Results Summary:[/bold]")
        for r in all_results:
            md = r.get("metadata", {})
            bm = md.get("benchmark", "?")
            seed = md.get("seed", "?")
            score = r.get("test_score", 0.0)
            console.print(f"  {bm:12s} seed={seed!s:4s}  test_score={score:.4f}")


if __name__ == "__main__":
    main()
