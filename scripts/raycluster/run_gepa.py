"""Run GEPA optimization on the raycluster.

Uses Qwen3.5-27B as BOTH task LM and reflection LM (self-reflection,
matching the GEPA paper which uses the same model for both roles).

Usage (on gho-vm-2):
    uv run python scripts/raycluster/run_gepa.py
    uv run python scripts/raycluster/run_gepa.py --benchmark hotpotqa hover
    uv run python scripts/raycluster/run_gepa.py --benchmark hotpotqa --seeds 42 123
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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

from config import (  # noqa: E402
    BENCHMARKS,
    BENCHMARK_MAX_TOKENS,
    BENCHMARK_PARALLEL_WORKERS,
    DISABLE_THINKING,
    INFERENCE_BASE_URL,
    INFRA_TAG,
    MAX_TOKENS_QA,
    MAX_TOKENS_REFLECT,
    MODEL_FULL_NAME,
    MODEL_NAME,
    MODEL_TAG,
    PARALLEL_WORKERS,
    SEEDS,
    TEMPERATURE,
    TOP_P,
)

from gepa_mutations.benchmarks.loader import load_benchmark  # noqa: E402
from gepa_mutations.benchmarks.evaluators import get_adapter  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()

# GEPA paper hyperparameters
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

# Rollout budgets matching the GEPA paper (arXiv:2507.19457v2)
PAPER_ROLLOUTS = {
    "hotpotqa": 6871,
    "hover": 2426,
    "pupa": 3936,
    "ifbench": 3593,
    "livebench": 1839,
    "aime": 7051,
}


class ScoreTracker:
    """GEPA callback that records per-iteration scores for convergence tracking."""

    def __init__(self):
        self.iteration_scores: list[dict] = []
        self.best_score_so_far = 0.0

    def on_iteration_end(self, event) -> None:
        """Called after each GEPA iteration with state."""
        iteration = event["iteration"]
        state = event["state"]

        # Extract current best score from state
        scores = state.val_aggregate_scores if hasattr(state, "val_aggregate_scores") else []
        current_best = max(scores) if scores else 0.0
        num_candidates = len(scores)

        self.best_score_so_far = max(self.best_score_so_far, current_best)
        accepted = event.get("proposal_accepted", None)

        self.iteration_scores.append({
            "iteration": iteration,
            "best_score": current_best,
            "best_score_cumulative": self.best_score_so_far,
            "num_candidates": num_candidates,
            "proposal_accepted": accepted,
            "all_scores": sorted(scores, reverse=True)[:10],  # Top 10
        })


class ClusterLM:
    """LM wrapper that calls the cluster inference API.

    Implements the callable interface expected by QAAdapter and gepa's
    reflection LM: __call__(messages) -> str
    """

    def __init__(self, max_tokens: int = MAX_TOKENS_QA, role: str = "task"):
        self.max_tokens = max_tokens
        self.role = role
        self.total_tokens = 0
        self.call_count = 0
        self.errors = 0

    def __call__(self, messages) -> str:
        """Call the inference API. Accepts a messages list or a plain string prompt."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }
        if DISABLE_THINKING:
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        try:
            resp = requests.post(
                f"{INFERENCE_BASE_URL}/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            msg = data["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""

            # Strip <think> blocks
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            usage = data.get("usage", {})
            self.total_tokens += usage.get("total_tokens", 0)
            self.call_count += 1
            return text

        except Exception as e:
            self.errors += 1
            logger.warning(f"LM call error ({self.role}): {e}")
            raise


def run_gepa_single(benchmark_name: str, seed: int, runs_dir: Path) -> dict:
    """Run GEPA optimization on a single benchmark + seed."""
    # Skip if result already exists
    result_path = runs_dir / benchmark_name / "gepa" / str(seed) / "result.json"
    if result_path.exists():
        console.print(f"\n[dim]GEPA: {benchmark_name} / seed={seed} — result exists, skipping[/dim]")
        with open(result_path) as f:
            return json.load(f)

    console.print(f"\n[bold]GEPA: {benchmark_name} / seed={seed}[/bold]")

    # Load data
    data = load_benchmark(benchmark_name, seed=0)  # GEPA uses seed=0 for data loading
    trainset = data.train
    valset = data.val
    testset = data.test
    console.print(f"  Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")

    # Build LMs
    max_tokens = BENCHMARK_MAX_TOKENS.get(benchmark_name, MAX_TOKENS_QA)
    task_lm = ClusterLM(max_tokens=max_tokens, role="task")
    reflection_lm = ClusterLM(max_tokens=MAX_TOKENS_REFLECT, role="reflection")

    # Build adapter with parallel evaluation (fewer workers for math benchmarks)
    workers = BENCHMARK_PARALLEL_WORKERS.get(benchmark_name, PARALLEL_WORKERS)
    adapter = get_adapter(benchmark_name, task_lm=task_lm, parallel_workers=workers)

    # Seed candidate
    seed_candidate = {"system_prompt": get_seed_prompt(benchmark_name)}

    # Rollout budget
    budget = PAPER_ROLLOUTS.get(benchmark_name, 5000)
    console.print(f"  Budget: {budget} rollouts")

    # Run directory for GEPA state
    run_dir = str(runs_dir / benchmark_name / "gepa" / str(seed) / "gepa_state")
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Import and run GEPA optimization
    from gepa.api import optimize

    # Score tracker callback — records per-iteration convergence data
    score_tracker = ScoreTracker()

    console.print(f"  Starting GEPA optimization...")
    try:
        result = optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            candidate_selection_strategy="pareto",
            frontier_type="instance",
            skip_perfect_score=True,
            perfect_score=1.0,
            module_selector="round_robin",
            use_merge=True,
            max_merge_invocations=5,
            merge_val_overlap_floor=5,
            reflection_minibatch_size=3,
            max_metric_calls=budget,
            cache_evaluation=True,
            seed=seed,
            run_dir=run_dir,
            callbacks=[score_tracker],
            display_progress_bar=True,
            raise_on_exception=False,
        )
    except Exception as e:
        console.print(f"  [bold red]GEPA optimization failed: {e}[/bold red]")
        # Save error result
        error_result = {
            "metadata": {
                "model_tag": MODEL_TAG,
                "model_name": MODEL_FULL_NAME,
                "infra": INFRA_TAG,
                "inference_endpoint": INFERENCE_BASE_URL,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "benchmark": benchmark_name,
                "method": "gepa",
                "error": str(e),
            },
            "test_score": 0.0,
            "wall_clock_seconds": time.time() - t0,
        }
        out_dir = runs_dir / benchmark_name / "gepa" / str(seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "result.json", "w") as f:
            json.dump(error_result, f, indent=2)
        return error_result

    # Extract best prompt
    best_prompt = result.best_candidate
    if isinstance(best_prompt, str):
        best_prompt = {"system_prompt": best_prompt}
    val_score = result.val_aggregate_scores[result.best_idx]

    console.print(f"  Optimization complete: val_score={val_score:.4f}")
    console.print(f"  Candidates explored: {result.num_candidates}")

    # Evaluate best prompt on test set
    console.print(f"  Evaluating on test set ({len(testset)} examples)...")
    test_lm = ClusterLM(max_tokens=max_tokens, role="test_eval")
    test_adapter = get_adapter(benchmark_name, task_lm=test_lm, parallel_workers=workers)

    from gepa.core.adapter import EvaluationBatch
    test_batch_result = test_adapter.evaluate(testset, best_prompt, capture_traces=False)
    test_score = sum(test_batch_result.scores) / len(test_batch_result.scores)

    wall_clock = time.time() - t0
    console.print(f"  [bold green]Test score: {test_score:.4f}[/bold green] (val: {val_score:.4f})")
    console.print(f"  Wall clock: {wall_clock:.1f}s")
    console.print(f"  Task LM: {task_lm.call_count} calls, {task_lm.total_tokens} tokens")
    console.print(f"  Reflection LM: {reflection_lm.call_count} calls, {reflection_lm.total_tokens} tokens")

    # Build result
    git_sha = ""
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        pass

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
            "method": "gepa",
            "seed_prompt": get_seed_prompt(benchmark_name),
            "rollout_budget": budget,
        },
        "test_score": test_score,
        "val_score": val_score,
        "train_score": None,
        "seed_prompt_test_score": None,  # Could evaluate if needed
        "best_prompt": best_prompt,
        "all_candidates": [
            {"prompt": c, "score": s}
            for c, s in zip(
                result.all_candidates[:20],
                result.val_aggregate_scores[:20],
            )
        ] if hasattr(result, "all_candidates") else [],
        "test_example_scores": test_batch_result.scores,
        "rollout_count": task_lm.call_count + test_lm.call_count,
        "wall_clock_seconds": wall_clock,
    }

    metrics = {
        "total_tokens": task_lm.total_tokens + reflection_lm.total_tokens + test_lm.total_tokens,
        "task_tokens": task_lm.total_tokens,
        "reflection_tokens": reflection_lm.total_tokens,
        "task_calls": task_lm.call_count,
        "reflection_calls": reflection_lm.call_count,
        "task_error_count": task_lm.errors,
        "reflection_error_count": reflection_lm.errors,
        "iteration_scores": score_tracker.iteration_scores,
        "val_score_trajectory": [
            (i["iteration"], i["best_score_cumulative"])
            for i in score_tracker.iteration_scores
        ],
    }

    # Save results
    out_dir = runs_dir / benchmark_name / "gepa" / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "result.json", "w") as f:
        json.dump(result_dict, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    console.print(f"  Saved to {out_dir.relative_to(PROJECT_ROOT)}")
    return result_dict


def main():
    parser = argparse.ArgumentParser(description="Run GEPA optimization on raycluster")
    parser.add_argument("--benchmark", nargs="+", default=BENCHMARKS, help="Benchmarks to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Random seeds")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Override runs directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    runs_dir = args.runs_dir or (PROJECT_ROOT / "runs" / MODEL_TAG)
    runs_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]=" * 60)
    console.print("[bold]Raycluster GEPA Optimization")
    console.print(f"[bold]Model: {MODEL_FULL_NAME} ({MODEL_TAG})")
    console.print(f"[bold]Endpoint: {INFERENCE_BASE_URL}")
    console.print(f"[bold]Benchmarks: {args.benchmark}")
    console.print(f"[bold]Seeds: {args.seeds}")
    console.print("[bold]=" * 60)

    # Quick connectivity check
    try:
        resp = requests.get(f"{INFERENCE_BASE_URL}/models", timeout=10)
        resp.raise_for_status()
        console.print("[green]API reachable[/green]\n")
    except Exception as e:
        console.print(f"[bold red]API not reachable: {e}[/bold red]")
        sys.exit(1)

    # Run all benchmark × seed combinations
    all_results = []
    total_t0 = time.time()

    for benchmark in args.benchmark:
        for seed in args.seeds:
            result = run_gepa_single(benchmark, seed, runs_dir)
            all_results.append(result)

    # Final summary
    total_time = time.time() - total_t0
    console.print(f"\n[bold]{'=' * 60}")
    console.print(f"[bold]GEPA COMPLETE — {len(all_results)} runs in {total_time:.0f}s ({total_time/3600:.1f}h)")
    console.print(f"[bold]{'=' * 60}\n")

    console.print("[bold]Results Summary:[/bold]")
    for r in all_results:
        bm = r["metadata"]["benchmark"]
        seed = r["metadata"]["seed"]
        score = r["test_score"]
        console.print(f"  {bm:12s} seed={seed:4d}  test_score={score:.4f}")


if __name__ == "__main__":
    main()
