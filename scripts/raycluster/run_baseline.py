"""Run baseline evaluation (no optimization) on the raycluster.

Evaluates the raw model on all benchmarks with a simple seed prompt.
Results are saved in runs/{model_tag}/{benchmark}/baseline/{seed}/.

Usage (on gho-vm-2):
    uv run python scripts/raycluster/run_baseline.py
    uv run python scripts/raycluster/run_baseline.py --benchmark hotpotqa hover
    uv run python scripts/raycluster/run_baseline.py --benchmark hotpotqa --seeds 42 123
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

# Setup paths — add src/ to path for imports
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
    MODEL_FULL_NAME,
    MODEL_NAME,
    MODEL_TAG,
    PARALLEL_WORKERS,
    SEEDS,
    TEMPERATURE,
    TOP_P,
)

# Now import benchmark infrastructure
from gepa_mutations.benchmarks.loader import load_benchmark  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()

# Seed prompt — matches GEPA canonical baseline
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


def call_model(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = MAX_TOKENS_QA,
) -> tuple[str, dict]:
    """Call the cluster inference API.

    Returns:
        (response_text, usage_dict)
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }

    # Disable thinking mode — model responds directly in content field
    if DISABLE_THINKING:
        payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    # Scale timeout with max_tokens — math benchmarks (4096 tokens) need more time
    timeout = 120 if max_tokens <= 512 else 300
    resp = requests.post(
        f"{INFERENCE_BASE_URL}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    msg = data["choices"][0]["message"]
    text = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""

    # Strip any <think>...</think> blocks (safety fallback)
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    usage = data.get("usage", {})
    return text, usage


def score_example(benchmark_name: str, example, response: str) -> float:
    """Score a single example response.

    Uses lightweight scoring logic matching the evaluators.
    """
    answer = str(example.answer).strip().lower() if hasattr(example, "answer") else ""

    if benchmark_name == "hotpotqa":
        # Word-boundary containment
        import re
        return 1.0 if re.search(r'\b' + re.escape(answer) + r'\b', response.lower()) else 0.0

    elif benchmark_name == "hover":
        # Label match — check if model verdict aligns with gold label.
        # Model may say: "NOT_SUPPORTED", "not supported", "false", "refuted", etc.
        resp_lower = response.lower()

        # Determine model's verdict: check for NOT_SUPPORTED indicators first
        # (since "supported" is a substring of "not supported")
        not_supported_indicators = [
            "not_supported", "not supported", "not enough", "insufficient",
            "does not support", "doesn't support", "cannot be verified",
            "no evidence", "contradicts", "refuted", "false", "incorrect",
        ]
        supported_indicators = [
            "supported", "verified", "confirmed", "true", "correct",
        ]

        model_says_not_supported = any(ind in resp_lower for ind in not_supported_indicators)
        model_says_supported = any(ind in resp_lower for ind in supported_indicators) and not model_says_not_supported

        if answer in ("not_supported", "0"):
            return 1.0 if model_says_not_supported else 0.0
        else:  # answer is "supported" or "1"
            return 1.0 if model_says_supported else 0.0

    elif benchmark_name == "ifbench":
        # Constraint satisfaction — use same evaluator as GEPA adapter
        from gepa_mutations.benchmarks.evaluators import _check_ifbench_by_id, _check_ifbench_constraint
        instruction_ids = getattr(example, "instruction_ids", [])
        kwargs_list = getattr(example, "kwargs_list", [])
        constraints = getattr(example, "constraints", [])

        if not instruction_ids and not constraints:
            return 0.0

        satisfied = 0
        total = 0
        if instruction_ids:
            for i, iid in enumerate(instruction_ids):
                kw = kwargs_list[i] if i < len(kwargs_list) else None
                if _check_ifbench_by_id(iid, kw, response):
                    satisfied += 1
                total += 1
        else:
            for constraint in constraints:
                if _check_ifbench_constraint(constraint, response):
                    satisfied += 1
                total += 1

        return satisfied / total if total > 0 else 0.0

    elif benchmark_name == "pupa":
        # PII leakage check
        pii_units = getattr(example, "pii_units", [])
        if not pii_units:
            return 1.0
        leaked = sum(1 for pii in pii_units if pii.lower() in response.lower())
        leakage_score = 1.0 - (leaked / len(pii_units))
        # Quality is harder to judge without LLM — use word overlap as proxy
        answer_words = set(answer.split())
        response_words = set(response.lower().split())
        quality = len(answer_words & response_words) / max(len(answer_words), 1)
        return (quality + leakage_score) / 2

    elif benchmark_name == "livebench":
        # Normalized string comparison
        norm_answer = answer.strip().lower().replace(",", "").replace(" ", "")
        norm_response = response.strip().lower().replace(",", "").replace(" ", "")
        if norm_answer == norm_response:
            return 1.0
        if norm_answer in norm_response:
            return 1.0
        # Check for boxed answer: \boxed{...}
        import re
        boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
        if boxed:
            norm_boxed = boxed[-1].strip().lower().replace(",", "").replace(" ", "")
            if norm_boxed == norm_answer:
                return 1.0
        return 0.0

    elif benchmark_name == "aime":
        # Integer extraction and exact match
        import re
        nums = re.findall(r'\b(\d+)\b', response)
        if nums:
            # Take the last number (most likely the final answer)
            predicted = int(nums[-1])
            return 1.0 if predicted == int(answer) else 0.0
        return 0.0

    return 0.0


def run_baseline_single(benchmark_name: str, seed: int, runs_dir: Path) -> dict:
    """Run baseline on a single benchmark + seed."""
    # Skip if result already exists
    result_path = runs_dir / benchmark_name / "baseline" / str(seed) / "result.json"
    if result_path.exists():
        console.print(f"\n[dim]Baseline: {benchmark_name} / seed={seed} — result exists, skipping[/dim]")
        with open(result_path) as f:
            return json.load(f)

    console.print(f"\n[bold]Running {benchmark_name} / seed={seed}[/bold]")

    # Load data
    data = load_benchmark(benchmark_name, seed=seed)
    test_set = data.test
    max_tokens = BENCHMARK_MAX_TOKENS.get(benchmark_name, MAX_TOKENS_QA)
    console.print(f"  Loaded {len(test_set)} test examples (max_tokens={max_tokens})")

    # Run evaluation in parallel — vLLM handles concurrent requests via continuous batching
    scores = [0.0] * len(test_set)
    outputs = [None] * len(test_set)
    total_tokens = 0
    errors = 0
    t0 = time.time()

    def _eval_one(idx: int, example):
        user_input = str(example.input) if hasattr(example, "input") else str(example.question)
        try:
            response, usage = call_model(get_seed_prompt(benchmark_name), user_input, max_tokens=max_tokens)
            score = score_example(benchmark_name, example, response)
            return idx, score, {"input": user_input[:200], "response": response, "score": score}, usage, None
        except Exception as e:
            return idx, 0.0, {"input": user_input[:200], "response": "", "score": 0.0, "error": str(e)}, {}, e

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"  {benchmark_name}", total=len(test_set))

        workers = BENCHMARK_PARALLEL_WORKERS.get(benchmark_name, PARALLEL_WORKERS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_eval_one, i, ex) for i, ex in enumerate(test_set)]
            for future in as_completed(futures):
                idx, score, output, usage, err = future.result()
                scores[idx] = score
                outputs[idx] = output
                total_tokens += usage.get("total_tokens", 0)
                if err:
                    errors += 1
                    logger.warning(f"Error on example {idx}: {err}")
                progress.update(task, advance=1)

    wall_clock = time.time() - t0
    test_score = sum(scores) / len(scores) if scores else 0.0

    console.print(f"  Score: [bold green]{test_score:.4f}[/bold green] ({sum(s > 0 for s in scores)}/{len(scores)} correct)")
    console.print(f"  Time: {wall_clock:.1f}s | Tokens: {total_tokens} | Errors: {errors}")

    # Build result
    git_sha = ""
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        pass

    result = {
        "metadata": {
            "model_tag": MODEL_TAG,
            "model_name": MODEL_FULL_NAME,
            "infra": INFRA_TAG,
            "inference_endpoint": INFERENCE_BASE_URL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha,
            "seed": seed,
            "benchmark": benchmark_name,
            "method": "baseline",
            "seed_prompt": get_seed_prompt(benchmark_name),
        },
        "test_score": test_score,
        "val_score": None,
        "train_score": None,
        "seed_prompt_test_score": test_score,  # Same as test_score for baseline
        "best_prompt": {"system": get_seed_prompt(benchmark_name)},
        "test_example_scores": scores,
        "rollout_count": len(test_set),
        "wall_clock_seconds": wall_clock,
    }

    metrics = {
        "total_tokens": total_tokens,
        "task_error_count": errors,
        "reflection_error_count": 0,
        "val_score_trajectory": [],
        "prompt_length_trajectory": [],
    }

    # Save results
    out_dir = runs_dir / benchmark_name / "baseline" / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "test_outputs.json", "w") as f:
        json.dump(outputs, f, indent=2)

    console.print(f"  Saved to {out_dir.relative_to(PROJECT_ROOT)}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation on raycluster")
    parser.add_argument("--benchmark", nargs="+", default=BENCHMARKS, help="Benchmarks to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Random seeds")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Override runs directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    runs_dir = args.runs_dir or (PROJECT_ROOT / "runs" / MODEL_TAG)
    runs_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]=" * 60)
    console.print("[bold]Raycluster Baseline Evaluation")
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
        console.print("Check VPN and cluster status.")
        sys.exit(1)

    # Run all benchmark × seed combinations
    # Seeds run in parallel per benchmark (each seed is independent)
    all_results = []
    total_t0 = time.time()

    for benchmark in args.benchmark:
        with ThreadPoolExecutor(max_workers=len(args.seeds)) as seed_pool:
            futures = {
                seed_pool.submit(run_baseline_single, benchmark, seed, runs_dir): seed
                for seed in args.seeds
            }
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    console.print(f"  [bold red]FAILED {benchmark}/seed={seed}: {e}[/bold red]")
                    all_results.append({"metadata": {"benchmark": benchmark, "seed": seed}, "test_score": 0.0})

    # Final summary
    total_time = time.time() - total_t0
    console.print(f"\n[bold]{'=' * 60}")
    console.print(f"[bold]COMPLETE — {len(all_results)} runs in {total_time:.0f}s")
    console.print(f"[bold]{'=' * 60}\n")

    # Summary table
    console.print("[bold]Results Summary:[/bold]")
    for r in all_results:
        bm = r["metadata"]["benchmark"]
        seed = r["metadata"]["seed"]
        score = r["test_score"]
        console.print(f"  {bm:12s} seed={seed:4d}  score={score:.4f}")

    # Aggregate by benchmark
    console.print(f"\n[bold]Aggregate (mean across seeds):[/bold]")
    from collections import defaultdict
    by_benchmark = defaultdict(list)
    for r in all_results:
        by_benchmark[r["metadata"]["benchmark"]].append(r["test_score"])
    for bm, scores in sorted(by_benchmark.items()):
        mean = sum(scores) / len(scores)
        console.print(f"  {bm:12s}  {mean:.4f} (n={len(scores)})")


if __name__ == "__main__":
    main()
