"""Random search baseline matching the unified CLI contract.

Sanity floor: generate N candidate prompts from the reflection LM,
evaluate each on the full validation set, return the best.

If a structured optimization method can't beat this, either the budget
is too low or the benchmark has no headroom.

Budget accounting:
    Each candidate costs len(valset) rollouts to evaluate.
    N = budget_rollouts // len(valset)
    Total rollouts = N * len(valset)
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console

from gepa_mutations.base import build_qa_task_lm, build_reflection_lm_for_model, evaluate_on_test
from gepa_mutations.benchmarks.loader import load_benchmark
from gepa_mutations.config import PAPER_ROLLOUTS, Settings, model_tag as get_model_tag
from gepa_mutations.metrics.standalone_eval import evaluate_prompt
from gepa_mutations.metrics.collector import MetricsCollector
from gepa_mutations.benchmarks.evaluators import get_adapter
from gepa_mutations.runner.experiment import (
    BENCHMARK_SEED_PROMPTS,
    BENCHMARK_TASK_INSTRUCTIONS,
    SEED_PROMPT,
    ExperimentResult,
)
from gepa_mutations.storage.local import save_result

console = Console()

_GENERATION_PROMPT = """Generate {n} diverse system prompts for an AI assistant.

Task description: {task_description}

Requirements:
- Each prompt should give the AI different instructions for how to approach the task
- Prompts should be diverse in strategy, tone, and specificity
- Each prompt should be 1-4 sentences
- Format: number each prompt on its own line, like:
1. [prompt text]
2. [prompt text]
...

Generate exactly {n} prompts:"""


def _generate_random_prompts(
    reflection_lm,
    task_description: str,
    n: int,
    seed: int,
) -> list[str]:
    """Generate N diverse prompts using the reflection LM."""
    import random

    prompt = _GENERATION_PROMPT.format(n=n, task_description=task_description)
    messages = [{"role": "user", "content": prompt}]

    response = reflection_lm(messages)
    text = response if isinstance(response, str) else str(response)

    # Parse numbered prompts from response
    prompts = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading number and punctuation
        import re
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        if cleaned and len(cleaned) > 10:
            prompts.append(cleaned)

    # Pad with variations if we didn't get enough
    rng = random.Random(seed)
    while len(prompts) < n:
        if prompts:
            base = rng.choice(prompts)
            prompts.append(f"{base} Be thorough and precise.")
        else:
            prompts.append("You are a helpful assistant. Think step by step.")

    return prompts[:n]


def run_random_search(
    benchmark: str = "hotpotqa",
    seed: int = 42,
    subset: int | None = None,
    max_metric_calls: int | None = None,
    settings: Settings | None = None,
) -> ExperimentResult:
    """Run random search baseline.

    Generate N prompts, evaluate each on validation set, return best.
    """
    settings = settings or Settings()
    start_time = time.time()

    # 1. Load benchmark
    console.print(f"[bold]Loading benchmark: {benchmark}[/bold]")
    data = load_benchmark(benchmark, seed=0)
    trainset = data.train[:subset] if subset else data.train
    valset = data.val[:subset] if subset else data.val
    testset = data.test

    # 2. Compute budget
    if max_metric_calls is None:
        max_metric_calls = PAPER_ROLLOUTS["gepa"].get(benchmark, 5000)

    n_candidates = max(1, max_metric_calls // len(valset))
    actual_rollouts = n_candidates * len(valset)

    console.print(f"  Budget: {max_metric_calls} → {n_candidates} candidates × {len(valset)} val examples = {actual_rollouts} rollouts")

    # 3. Build LMs
    task_lm = build_qa_task_lm(settings)
    reflection_lm = build_reflection_lm_for_model(settings)
    adapter = get_adapter(benchmark, task_lm=task_lm)
    collector = MetricsCollector()

    # 4. Generate random prompts
    from gepa_mutations.runner.experiment import BENCHMARK_TASK_INSTRUCTIONS
    task_desc = BENCHMARK_TASK_INSTRUCTIONS.get(benchmark, "Solve the task correctly.")

    console.print(f"\n[bold]Generating {n_candidates} random prompts...[/bold]")
    prompts = _generate_random_prompts(reflection_lm, task_desc, n_candidates, seed)
    console.print(f"  Generated {len(prompts)} prompts")

    # 5. Evaluate each on validation set
    console.print(f"\n[bold]Evaluating {len(prompts)} candidates on val set...[/bold]")
    best_score = -1.0
    best_prompt_text = SEED_PROMPT
    all_candidates = []

    for i, prompt_text in enumerate(prompts):
        candidate = {"system_prompt": prompt_text}
        score, _ = evaluate_prompt(adapter, valset, candidate, collector)
        all_candidates.append(candidate)
        console.print(f"  Candidate {i+1}/{len(prompts)}: val={score:.4f}")
        if score > best_score:
            best_score = score
            best_prompt_text = prompt_text

    best_prompt = {"system_prompt": best_prompt_text}
    console.print(f"\n  Best val score: {best_score:.4f}")

    # 6. Evaluate seed and best on test
    seed_prompt = BENCHMARK_SEED_PROMPTS.get(benchmark, SEED_PROMPT)
    seed_candidate = {"system_prompt": seed_prompt}

    console.print(f"\n[bold]Evaluating on test set...[/bold]")
    seed_test_eval = evaluate_on_test(benchmark, seed_candidate, testset, settings)
    test_eval = evaluate_on_test(benchmark, best_prompt, testset, settings)
    console.print(f"  Seed test: {seed_test_eval.score:.4f}, Best test: {test_eval.score:.4f}")

    # 7. Build result
    wall_clock = time.time() - start_time
    model_tag_str = get_model_tag(settings)

    result = ExperimentResult(
        benchmark=benchmark,
        seed=seed,
        test_score=test_eval.score,
        val_score=best_score,
        best_prompt=best_prompt,
        rollout_count=actual_rollouts,
        config_snapshot={
            "method": "random_search",
            "n_candidates": n_candidates,
            "budget_requested": max_metric_calls,
            "budget_actual": actual_rollouts,
        },
        wall_clock_seconds=wall_clock,
        method="random_search",
        all_candidates=all_candidates,
        test_example_scores=test_eval.per_example_scores,
        seed_prompt_test_score=seed_test_eval.score,
        train_score=None,
    )

    save_result(result, model_tag=model_tag_str)
    console.print(f"\n[bold green]Random search complete![/bold green] Wall clock: {wall_clock:.0f}s")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random search baseline")
    parser.add_argument("--benchmark", default="hotpotqa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--max-metric-calls", type=int, default=None)
    args = parser.parse_args()

    run_random_search(
        benchmark=args.benchmark,
        seed=args.seed,
        subset=args.subset,
        max_metric_calls=args.max_metric_calls,
    )
