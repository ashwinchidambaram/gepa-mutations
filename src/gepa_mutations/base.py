"""Phase 2 mutation infrastructure.

Provides MutationConfig (paper-default dataclass) and run_mutation() so each
mutation's runner.py is just config construction + a single function call.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import dspy
from gepa.api import optimize
from rich.console import Console

from gepa_mutations.benchmarks.evaluators import get_adapter
from gepa_mutations.benchmarks.loader import load_benchmark
from gepa_mutations.config import PAPER_ROLLOUTS, Settings
from gepa_mutations.runner.callbacks import MetricsCallback
from gepa_mutations.runner.experiment import (
    BENCHMARK_SEED_PROMPTS,
    SEED_PROMPT,
    ExperimentResult,
    LM,
    TestEvalResult,
)
from gepa_mutations.storage.local import save_result

console = Console()


# ---------------------------------------------------------------------------
# MutationConfig — paper defaults; mutations override only what they change
# ---------------------------------------------------------------------------


@dataclass
class MutationConfig:
    """Configuration for a single mutation experiment.

    All fields default to paper values. A mutation overrides only the fields it
    changes, keeping the rest identical to the Phase 1 reproduction baseline.
    """

    mutation_name: str  # e.g. "epsilon_greedy_selection"
    description: str = ""
    benchmark: str = "hotpotqa"
    seed: int = 42
    subset: int | None = None

    # optimize() parameter overrides (paper defaults)
    candidate_selection_strategy: str | Any = "pareto"
    frontier_type: str = "instance"
    module_selector: str = "round_robin"
    use_merge: bool = True
    max_merge_invocations: int = 5
    merge_val_overlap_floor: int = 5
    reflection_minibatch_size: int = 3
    reflection_prompt_template: str | dict[str, str] | None = None
    max_metric_calls: int | None = None  # None = use paper budget
    mutation_candidates: int = 1  # K value for best-of-K mutation
    use_failure_stratified_k: bool = False  # Enable failure stratification across K candidates


# ---------------------------------------------------------------------------
# Helper functions (standalone versions of ExperimentRunner methods)
# ---------------------------------------------------------------------------


def build_reflection_lm(settings: Settings) -> LM:
    """Build the reflection LM wrapper."""
    return LM(
        f"openrouter/{settings.gepa_model}",
        temperature=settings.gepa_temperature,
        max_tokens=settings.gepa_max_context,
        top_p=settings.gepa_top_p,
        top_k=settings.gepa_top_k,
    )


def build_task_lm(settings: Settings) -> dspy.LM:
    """Build the task LM (dspy.LM for math benchmarks)."""
    return dspy.LM(
        f"openrouter/{settings.gepa_model}",
        temperature=settings.gepa_temperature,
        top_p=settings.gepa_top_p,
        top_k=settings.gepa_top_k,
        max_tokens=settings.gepa_max_context,
    )


def build_qa_task_lm(settings: Settings) -> LM:
    """Build the task LM for QA benchmarks (direct LiteLLM calls)."""
    return LM(
        f"openrouter/{settings.gepa_model}",
        temperature=settings.gepa_temperature,
        max_tokens=settings.gepa_max_context,
        top_p=settings.gepa_top_p,
        top_k=settings.gepa_top_k,
    )


def _uses_dspy(benchmark: str) -> bool:
    """Check if benchmark uses dspy (AIME only) vs direct LM calls."""
    return benchmark == "aime"


def evaluate_on_test(
    benchmark: str,
    best_prompt: dict[str, str],
    testset: list,
    settings: Settings,
) -> TestEvalResult:
    """Evaluate the best prompt on the test set.

    Uses temperature=0 for deterministic test evaluation (Fix 9).
    """
    workers = settings.test_eval_workers
    if _uses_dspy(benchmark):
        # Use temperature=0 for deterministic test evaluation
        test_lm = dspy.LM(
            f"openrouter/{settings.gepa_model}",
            temperature=0,
            top_p=settings.gepa_top_p,
            top_k=settings.gepa_top_k,
            max_tokens=settings.gepa_max_context,
        )
        dspy.configure(lm=test_lm)
        scores = _evaluate_dspy(best_prompt, testset, workers)
    else:
        # Use temperature=0 for deterministic test evaluation
        qa_lm = LM(
            f"openrouter/{settings.gepa_model}",
            temperature=0,
            max_tokens=settings.gepa_max_context,
            top_p=settings.gepa_top_p,
            top_k=settings.gepa_top_k,
        )
        adapter = get_adapter(benchmark, task_lm=qa_lm)
        scores = _evaluate_qa(best_prompt, testset, qa_lm, adapter, workers)

    total = len(testset)
    example_ids = [f"{benchmark}_test_{i}" for i in range(total)]
    mean_score = sum(scores) / total if total else 0.0
    return TestEvalResult(score=mean_score, example_scores=scores, example_ids=example_ids)


def _evaluate_dspy(prompt: dict[str, str], testset: list, workers: int = 10) -> list[float]:
    """Evaluate using dspy (for math benchmarks) — parallel."""
    from gepa_mutations.benchmarks.evaluators import _math_metric, _run_llm
    from gepa_mutations.benchmarks.signatures import MathSolverSignature

    prompt_text = prompt["system_prompt"]
    total = len(testset)

    lock = threading.Lock()
    completed = [0]
    correct_sum = [0.0]
    error_count = [0]

    def eval_one(example):
        predictor = dspy.ChainOfThought(MathSolverSignature)
        try:
            prediction = _run_llm(example, prompt_text, predictor)
            score, _ = _math_metric(example, prediction)
        except Exception as e:
            score = 0.0
            with lock:
                error_count[0] += 1
                if error_count[0] <= 3:
                    console.print(f"  [dim]Test eval error: {e}[/dim]")

        with lock:
            correct_sum[0] += score
            completed[0] += 1
            done = completed[0]
            if done % 10 == 0 or done == total:
                pct = done / total * 100
                acc = correct_sum[0] / done * 100
                console.print(
                    f"  Test eval: {done}/{total} ({pct:.0f}%) | "
                    f"correct: {correct_sum[0]:.1f}/{done} ({acc:.1f}%) | errors: {error_count[0]}"
                )

        return score

    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = list(pool.map(eval_one, testset))

    return scores


def _evaluate_qa(prompt: dict[str, str], testset: list, lm: LM, adapter, workers: int = 10) -> list[float]:
    """Evaluate using adapter scoring and parallel LM calls."""
    prompt_text = prompt["system_prompt"]
    total = len(testset)

    lock = threading.Lock()
    completed = [0]
    correct_sum = [0.0]
    error_count = [0]

    def eval_one(example):
        try:
            messages = [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": example.input},
            ]
            response = lm(messages)
            score, _ = adapter._score(example, response)
        except Exception as e:
            score = 0.0
            with lock:
                error_count[0] += 1
                if error_count[0] <= 3:
                    console.print(f"  [dim]Test eval error: {e}[/dim]")

        with lock:
            correct_sum[0] += score
            completed[0] += 1
            done = completed[0]
            if done % 10 == 0 or done == total:
                pct = done / total * 100
                acc = correct_sum[0] / done * 100
                console.print(
                    f"  Test eval: {done}/{total} ({pct:.0f}%) | "
                    f"correct: {correct_sum[0]:.1f}/{done} ({acc:.1f}%) | errors: {error_count[0]}"
                )

        return score

    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = list(pool.map(eval_one, testset))

    return scores


def config_snapshot(config: MutationConfig, settings: Settings) -> dict[str, Any]:
    """Build a config dict from MutationConfig fields for result persistence."""
    return {
        "mutation_name": config.mutation_name,
        "description": config.description,
        "benchmark": config.benchmark,
        "seed": config.seed,
        "subset": config.subset,
        "model": f"openrouter/{settings.gepa_model}",
        "temperature": settings.gepa_temperature,
        "top_p": settings.gepa_top_p,
        "top_k": settings.gepa_top_k,
        "max_context": settings.gepa_max_context,
        "candidate_selection": str(config.candidate_selection_strategy),
        "frontier_type": config.frontier_type,
        "module_selector": config.module_selector,
        "use_merge": config.use_merge,
        "max_merge_invocations": config.max_merge_invocations,
        "merge_val_overlap_floor": config.merge_val_overlap_floor,
        "reflection_minibatch_size": config.reflection_minibatch_size,
        "reflection_prompt_template": config.reflection_prompt_template,
        "max_metric_calls": config.max_metric_calls,
        "rollout_budget": PAPER_ROLLOUTS["gepa"].get(config.benchmark),
    }


# ---------------------------------------------------------------------------
# run_mutation() — shared runner for all mutations
# ---------------------------------------------------------------------------


def run_mutation(config: MutationConfig, settings: Settings | None = None) -> ExperimentResult:
    """Run a single mutation experiment.

    Mirrors ExperimentRunner.run() but accepts a MutationConfig so each
    mutation's runner.py only needs to construct the config and call this.

    Args:
        config: Mutation configuration with paper defaults overridden as needed.
        settings: Environment settings (loaded from .env if not provided).

    Returns:
        ExperimentResult with scores, best prompt, and diagnostics.
    """
    settings = settings or Settings()
    start_time = time.time()

    # 1. Load benchmark data
    console.print(f"[bold]Loading benchmark: {config.benchmark}[/bold]")
    data = load_benchmark(config.benchmark, seed=0)
    console.print(
        f"  Train: {len(data.train)}, Val: {len(data.val)}, Test: {len(data.test)}"
    )

    # 2. Apply subset if specified
    trainset = data.train[: config.subset] if config.subset else data.train
    valset = data.val[: config.subset] if config.subset else data.val
    testset = data.test

    # 3. Build task LM and adapter
    if _uses_dspy(config.benchmark):
        task_lm = build_task_lm(settings)
        dspy.configure(lm=task_lm)
        adapter = get_adapter(config.benchmark)
    else:
        qa_lm = build_qa_task_lm(settings)
        adapter = get_adapter(config.benchmark, task_lm=qa_lm)

    # 4. Build reflection LM
    reflection_lm = build_reflection_lm(settings)

    # 5. Rollout budget
    max_metric_calls = config.max_metric_calls
    if max_metric_calls is None:
        max_metric_calls = PAPER_ROLLOUTS["gepa"].get(config.benchmark, 5000)

    # 6. Metrics callback
    metrics_cb = MetricsCallback(benchmark=config.benchmark, seed=config.seed)

    # 7. Seed candidate
    seed_prompt = BENCHMARK_SEED_PROMPTS.get(config.benchmark, SEED_PROMPT)
    seed_candidate = {"system_prompt": seed_prompt}

    # 8. Run directory
    run_dir = f"runs/{config.benchmark}/{config.mutation_name}/{config.seed}/gepa_state"

    console.print(f"[bold]Running mutation: {config.mutation_name}[/bold]")
    console.print(f"  Benchmark: {config.benchmark}, Seed: {config.seed}")
    console.print(f"  Rollout budget: {max_metric_calls}")
    console.print(f"  Merge: {config.use_merge}")
    if config.description:
        console.print(f"  Description: {config.description}")

    # 9. Run optimization via optimize() API
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        candidate_selection_strategy=config.candidate_selection_strategy,
        frontier_type=config.frontier_type,
        skip_perfect_score=True,
        perfect_score=1.0,
        module_selector=config.module_selector,
        use_merge=config.use_merge,
        max_merge_invocations=config.max_merge_invocations,
        merge_val_overlap_floor=config.merge_val_overlap_floor,
        reflection_minibatch_size=config.reflection_minibatch_size,
        reflection_prompt_template=config.reflection_prompt_template,
        max_metric_calls=max_metric_calls,
        cache_evaluation=True,
        seed=config.seed,
        run_dir=run_dir,
        callbacks=[metrics_cb],
        display_progress_bar=True,
        raise_on_exception=True,
    )

    # 10. Extract results
    best_prompt = result.best_candidate
    if isinstance(best_prompt, str):
        best_prompt = {"system_prompt": best_prompt}
    val_score = result.val_aggregate_scores[result.best_idx]

    console.print(f"\n[bold green]Optimization complete![/bold green]")
    console.print(f"  Best val score: {val_score:.4f}")
    console.print(f"  Candidates explored: {result.num_candidates}")

    # 11. Evaluate best prompt on test set
    console.print(f"\n[bold]Evaluating on test set ({len(testset)} examples)...[/bold]")
    test_eval = evaluate_on_test(config.benchmark, best_prompt, testset, settings)
    console.print(f"  Test score: {test_eval.score:.4f} ({test_eval.score * 100:.2f}%)")

    wall_clock = time.time() - start_time

    # 12. Build experiment result
    exp_result = ExperimentResult(
        benchmark=config.benchmark,
        seed=config.seed,
        test_score=test_eval.score,
        val_score=val_score,
        best_prompt=best_prompt,
        rollout_count=result.total_metric_calls or 0,
        config_snapshot=config_snapshot(config, settings),
        wall_clock_seconds=wall_clock,
        method=config.mutation_name,
        metrics=metrics_cb.metrics.to_dict(),
        all_candidates=result.candidates,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
    )

    # 13. Save results (method= routes to mutation-specific directory)
    save_result(
        benchmark=config.benchmark,
        seed=config.seed,
        result_data=exp_result.to_dict(),
        config_data=exp_result.config_snapshot,
        metrics_data=exp_result.metrics,
        method=config.mutation_name,
    )
    console.print(
        f"  Results saved to runs/{config.benchmark}/{config.mutation_name}/{config.seed}/"
    )

    return exp_result
