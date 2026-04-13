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

from gepa.api import optimize
from rich.console import Console

from gepa_mutations.benchmarks.evaluators import get_adapter
from gepa_mutations.benchmarks.loader import load_benchmark
from gepa_mutations.config import (
    PAPER_ROLLOUTS,
    Settings,
    api_base_kwargs,
    model_id,
    model_tag as get_model_tag,
)
from gepa_mutations.metrics.collector import MetricsCollector
from gepa_mutations.metrics.tracked_lm import TrackedLM
from gepa_mutations.runner.callbacks import MetricsCallback, ProgressStreamerCallback
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
        model_id(settings),
        temperature=settings.gepa_temperature,
        max_tokens=settings.max_tokens_reflection,
        top_p=settings.gepa_top_p,
        top_k=settings.gepa_top_k,
        timeout=settings.lm_timeout,
        **api_base_kwargs(settings),
    )


def build_qa_task_lm(settings: Settings) -> LM:
    """Build the task LM for QA benchmarks (direct LiteLLM calls)."""
    return LM(
        model_id(settings),
        temperature=settings.gepa_temperature,
        max_tokens=settings.max_tokens_qa,
        top_p=settings.gepa_top_p,
        top_k=settings.gepa_top_k,
        timeout=settings.lm_timeout,
        **api_base_kwargs(settings),
    )


def evaluate_on_test(
    benchmark: str,
    best_prompt: dict[str, str],
    testset: list,
    settings: Settings,
) -> TestEvalResult:
    """Evaluate the best prompt on the test set.

    Uses the configured temperature (matching optimization conditions).
    Fix 9 (temperature=0) reverted: Qwen3-8B via OpenRouter produces
    incompatible output at temperature=0, causing all-zero QA scores.
    """
    workers = settings.test_eval_workers
    qa_lm = build_qa_task_lm(settings)
    adapter = get_adapter(benchmark, task_lm=qa_lm)
    scores = _evaluate_qa(best_prompt, testset, qa_lm, adapter, workers)

    total = len(testset)
    example_ids = [f"{benchmark}_test_{i}" for i in range(total)]
    mean_score = sum(scores) / total if total else 0.0
    return TestEvalResult(score=mean_score, example_scores=scores, example_ids=example_ids)


_MAX_CONSECUTIVE_TEST_ERRORS = 10


def _evaluate_qa(prompt: dict[str, str], testset: list, lm: LM, adapter, workers: int = 10) -> list[float]:
    """Evaluate using adapter scoring and parallel LM calls.

    Aborts if consecutive LM errors exceed _MAX_CONSECUTIVE_TEST_ERRORS to
    prevent silently scoring an entire test set as 0 when the endpoint is down.
    """
    prompt_text = prompt["system_prompt"]
    total = len(testset)

    lock = threading.Lock()
    completed = [0]
    correct_sum = [0.0]
    error_count = [0]
    consecutive_errors = [0]
    abort_exc: list[Exception] = []

    def eval_one(example):
        if abort_exc:
            return 0.0
        score = 0.0
        try:
            messages = [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": example.input},
            ]
            response = lm(messages)
            score, _ = adapter._score(example, response)
            with lock:
                consecutive_errors[0] = 0
        except Exception as e:
            score = 0.0
            with lock:
                error_count[0] += 1
                consecutive_errors[0] += 1
                if error_count[0] <= 3:
                    console.print(f"  [dim]Test eval error: {e}[/dim]")
                if consecutive_errors[0] >= _MAX_CONSECUTIVE_TEST_ERRORS and not abort_exc:
                    abort_exc.append(RuntimeError(
                        f"Aborting test evaluation: {consecutive_errors[0]} consecutive LM errors. "
                        f"Last error: {e}"
                    ))

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

    if abort_exc:
        raise abort_exc[0]

    return scores


def config_snapshot(config: MutationConfig, settings: Settings) -> dict[str, Any]:
    """Build a config dict from MutationConfig fields for result persistence."""
    return {
        "mutation_name": config.mutation_name,
        "description": config.description,
        "benchmark": config.benchmark,
        "seed": config.seed,
        "subset": config.subset,
        "model": model_id(settings),
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
    trainset = data.train[: config.subset] if config.subset is not None else data.train
    valset = data.val[: config.subset] if config.subset is not None else data.val
    testset = data.test

    # 3. Build task LM and adapter — all benchmarks use direct LM calls
    # (AIME switched from DSPy/JSONAdapter to avoid math notation parse failures)
    collector = MetricsCollector()
    qa_lm = build_qa_task_lm(settings)
    tracked_qa_lm = TrackedLM(qa_lm, collector, role="task")
    adapter = get_adapter(config.benchmark, task_lm=tracked_qa_lm)

    # 4. Build reflection LM
    reflection_lm = build_reflection_lm(settings)
    tracked_reflection_lm = TrackedLM(reflection_lm, collector, role="reflection")

    # 5. Rollout budget
    max_metric_calls = config.max_metric_calls
    if max_metric_calls is None:
        max_metric_calls = PAPER_ROLLOUTS["gepa"].get(config.benchmark, 5000)

    # 6. Run directory
    _mtag = get_model_tag(settings)
    run_dir = f"runs/{_mtag}/{config.benchmark}/{config.mutation_name}/{config.seed}/gepa_state"

    # 7. Callbacks (metrics + progress streamer)
    metrics_cb = MetricsCallback(
        benchmark=config.benchmark, seed=config.seed, run_dir=run_dir,
    )
    progress_cb = ProgressStreamerCallback(
        benchmark=config.benchmark, seed=config.seed, run_dir=run_dir,
    )

    # 8. Seed candidate
    seed_prompt = BENCHMARK_SEED_PROMPTS.get(config.benchmark, SEED_PROMPT)
    seed_candidate = {"system_prompt": seed_prompt}

    # Evaluate seed prompt on test and val sets BEFORE optimization
    console.print(f"\n[bold]Evaluating seed prompt on test set ({len(testset)} examples)...[/bold]")
    seed_test_eval = evaluate_on_test(config.benchmark, seed_candidate, testset, settings)
    console.print(f"  Seed test score: {seed_test_eval.score:.4f}")
    console.print(f"\n[bold]Evaluating seed prompt on val set ({len(data.val)} examples)...[/bold]")
    seed_val_eval = evaluate_on_test(config.benchmark, seed_candidate, data.val, settings)
    console.print(f"  Seed val score: {seed_val_eval.score:.4f}")
    seed_prompt_test_score = seed_test_eval.score
    seed_prompt_val_score = seed_val_eval.score

    # Inject rollout=0 as the first trajectory point
    collector.record_val_score(iteration=0, score=seed_prompt_val_score)

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
        reflection_lm=tracked_reflection_lm,
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
        callbacks=[metrics_cb, progress_cb],
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

    # Evaluate best prompt on train set
    console.print(f"\n[bold]Evaluating on train set ({len(trainset)} examples)...[/bold]")
    train_eval = evaluate_on_test(config.benchmark, best_prompt, trainset, settings)
    console.print(f"  Train score: {train_eval.score:.4f}")

    # Evaluate best prompt on val set (per-example scores)
    console.print(f"\n[bold]Evaluating on val set ({len(valset)} examples)...[/bold]")
    val_eval = evaluate_on_test(config.benchmark, best_prompt, valset, settings)

    wall_clock = time.time() - start_time

    # 12. Finalize token-tracked metrics and merge with GEPA callback data
    token_metrics = collector.finalize(
        test_score=test_eval.score,
        best_prompt=best_prompt,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
        model=model_id(settings),
        model_tag=_mtag,
        benchmark=config.benchmark,
        seed=config.seed,
        method=config.mutation_name,
        seed_prompt=seed_prompt,
    )
    # Combine: GEPA-specific fields take precedence, token fields fill the gap
    gepa_metrics = metrics_cb.metrics.to_dict()
    combined_metrics = {**token_metrics, **gepa_metrics}
    # Ensure token fields from collector are always present (gepa_metrics lacks them)
    for k in ["total_tokens", "task_input_tokens", "task_output_tokens",
              "reflection_input_tokens", "reflection_output_tokens",
              "model", "model_tag"]:
        combined_metrics[k] = token_metrics.get(k, 0)
    # Add train/val per-example scores
    combined_metrics["train_score"] = train_eval.score
    combined_metrics["train_example_scores"] = train_eval.example_scores
    combined_metrics["val_example_scores"] = val_eval.example_scores
    combined_metrics["val_example_ids"] = val_eval.example_ids

    # 13. Build experiment result
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
        metrics=combined_metrics,
        all_candidates=result.candidates,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
        seed_prompt_test_score=seed_prompt_test_score,
        seed_prompt_val_score=seed_prompt_val_score,
        train_score=train_eval.score,
    )

    # 14. Save results (method= routes to mutation-specific directory)
    save_result(
        benchmark=config.benchmark,
        seed=config.seed,
        result_data=exp_result.to_dict(),
        config_data=exp_result.config_snapshot,
        metrics_data=combined_metrics,
        method=config.mutation_name,
        model_tag=_mtag,
    )
    console.print(
        f"  Results saved to runs/{_mtag}/{config.benchmark}/{config.mutation_name}/{config.seed}/"
    )

    return exp_result
