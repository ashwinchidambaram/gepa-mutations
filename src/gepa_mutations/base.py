"""Phase 2 mutation infrastructure.

Provides MutationConfig (paper-default dataclass) and run_mutation() so each
mutation's runner.py is just config construction + a single function call.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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


def build_reflection_lm_for_model(settings: Settings) -> LM:
    """Build the reflection LM, using a separate model when configured.

    When settings.reflection_model is set, this builds an LM using that model
    and endpoint (the 'external' analyzer level). Otherwise falls back to
    build_reflection_lm() which uses the task model (the 'self' analyzer level).
    """
    if settings.reflection_model:
        # Build kwargs for the separate reflection endpoint
        kw: dict[str, Any] = {}
        if settings.reflection_base_url:
            kw["api_base"] = settings.reflection_base_url
        elif settings.gepa_base_url:
            kw["api_base"] = settings.gepa_base_url
        api_key = settings.reflection_api_key or settings.api_key
        if api_key:
            kw["api_key"] = api_key
        ref_model_id = f"{settings.model_prefix}/{settings.reflection_model}"
        return LM(
            ref_model_id,
            temperature=settings.gepa_temperature,
            max_tokens=settings.max_tokens_reflection,
            top_p=settings.gepa_top_p,
            top_k=settings.gepa_top_k,
            timeout=settings.lm_timeout,
            **kw,
        )
    return build_reflection_lm(settings)


def build_qa_task_lm(settings: Settings) -> LM:
    """Build the task LM for QA benchmarks (direct LiteLLM calls).

    Also available as ``build_task_lm`` (legacy alias used by method runners).
    """
    return LM(
        model_id(settings),
        temperature=settings.gepa_temperature,
        max_tokens=settings.max_tokens_qa,
        top_p=settings.gepa_top_p,
        top_k=settings.gepa_top_k,
        timeout=settings.lm_timeout,
        **api_base_kwargs(settings),
    )


# Legacy alias — many method runners import this name
build_task_lm = build_qa_task_lm


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
    scores, outputs = _evaluate_qa(best_prompt, testset, qa_lm, adapter, workers)

    total = len(testset)
    example_ids = [f"{benchmark}_test_{i}" for i in range(total)]
    mean_score = sum(scores) / total if total else 0.0
    return TestEvalResult(
        score=mean_score,
        example_scores=scores,
        example_ids=example_ids,
        example_outputs=outputs,
    )


_MAX_CONSECUTIVE_TEST_ERRORS = 10

# Test-evaluation deadlines. Our orchestrator kills any subprocess that shows
# no progress for _STALL_THRESHOLD=1800s, so we give the test eval itself a
# deadline below that. If we hit it, we return partial results + raise so the
# orchestrator logs the incident rather than silently hanging and losing the
# entire run's result.json.
_TEST_EVAL_TOTAL_TIMEOUT_SECONDS = 1500   # 25 min total (orchestrator kills at 30 min)
_TEST_EVAL_IDLE_TIMEOUT_SECONDS = 300     # if nothing completes for 5 min, consider stalled


def _evaluate_qa(
    prompt: dict[str, str], testset: list, lm: LM, adapter, workers: int = 10
) -> tuple[list[float], list[str]]:
    """Evaluate using adapter scoring and parallel LM calls.

    Aborts if consecutive LM errors exceed _MAX_CONSECUTIVE_TEST_ERRORS to
    prevent silently scoring an entire test set as 0 when the endpoint is down.

    Returns:
        (scores, outputs) — per-example scores and raw model response strings.
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
            return 0.0, ""
        score = 0.0
        response = ""
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

        return score, response

    # Timeout-aware replacement for the previous `list(pool.map(eval_one, testset))`.
    # The old version would block forever if ANY single LM call hung (e.g., litellm
    # retry loop not respecting its timeout, or vLLM accepting but never returning).
    # We observed this during the exp-04 pilot on gepa and iso alike.
    #
    # New strategy:
    #   - submit all tasks, track by index so we can write results in order
    #   - wait with FIRST_COMPLETED and an idle-timeout (_TEST_EVAL_IDLE_TIMEOUT_SECONDS)
    #   - also enforce a total deadline (_TEST_EVAL_TOTAL_TIMEOUT_SECONDS)
    #   - on either timeout, cancel pending futures, raise a RuntimeError so the
    #     caller can distinguish hang-abort from normal completion
    #   - use shutdown(wait=False, cancel_futures=True) so we don't hang on stuck
    #     worker threads (they leak, but we exit cleanly)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        future_to_idx = {pool.submit(eval_one, ex): i for i, ex in enumerate(testset)}
        results: list[tuple[float, str]] = [(0.0, "")] * len(testset)
        pending = set(future_to_idx.keys())
        eval_start = time.time()

        while pending:
            elapsed = time.time() - eval_start
            time_budget = _TEST_EVAL_TOTAL_TIMEOUT_SECONDS - elapsed
            if time_budget <= 0:
                with lock:
                    if not abort_exc:
                        abort_exc.append(RuntimeError(
                            f"Test eval exceeded total timeout "
                            f"{_TEST_EVAL_TOTAL_TIMEOUT_SECONDS}s. "
                            f"Completed: {len(testset) - len(pending)}/{len(testset)}"
                        ))
                break

            wait_timeout = min(_TEST_EVAL_IDLE_TIMEOUT_SECONDS, time_budget)
            done, _ = wait(pending, timeout=wait_timeout, return_when=FIRST_COMPLETED)

            if not done:
                # Nothing finished in wait_timeout — every worker is stuck.
                with lock:
                    if not abort_exc:
                        abort_exc.append(RuntimeError(
                            f"Test eval idle: no completion in {wait_timeout:.0f}s. "
                            f"Completed: {len(testset) - len(pending)}/{len(testset)}"
                        ))
                break

            for fut in done:
                pending.discard(fut)
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result(timeout=1)  # done, should be fast
                except Exception:
                    results[idx] = (0.0, "")

        # Cancel whatever's still pending; shutdown below will not wait on them.
        for fut in pending:
            fut.cancel()
    finally:
        # Abandon stuck threads instead of blocking the main thread on their
        # completion. Stuck threads leak FDs and memory but we're about to exit
        # the subprocess anyway (either via error or clean return).
        pool.shutdown(wait=False, cancel_futures=True)

    if abort_exc:
        raise abort_exc[0]

    scores = [r[0] for r in results]
    outputs = [r[1] for r in results]
    return scores, outputs


def config_snapshot(
    config: MutationConfig, settings: Settings,
    train_size: int = 0, val_size: int = 0, test_size: int = 0
) -> dict[str, Any]:
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
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "gepa_base_url": settings.gepa_base_url,
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

    # 13. Build candidates_with_scores: pair each candidate with its val score
    candidates_with_scores = []
    for i, cand in enumerate(result.candidates[:20]):  # cap at 20
        score = result.val_aggregate_scores[i] if i < len(result.val_aggregate_scores) else None
        candidates_with_scores.append({
            "system_prompt": cand.get("system_prompt", str(cand)) if isinstance(cand, dict) else str(cand),
            "val_score": score,
        })

    # Build experiment result
    exp_result = ExperimentResult(
        benchmark=config.benchmark,
        seed=config.seed,
        test_score=test_eval.score,
        val_score=val_score,
        best_prompt=best_prompt,
        rollout_count=result.total_metric_calls or 0,
        config_snapshot=config_snapshot(
            config, settings,
            train_size=len(trainset), val_size=len(valset), test_size=len(testset)
        ),
        wall_clock_seconds=wall_clock,
        method=config.mutation_name,
        metrics=combined_metrics,
        all_candidates=candidates_with_scores,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
        seed_prompt_test_score=seed_prompt_test_score,
        seed_prompt_val_score=seed_prompt_val_score,
        train_score=train_eval.score,
    )

    # 14. Build per-example outputs list for qualitative error analysis
    test_outputs = [
        {
            "example_id": test_eval.example_ids[i],
            "input": testset[i].input if hasattr(testset[i], "input") else str(testset[i]),
            "expected": getattr(testset[i], "output", getattr(testset[i], "answer", "")),
            "output": test_eval.example_outputs[i] if i < len(test_eval.example_outputs) else "",
            "score": test_eval.example_scores[i],
        }
        for i in range(len(testset))
    ]

    # 15. Save results (method= routes to mutation-specific directory)
    save_result(
        benchmark=config.benchmark,
        seed=config.seed,
        result_data=exp_result.to_dict(),
        config_data=exp_result.config_snapshot,
        metrics_data=combined_metrics,
        method=config.mutation_name,
        model_tag=_mtag,
        test_outputs=test_outputs,
    )
    console.print(
        f"  Results saved to runs/{_mtag}/{config.benchmark}/{config.mutation_name}/{config.seed}/"
    )

    return exp_result
