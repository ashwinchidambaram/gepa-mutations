"""Runner for the Active Minibatch Selection (AMS) experiment.

Constructs GEPAEngine directly (bypassing optimize()) to inject the
ActiveMinibatchSampler. Replicates the setup logic from gepa.api.optimize()
and the result handling from gepa_mutations.base.run_mutation().

The key difference from vanilla GEPA: the EpochShuffledBatchSampler is replaced
with ActiveMinibatchSampler, which preferentially selects training examples
where candidates have shown high score variance (disagreement). During the
first warmup_iterations it falls back to epoch-shuffled sampling.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

import dspy
from gepa.core.data_loader import ensure_loader
from gepa.core.engine import GEPAEngine
from gepa.core.result import GEPAResult
from gepa.core.state import EvaluationCache
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.logging.logger import Logger, StdOutLogger
from gepa.proposer.merge import MergeProposer
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.component_selector import (
    AllReflectionComponentSelector,
    RoundRobinReflectionComponentSelector,
)
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.utils import CompositeStopper, FileStopper, MaxMetricCallsStopper
from rich.console import Console

from active_minibatch.callbacks import ActiveMinibatchCallback
from active_minibatch.sampler import ActiveMinibatchSampler
from gepa_mutations.base import (
    MutationConfig,
    build_qa_task_lm,
    build_reflection_lm,
    build_task_lm,
    config_snapshot,
    evaluate_on_test,
)
from gepa_mutations.benchmarks.evaluators import get_adapter
from gepa_mutations.benchmarks.loader import load_benchmark
from gepa_mutations.config import PAPER_ROLLOUTS, Settings, model_tag as get_model_tag, model_id as get_model_id
from gepa_mutations.metrics.collector import MetricsCollector
from gepa_mutations.metrics.tracked_lm import TrackedLM
from gepa_mutations.runner.callbacks import MetricsCallback, ProgressStreamerCallback
from gepa_mutations.runner.experiment import (
    BENCHMARK_SEED_PROMPTS,
    SEED_PROMPT,
    ExperimentResult,
)
from gepa_mutations.storage.local import save_result

console = Console()


def _uses_dspy(benchmark: str) -> bool:
    """Check if benchmark uses dspy (AIME only) vs direct LM calls."""
    return benchmark == "aime"


def run_active_minibatch(
    config: MutationConfig,
    warmup_iterations: int = 10,
    fallback_ratio: float = 0.3,
    settings: Settings | None = None,
) -> ExperimentResult:
    """Run an Active Minibatch Selection (AMS) mutation experiment.

    Replicates the full GEPA setup from gepa.api.optimize() but replaces
    EpochShuffledBatchSampler with ActiveMinibatchSampler.

    Args:
        config: Mutation configuration (should have mutation_name="active_minibatch").
        warmup_iterations: Number of iterations to use epoch-shuffled sampling
            before switching to active selection. Default: 10.
        fallback_ratio: Fraction of the minibatch to fill with random examples
            for exploration after warmup. Default: 0.3 (30% random, 70% active).
        settings: Environment settings (loaded from .env if not provided).

    Returns:
        ExperimentResult with scores, best prompt, and diagnostics.
    """
    settings = settings or Settings()
    start_time = time.time()
    collector = MetricsCollector()

    # =========================================================================
    # 1. Load benchmark data
    # =========================================================================
    console.print(f"[bold]Loading benchmark: {config.benchmark}[/bold]")
    data = load_benchmark(config.benchmark, seed=0)
    console.print(
        f"  Train: {len(data.train)}, Val: {len(data.val)}, Test: {len(data.test)}"
    )

    # Apply subset if specified
    trainset = data.train[: config.subset] if config.subset else data.train
    valset = data.val[: config.subset] if config.subset else data.val
    testset = data.test

    # =========================================================================
    # 2. Build LMs and adapter
    # =========================================================================
    if _uses_dspy(config.benchmark):
        task_lm = build_task_lm(settings)
        dspy.configure(lm=task_lm)
        qa_lm = build_qa_task_lm(settings)
        adapter = get_adapter(config.benchmark, task_lm=qa_lm)
    else:
        qa_lm = build_qa_task_lm(settings)
        tracked_task_lm = TrackedLM(qa_lm, collector, role="task")
        adapter = get_adapter(config.benchmark, task_lm=tracked_task_lm)

    reflection_lm = build_reflection_lm(settings)
    tracked_reflection_lm = TrackedLM(reflection_lm, collector, role="reflection")

    # =========================================================================
    # 3. Budget and stopping
    # =========================================================================
    max_metric_calls = config.max_metric_calls
    if max_metric_calls is None:
        max_metric_calls = PAPER_ROLLOUTS["gepa"].get(config.benchmark, 5000)

    # =========================================================================
    # 4. Seed candidate and run directory
    # =========================================================================
    seed_prompt = BENCHMARK_SEED_PROMPTS.get(config.benchmark, SEED_PROMPT)
    seed_candidate = {"system_prompt": seed_prompt}
    _mtag = get_model_tag(settings)
    run_dir = f"runs/{_mtag + '/' if _mtag else ''}{config.benchmark}/{config.mutation_name}/{config.seed}/gepa_state"

    # =========================================================================
    # 5. RNG
    # =========================================================================
    rng = random.Random(config.seed)

    # =========================================================================
    # 6. ActiveMinibatchSampler (replaces EpochShuffledBatchSampler)
    # =========================================================================
    batch_sampler = ActiveMinibatchSampler(
        minibatch_size=config.reflection_minibatch_size,
        rng=rng,
        warmup_iterations=warmup_iterations,
        fallback_ratio=fallback_ratio,
    )

    # =========================================================================
    # 7. Callbacks
    # =========================================================================
    metrics_cb = MetricsCallback(
        benchmark=config.benchmark, seed=config.seed, run_dir=run_dir,
    )
    progress_cb = ProgressStreamerCallback(
        benchmark=config.benchmark, seed=config.seed, run_dir=run_dir,
    )
    ams_cb = ActiveMinibatchCallback(sampler=batch_sampler)
    callbacks: list[Any] = [metrics_cb, progress_cb, ams_cb]

    console.print(f"[bold]Running mutation: {config.mutation_name}[/bold]")
    console.print(f"  Benchmark: {config.benchmark}, Seed: {config.seed}")
    console.print(f"  Rollout budget: {max_metric_calls}")
    console.print(f"  Warmup iterations: {warmup_iterations}")
    console.print(f"  Fallback ratio: {fallback_ratio}")
    console.print(f"  Merge: {config.use_merge}")
    if config.description:
        console.print(f"  Description: {config.description}")

    # =========================================================================
    # 8. Replicate gepa.api.optimize() setup
    # =========================================================================
    train_loader = ensure_loader(trainset)
    val_loader = ensure_loader(valset)

    # Stop callbacks
    stop_callbacks_list: list[Any] = []
    if run_dir is not None:
        stop_file_path = os.path.join(run_dir, "gepa.stop")
        stop_callbacks_list.append(FileStopper(stop_file_path))
    stop_callbacks_list.append(MaxMetricCallsStopper(max_metric_calls))

    if len(stop_callbacks_list) == 1:
        stop_callback = stop_callbacks_list[0]
    else:
        stop_callback = CompositeStopper(*stop_callbacks_list)

    # Logger
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        logger = Logger(os.path.join(run_dir, "run_log.txt"))
    else:
        logger = StdOutLogger()

    # Candidate selector (always pareto for paper reproduction)
    candidate_selector = ParetoCandidateSelector(rng=rng)

    # Module selector
    if config.module_selector == "round_robin":
        module_selector_instance = RoundRobinReflectionComponentSelector()
    elif config.module_selector == "all":
        module_selector_instance = AllReflectionComponentSelector()
    else:
        raise ValueError(f"Unknown module_selector: {config.module_selector}")

    # Experiment tracker
    experiment_tracker = create_experiment_tracker(
        use_wandb=False,
        use_mlflow=False,
    )

    # Evaluation cache
    evaluation_cache = EvaluationCache()

    # Validation evaluation policy
    val_evaluation_policy = FullEvaluationPolicy()

    # =========================================================================
    # 9. Create ReflectiveMutationProposer (vanilla; AMS is in the sampler)
    # =========================================================================
    reflective_proposer = ReflectiveMutationProposer(
        logger=logger,
        trainset=train_loader,
        adapter=adapter,
        candidate_selector=candidate_selector,
        module_selector=module_selector_instance,
        batch_sampler=batch_sampler,
        perfect_score=1.0,
        skip_perfect_score=True,
        experiment_tracker=experiment_tracker,
        reflection_lm=tracked_reflection_lm,
        reflection_prompt_template=config.reflection_prompt_template,
        callbacks=callbacks,
    )

    # =========================================================================
    # 10. Merge proposer (if enabled)
    # =========================================================================
    def evaluator_fn(inputs, prog):
        eval_out = adapter.evaluate(inputs, prog, capture_traces=False)
        return eval_out.outputs, eval_out.scores, eval_out.objective_scores

    merge_proposer = None
    if config.use_merge:
        merge_proposer = MergeProposer(
            logger=logger,
            valset=val_loader,
            evaluator=evaluator_fn,
            use_merge=config.use_merge,
            max_merge_invocations=config.max_merge_invocations,
            rng=rng,
            val_overlap_floor=config.merge_val_overlap_floor,
            callbacks=callbacks,
        )

    # =========================================================================
    # 11. Create GEPAEngine
    # =========================================================================
    engine = GEPAEngine(
        adapter=adapter,
        run_dir=run_dir,
        valset=val_loader,
        seed_candidate=seed_candidate,
        perfect_score=1.0,
        seed=config.seed,
        reflective_proposer=reflective_proposer,
        merge_proposer=merge_proposer,
        frontier_type=config.frontier_type,
        logger=logger,
        experiment_tracker=experiment_tracker,
        callbacks=callbacks,
        track_best_outputs=False,
        display_progress_bar=True,
        raise_on_exception=True,
        stop_callback=stop_callback,
        val_evaluation_policy=val_evaluation_policy,
        evaluation_cache=evaluation_cache,
    )

    # =========================================================================
    # 12. Run optimization
    # =========================================================================
    with experiment_tracker:
        if isinstance(logger, Logger):
            with logger:
                state = engine.run()
        else:
            state = engine.run()

    result = GEPAResult.from_state(state, run_dir=run_dir, seed=config.seed)

    # =========================================================================
    # 13. Extract results
    # =========================================================================
    best_prompt = result.best_candidate
    if isinstance(best_prompt, str):
        best_prompt = {"system_prompt": best_prompt}
    val_score = result.val_aggregate_scores[result.best_idx]

    console.print(f"\n[bold green]Optimization complete![/bold green]")
    console.print(f"  Best val score: {val_score:.4f}")
    console.print(f"  Candidates explored: {result.num_candidates}")

    # =========================================================================
    # 14. Evaluate on test set
    # =========================================================================
    console.print(f"\n[bold]Evaluating on test set ({len(testset)} examples)...[/bold]")
    test_eval = evaluate_on_test(config.benchmark, best_prompt, testset, settings)
    console.print(f"  Test score: {test_eval.score:.4f} ({test_eval.score * 100:.2f}%)")

    wall_clock = time.time() - start_time

    # =========================================================================
    # 15. Build experiment result
    # =========================================================================
    cfg_snap = config_snapshot(config, settings)
    cfg_snap["warmup_iterations"] = warmup_iterations
    cfg_snap["fallback_ratio"] = fallback_ratio

    exp_result = ExperimentResult(
        benchmark=config.benchmark,
        seed=config.seed,
        test_score=test_eval.score,
        val_score=val_score,
        best_prompt=best_prompt,
        rollout_count=result.total_metric_calls or 0,
        config_snapshot=cfg_snap,
        wall_clock_seconds=wall_clock,
        method=config.mutation_name,
        metrics=metrics_cb.metrics.to_dict(),
        all_candidates=result.candidates,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
    )

    # =========================================================================
    # 16. Save results
    # =========================================================================
    # Only merge token-tracking fields from collector to avoid overwriting
    # GEPA-tracked rollout_count, val_score, etc. with zeros.
    token_fields = {
        "total_tokens": collector.total_tokens,
        "task_input_tokens": collector.task_input_tokens,
        "task_output_tokens": collector.task_output_tokens,
        "reflection_input_tokens": collector.reflection_input_tokens,
        "reflection_output_tokens": collector.reflection_output_tokens,
        "reflection_call_count": collector.reflection_call_count,
    }
    combined_metrics = {**metrics_cb.metrics.to_dict(), **token_fields}
    combined_metrics["active_minibatch"] = ams_cb.metrics.to_dict(sampler=batch_sampler)
    # Ensure required schema fields are always present
    combined_metrics.setdefault("rollout_count", result.total_metric_calls or 0)
    combined_metrics.setdefault("wall_clock_seconds", round(wall_clock, 2))
    combined_metrics["model"] = get_model_id(settings)
    combined_metrics["model_tag"] = get_model_tag(settings)
    combined_metrics["method"] = config.mutation_name

    mtagval = get_model_tag(settings)
    save_result(
        benchmark=config.benchmark,
        seed=config.seed,
        result_data=exp_result.to_dict(),
        config_data=exp_result.config_snapshot,
        metrics_data=combined_metrics,
        method=config.mutation_name,
        model_tag=mtagval,
    )
    console.print(
        f"  Results saved to runs/{mtagval + '/' if mtagval else ''}{config.benchmark}/{config.mutation_name}/{config.seed}/"
    )

    return exp_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Active Minibatch Selection experiment")
    parser.add_argument("--benchmark", "-b", required=True, help="Benchmark name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subset", "-s", type=int, default=None, help="Use only N train/val examples")
    parser.add_argument("--max-metric-calls", "-m", type=int, default=None, help="Rollout budget override")
    parser.add_argument("--warmup-iterations", type=int, default=10, help="Iterations before active sampling kicks in")
    parser.add_argument("--fallback-ratio", type=float, default=0.3, help="Fraction of minibatch to fill randomly")
    args = parser.parse_args()

    _config = MutationConfig(
        mutation_name="active_minibatch",
        benchmark=args.benchmark,
        seed=args.seed,
        subset=args.subset,
        max_metric_calls=args.max_metric_calls,
    )
    run_active_minibatch(
        _config,
        warmup_iterations=args.warmup_iterations,
        fallback_ratio=args.fallback_ratio,
    )
