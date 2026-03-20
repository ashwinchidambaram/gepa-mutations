"""Runner for the best-of-K mutation experiment.

Constructs GEPAEngine directly (bypassing optimize()) to inject the
BestOfKProposer. Replicates the setup logic from gepa.api.optimize()
and the result handling from gepa_mutations.base.run_mutation().
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Sequence
from typing import Any

import dspy
from gepa.core.data_loader import ensure_loader
from gepa.core.engine import GEPAEngine
from gepa.core.result import GEPAResult
from gepa.core.state import EvaluationCache
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.logging.logger import Logger, StdOutLogger
from gepa.proposer.merge import MergeProposer
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.component_selector import (
    AllReflectionComponentSelector,
    RoundRobinReflectionComponentSelector,
)
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.utils import CompositeStopper, FileStopper, MaxMetricCallsStopper
from rich.console import Console

from best_of_k.callbacks import BestOfKMetricsCallback
from best_of_k.proposer import BestOfKProposer
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
from gepa_mutations.config import PAPER_ROLLOUTS, Settings
from gepa_mutations.runner.callbacks import MetricsCallback
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


def run_best_of_k(
    config: MutationConfig,
    k: int | None = None,
    settings: Settings | None = None,
) -> ExperimentResult:
    """Run a best-of-K mutation experiment.

    Replicates the full GEPA setup from gepa.api.optimize() but injects
    a BestOfKProposer instead of the vanilla ReflectiveMutationProposer.

    Args:
        config: Mutation configuration (should have mutation_name="best_of_k").
        k: Override for mutation_candidates (K value). If None, uses
           config.mutation_candidates.
        settings: Environment settings (loaded from .env if not provided).

    Returns:
        ExperimentResult with scores, best prompt, and diagnostics.
    """
    settings = settings or Settings()
    start_time = time.time()

    # Resolve K value
    mutation_candidates = k if k is not None else config.mutation_candidates

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
        adapter = get_adapter(config.benchmark)
    else:
        qa_lm = build_qa_task_lm(settings)
        adapter = get_adapter(config.benchmark, task_lm=qa_lm)

    reflection_lm = build_reflection_lm(settings)

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
    run_dir = f"runs/{config.benchmark}/{config.mutation_name}/{config.seed}/gepa_state"

    # =========================================================================
    # 5. Callbacks
    # =========================================================================
    metrics_cb = MetricsCallback(benchmark=config.benchmark, seed=config.seed)
    best_of_k_cb = BestOfKMetricsCallback()
    callbacks = [metrics_cb, best_of_k_cb]

    console.print(f"[bold]Running mutation: {config.mutation_name}[/bold]")
    console.print(f"  Benchmark: {config.benchmark}, Seed: {config.seed}")
    console.print(f"  K (mutation_candidates): {mutation_candidates}")
    console.print(f"  Rollout budget: {max_metric_calls}")
    console.print(f"  Merge: {config.use_merge}")
    if config.description:
        console.print(f"  Description: {config.description}")

    # =========================================================================
    # 6. Replicate gepa.api.optimize() setup
    # =========================================================================
    train_loader = ensure_loader(trainset)
    val_loader = ensure_loader(valset)

    # Stop callbacks
    stop_callbacks_list = []
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

    # RNG
    rng = random.Random(config.seed)

    # Candidate selector (always pareto for paper reproduction)
    candidate_selector = ParetoCandidateSelector(rng=rng)

    # Module selector
    if config.module_selector == "round_robin":
        module_selector_instance = RoundRobinReflectionComponentSelector()
    elif config.module_selector == "all":
        module_selector_instance = AllReflectionComponentSelector()
    else:
        raise ValueError(f"Unknown module_selector: {config.module_selector}")

    # Batch sampler
    batch_sampler = EpochShuffledBatchSampler(
        minibatch_size=config.reflection_minibatch_size, rng=rng
    )

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
    # 7. Create BestOfKProposer (instead of ReflectiveMutationProposer)
    # =========================================================================
    reflective_proposer = BestOfKProposer(
        logger=logger,
        trainset=train_loader,
        adapter=adapter,
        candidate_selector=candidate_selector,
        module_selector=module_selector_instance,
        batch_sampler=batch_sampler,
        perfect_score=1.0,
        skip_perfect_score=True,
        experiment_tracker=experiment_tracker,
        mutation_candidates=mutation_candidates,
        reflection_lm=reflection_lm,
        reflection_prompt_template=config.reflection_prompt_template,
        callbacks=callbacks,
    )

    # =========================================================================
    # 8. Merge proposer (if enabled)
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
    # 9. Create GEPAEngine
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
    # 10. Run optimization
    # =========================================================================
    with experiment_tracker:
        if isinstance(logger, Logger):
            with logger:
                state = engine.run()
        else:
            state = engine.run()

    result = GEPAResult.from_state(state, run_dir=run_dir, seed=config.seed)

    # =========================================================================
    # 11. Extract results
    # =========================================================================
    best_prompt = result.best_candidate
    if isinstance(best_prompt, str):
        best_prompt = {"system_prompt": best_prompt}
    val_score = result.val_aggregate_scores[result.best_idx]

    console.print(f"\n[bold green]Optimization complete![/bold green]")
    console.print(f"  Best val score: {val_score:.4f}")
    console.print(f"  Candidates explored: {result.num_candidates}")

    # =========================================================================
    # 12. Evaluate on test set
    # =========================================================================
    console.print(f"\n[bold]Evaluating on test set ({len(testset)} examples)...[/bold]")
    test_eval = evaluate_on_test(config.benchmark, best_prompt, testset, settings)
    console.print(f"  Test score: {test_eval.score:.4f} ({test_eval.score * 100:.2f}%)")

    wall_clock = time.time() - start_time

    # =========================================================================
    # 13. Build experiment result
    # =========================================================================
    # Augment config snapshot with K value
    cfg_snap = config_snapshot(config, settings)
    cfg_snap["mutation_candidates"] = mutation_candidates

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
    # 14. Save results
    # =========================================================================
    # Combine standard metrics with best-of-K specific metrics
    combined_metrics = metrics_cb.metrics.to_dict()
    combined_metrics["best_of_k"] = best_of_k_cb.metrics.to_dict()

    save_result(
        benchmark=config.benchmark,
        seed=config.seed,
        result_data=exp_result.to_dict(),
        config_data=exp_result.config_snapshot,
        metrics_data=combined_metrics,
        method=config.mutation_name,
    )
    console.print(
        f"  Results saved to runs/{config.benchmark}/{config.mutation_name}/{config.seed}/"
    )

    return exp_result
