"""Runner for ESO (Ecological Succession Optimization).

Algorithm:
  Phase 0 — Difficulty estimation: evaluate the seed prompt 3x per training
             example (temperature > 0), average scores for continuous difficulty.
  Phase 1 (Pioneer): GEPA on easy examples only (15% of budget).
  Phase 2 (Shrub):   GEPA on easy + medium examples (30% of budget), seeded
                     with the pioneer prompt.
  Phase 3 (Forest):  GEPA on ALL training examples (55% of budget), seeded
                     with the shrub prompt.

Budget:
  difficulty_estimation_rollouts = 3 * len(trainset)
  gepa_budget = PAPER_ROLLOUTS["gepa"][benchmark], split 0.15 / 0.30 / 0.55

Each GEPA phase is a full GEPAEngine run in a separate run_dir.
"""

from __future__ import annotations

import argparse
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
from gepa.logging.logger import Logger
from gepa.proposer.merge import MergeProposer
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.component_selector import RoundRobinReflectionComponentSelector
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.utils import CompositeStopper, FileStopper, MaxMetricCallsStopper
from rich.console import Console

from gepa_mutations.base import (
    build_qa_task_lm,
    build_reflection_lm,
    build_task_lm,
    evaluate_on_test,
)
from gepa_mutations.benchmarks.evaluators import get_adapter
from gepa_mutations.benchmarks.loader import load_benchmark
from gepa_mutations.config import PAPER_ROLLOUTS, Settings, model_id, model_tag as get_model_tag
from gepa_mutations.metrics.collector import MetricsCollector
from gepa_mutations.metrics.tracked_lm import TrackedLM
from gepa_mutations.runner.callbacks import MetricsCallback, ProgressStreamerCallback
from gepa_mutations.runner.experiment import (
    BENCHMARK_SEED_PROMPTS,
    SEED_PROMPT,
    ExperimentResult,
)
from gepa_mutations.storage.local import save_result

from ecological_succession.difficulty import estimate_difficulty, partition_by_difficulty

console = Console()


def _uses_dspy(benchmark: str) -> bool:
    """Check if benchmark uses dspy (AIME only) vs direct LM calls."""
    return benchmark == "aime"

# Budget split across the 3 succession phases (must sum to 1.0)
_PIONEER_BUDGET_FRAC = 0.15
_SHRUB_BUDGET_FRAC = 0.30
_FOREST_BUDGET_FRAC = 0.55

# Number of difficulty estimation passes per training example
_N_DIFFICULTY_EVALS = 3


def _run_gepa_phase(
    adapter: Any,
    reflection_lm: Any,
    trainset: list,
    valset: list,
    seed_candidate: dict[str, str],
    seed: int,
    budget: int,
    run_dir: str,
    callbacks: list | None = None,
    use_merge: bool = True,
) -> tuple[dict[str, str], float, GEPAResult]:
    """Run a single GEPA optimization phase.

    Args:
        adapter: GEPAAdapter for the benchmark.
        reflection_lm: LM used for reflective mutation.
        trainset: Training examples for this phase (may be a subset).
        valset: Validation examples (full valset always used for val eval).
        seed_candidate: Starting prompt dict.
        seed: Random seed.
        budget: Max metric calls for this phase.
        run_dir: Directory for GEPA state files.
        callbacks: Optional list of GEPA callbacks.
        use_merge: Whether to enable MergeProposer.

    Returns:
        (best_prompt_dict, val_score, GEPAResult)
    """
    train_loader = ensure_loader(trainset)
    val_loader = ensure_loader(valset)
    rng = random.Random(seed)

    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(os.path.join(run_dir, "run_log.txt"))

    stop_callbacks = [MaxMetricCallsStopper(budget)]
    stop_file = os.path.join(run_dir, "gepa.stop")
    stop_callbacks.append(FileStopper(stop_file))
    stop_callback = CompositeStopper(*stop_callbacks)

    experiment_tracker = create_experiment_tracker(use_wandb=False, use_mlflow=False)
    evaluation_cache = EvaluationCache()

    batch_sampler = EpochShuffledBatchSampler(minibatch_size=3, rng=rng)
    candidate_selector = ParetoCandidateSelector(rng=rng)
    module_selector = RoundRobinReflectionComponentSelector()
    val_evaluation_policy = FullEvaluationPolicy()

    cbs = callbacks or []

    proposer = ReflectiveMutationProposer(
        logger=logger,
        trainset=train_loader,
        adapter=adapter,
        candidate_selector=candidate_selector,
        module_selector=module_selector,
        batch_sampler=batch_sampler,
        perfect_score=1.0,
        skip_perfect_score=True,
        experiment_tracker=experiment_tracker,
        reflection_lm=reflection_lm,
        callbacks=cbs,
    )

    def evaluator_fn(inputs: Any, prog: Any) -> tuple:
        eval_out = adapter.evaluate(inputs, prog, capture_traces=False)
        return eval_out.outputs, eval_out.scores, eval_out.objective_scores

    merge_proposer = None
    if use_merge:
        merge_proposer = MergeProposer(
            logger=logger,
            valset=val_loader,
            evaluator=evaluator_fn,
            use_merge=True,
            max_merge_invocations=5,
            rng=rng,
            val_overlap_floor=5,
            callbacks=cbs,
        )

    engine = GEPAEngine(
        adapter=adapter,
        run_dir=run_dir,
        valset=val_loader,
        seed_candidate=seed_candidate,
        perfect_score=1.0,
        seed=seed,
        reflective_proposer=proposer,
        merge_proposer=merge_proposer,
        frontier_type="instance",
        logger=logger,
        experiment_tracker=experiment_tracker,
        callbacks=cbs,
        track_best_outputs=False,
        display_progress_bar=True,
        raise_on_exception=True,
        stop_callback=stop_callback,
        val_evaluation_policy=val_evaluation_policy,
        evaluation_cache=evaluation_cache,
    )

    with experiment_tracker:
        with logger:
            state = engine.run()

    result = GEPAResult.from_state(state, run_dir=run_dir, seed=seed)
    best_prompt = result.best_candidate
    if isinstance(best_prompt, str):
        best_prompt = {"system_prompt": best_prompt}
    val_score = result.val_aggregate_scores[result.best_idx]

    return best_prompt, val_score, result


def run_ecological_succession(
    benchmark: str,
    seed: int,
    subset: int | None = None,
    max_metric_calls: int | None = None,
    settings: Settings | None = None,
) -> ExperimentResult:
    """Run the ESO (Ecological Succession Optimization) experiment.

    Three-phase progressive curriculum: pioneer (easy) -> shrub (easy+medium)
    -> forest (all), where each phase seeds the next with its best prompt.

    Args:
        benchmark: Benchmark name (hotpotqa, ifbench, hover, pupa, etc.).
        seed: Random seed for reproducibility.
        subset: If set, use only this many train/val examples (for testing).
        max_metric_calls: Rollout budget override (defaults to paper GEPA budget).
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
    console.print(f"[bold]Loading benchmark: {benchmark}[/bold]")
    data = load_benchmark(benchmark, seed=0)
    console.print(
        f"  Train: {len(data.train)}, Val: {len(data.val)}, Test: {len(data.test)}"
    )

    trainset = data.train[:subset] if subset else data.train
    valset = data.val[:subset] if subset else data.val
    testset = data.test

    # =========================================================================
    # 2. Build LMs and adapter
    # =========================================================================
    if _uses_dspy(benchmark):
        task_lm = build_task_lm(settings)
        dspy.configure(lm=task_lm)
        qa_lm = build_qa_task_lm(settings)
        adapter = get_adapter(benchmark, task_lm=qa_lm)
    else:
        qa_lm = build_qa_task_lm(settings)
        tracked_task_lm = TrackedLM(qa_lm, collector, role="task")
        adapter = get_adapter(benchmark, task_lm=tracked_task_lm)
    reflection_lm = build_reflection_lm(settings)
    tracked_reflection_lm = TrackedLM(reflection_lm, collector, role="reflection")

    # =========================================================================
    # 3. Seed prompt and budget
    # =========================================================================
    seed_prompt = BENCHMARK_SEED_PROMPTS.get(benchmark, SEED_PROMPT)
    seed_candidate = {"system_prompt": seed_prompt}
    total_budget = max_metric_calls or PAPER_ROLLOUTS["gepa"].get(benchmark, 5000)

    console.print(f"[bold]Running ESO (Ecological Succession)[/bold]")
    console.print(f"  Benchmark: {benchmark}, Seed: {seed}")
    console.print(f"  Total GEPA rollout budget: {total_budget}")
    console.print(f"  Phase split: {_PIONEER_BUDGET_FRAC:.0%} / {_SHRUB_BUDGET_FRAC:.0%} / {_FOREST_BUDGET_FRAC:.0%}")

    # =========================================================================
    # 4. Phase 0: Difficulty estimation
    # =========================================================================
    console.print(f"\n[bold]Phase 0: Estimating example difficulty ({_N_DIFFICULTY_EVALS} passes)...[/bold]")
    difficulty_scores = estimate_difficulty(
        adapter, trainset, seed_candidate, n_evals=_N_DIFFICULTY_EVALS
    )
    difficulty_estimation_rollouts = _N_DIFFICULTY_EVALS * len(trainset)
    collector.record_rollouts(n=difficulty_estimation_rollouts)

    easy_ids, medium_ids, hard_ids = partition_by_difficulty(difficulty_scores)
    console.print(
        f"  Easy: {len(easy_ids)}, Medium: {len(medium_ids)}, Hard: {len(hard_ids)} examples"
    )

    # =========================================================================
    # 5. Phase 1 (Pioneer): GEPA on easy subset
    # =========================================================================
    phase1_budget = max(1, int(total_budget * _PIONEER_BUDGET_FRAC))
    easy_trainset = [trainset[i] for i in easy_ids]

    console.print(f"\n[bold]Phase 1 (Pioneer): GEPA on {len(easy_trainset)} easy examples (budget={phase1_budget})...[/bold]")
    _mtag = get_model_tag(settings)
    phase1_run_dir = f"runs/{_mtag + '/' if _mtag else ''}{benchmark}/ecological_succession/{seed}/phase_1/gepa_state"

    phase1_metrics_cb = MetricsCallback(benchmark=benchmark, seed=seed, run_dir=phase1_run_dir)
    phase1_progress_cb = ProgressStreamerCallback(benchmark=benchmark, seed=seed, run_dir=phase1_run_dir)

    pioneer_prompt, pioneer_val_score, phase1_result = _run_gepa_phase(
        adapter=adapter,
        reflection_lm=tracked_reflection_lm,
        trainset=easy_trainset,
        valset=valset,
        seed_candidate=seed_candidate,
        seed=seed,
        budget=phase1_budget,
        run_dir=phase1_run_dir,
        callbacks=[phase1_metrics_cb, phase1_progress_cb],
        use_merge=True,
    )
    phase1_rollouts = phase1_result.total_metric_calls or 0
    collector.record_rollouts(n=phase1_rollouts)
    collector.record_val_score(iteration=1, score=pioneer_val_score)

    console.print(f"  Pioneer prompt val score: {pioneer_val_score:.4f}")
    console.print(f"  Phase 1 rollouts used: {phase1_rollouts}")

    # =========================================================================
    # 6. Phase 2 (Shrub): GEPA on easy + medium
    # =========================================================================
    phase2_budget = max(1, int(total_budget * _SHRUB_BUDGET_FRAC))
    shrub_trainset = [trainset[i] for i in easy_ids + medium_ids]

    console.print(f"\n[bold]Phase 2 (Shrub): GEPA on {len(shrub_trainset)} easy+medium examples (budget={phase2_budget})...[/bold]")
    phase2_run_dir = f"runs/{_mtag + '/' if _mtag else ''}{benchmark}/ecological_succession/{seed}/phase_2/gepa_state"

    phase2_metrics_cb = MetricsCallback(benchmark=benchmark, seed=seed, run_dir=phase2_run_dir)
    phase2_progress_cb = ProgressStreamerCallback(benchmark=benchmark, seed=seed, run_dir=phase2_run_dir)

    shrub_prompt, shrub_val_score, phase2_result = _run_gepa_phase(
        adapter=adapter,
        reflection_lm=tracked_reflection_lm,
        trainset=shrub_trainset,
        valset=valset,
        seed_candidate=pioneer_prompt,
        seed=seed,
        budget=phase2_budget,
        run_dir=phase2_run_dir,
        callbacks=[phase2_metrics_cb, phase2_progress_cb],
        use_merge=True,
    )
    phase2_rollouts = phase2_result.total_metric_calls or 0
    collector.record_rollouts(n=phase2_rollouts)
    collector.record_val_score(iteration=2, score=shrub_val_score)

    console.print(f"  Shrub prompt val score: {shrub_val_score:.4f}")
    console.print(f"  Phase 2 rollouts used: {phase2_rollouts}")

    # =========================================================================
    # 7. Phase 3 (Forest): GEPA on ALL training examples
    # =========================================================================
    phase3_budget = max(1, int(total_budget * _FOREST_BUDGET_FRAC))

    console.print(f"\n[bold]Phase 3 (Forest): GEPA on ALL {len(trainset)} examples (budget={phase3_budget})...[/bold]")
    phase3_run_dir = f"runs/{_mtag + '/' if _mtag else ''}{benchmark}/ecological_succession/{seed}/phase_3/gepa_state"

    phase3_metrics_cb = MetricsCallback(benchmark=benchmark, seed=seed, run_dir=phase3_run_dir)
    phase3_progress_cb = ProgressStreamerCallback(benchmark=benchmark, seed=seed, run_dir=phase3_run_dir)

    forest_prompt, forest_val_score, phase3_result = _run_gepa_phase(
        adapter=adapter,
        reflection_lm=tracked_reflection_lm,
        trainset=trainset,
        valset=valset,
        seed_candidate=shrub_prompt,
        seed=seed,
        budget=phase3_budget,
        run_dir=phase3_run_dir,
        callbacks=[phase3_metrics_cb, phase3_progress_cb],
        use_merge=True,
    )
    phase3_rollouts = phase3_result.total_metric_calls or 0
    collector.record_rollouts(n=phase3_rollouts)
    collector.record_val_score(iteration=3, score=forest_val_score)

    console.print(f"  Forest prompt val score: {forest_val_score:.4f}")
    console.print(f"  Phase 3 rollouts used: {phase3_rollouts}")

    # =========================================================================
    # 8. Populate method-specific metrics
    # =========================================================================
    collector.method_specific.update({
        "stage_scores": [pioneer_val_score, shrub_val_score, forest_val_score],
        "difficulty_distribution": {
            "easy": len(easy_ids),
            "medium": len(medium_ids),
            "hard": len(hard_ids),
        },
        "stage_budgets_used": [phase1_rollouts, phase2_rollouts, phase3_rollouts],
        "difficulty_estimation_rollouts": difficulty_estimation_rollouts,
        "phase_budget_targets": {
            "phase1": phase1_budget,
            "phase2": phase2_budget,
            "phase3": phase3_budget,
        },
    })

    # =========================================================================
    # 9. Evaluate on test set
    # =========================================================================
    console.print(f"\n[bold]Evaluating on test set ({len(testset)} examples)...[/bold]")
    test_eval = evaluate_on_test(benchmark, forest_prompt, testset, settings)
    console.print(f"  Test score: {test_eval.score:.4f} ({test_eval.score * 100:.2f}%)")

    # =========================================================================
    # 10. Save results
    # =========================================================================
    metrics_data = collector.finalize(
        test_score=test_eval.score,
        best_prompt=forest_prompt,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
        model=model_id(settings),
        model_tag=get_model_tag(settings),
        benchmark=benchmark,
        seed=seed,
        method="ecological_succession",
    )

    config_snap = {
        "benchmark": benchmark,
        "seed": seed,
        "subset": subset,
        "method": "ecological_succession",
        "model": model_id(settings),
        "temperature": settings.gepa_temperature,
        "top_p": settings.gepa_top_p,
        "top_k": settings.gepa_top_k,
        "rollout_budget": total_budget,
        "n_difficulty_evals": _N_DIFFICULTY_EVALS,
        "easy_pct": 0.20,
        "medium_pct": 0.30,
        "pioneer_budget_frac": _PIONEER_BUDGET_FRAC,
        "shrub_budget_frac": _SHRUB_BUDGET_FRAC,
        "forest_budget_frac": _FOREST_BUDGET_FRAC,
        "use_merge": True,
    }

    exp_result = ExperimentResult(
        benchmark=benchmark,
        seed=seed,
        test_score=test_eval.score,
        val_score=forest_val_score,
        best_prompt=forest_prompt,
        rollout_count=collector.rollout_count,
        config_snapshot=config_snap,
        wall_clock_seconds=time.time() - start_time,
        method="ecological_succession",
        metrics=metrics_data,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
    )

    mtagval = get_model_tag(settings)
    save_result(
        benchmark=benchmark,
        seed=seed,
        result_data=exp_result.to_dict(),
        config_data=exp_result.config_snapshot,
        metrics_data=metrics_data,
        method="ecological_succession",
        model_tag=mtagval,
    )
    console.print(f"  Results saved to runs/{mtagval + '/' if mtagval else ''}{benchmark}/ecological_succession/{seed}/")

    return exp_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ESO (Ecological Succession Optimization)"
    )
    parser.add_argument("--benchmark", "-b", required=True, help="Benchmark name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--subset", "-s", type=int, default=None, help="Use only N train/val examples"
    )
    parser.add_argument(
        "--max-metric-calls",
        "-m",
        type=int,
        default=None,
        help="Rollout budget override (defaults to paper GEPA budget)",
    )
    args = parser.parse_args()
    run_ecological_succession(args.benchmark, args.seed, args.subset, args.max_metric_calls)
