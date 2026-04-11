"""Runner for PDMO (Prompt Decomposition and Modular Optimization).

Algorithm:
  Phase 1 — Decompose: Ask the reflection LM to split the seed prompt into
             4 independent modules (task_framing, reasoning_strategy,
             format_constraints, error_prevention).
  Phase 2 — Module optimization: For each module (4 total), compose the
             current module set into a system_prompt and run a mini-GEPA
             (budget = total_budget // 5). The best prompt from each
             mini-GEPA seeds the next module's run.
  Phase 3 — Composition + smoothing: Take the best prompt from the last
             module optimization and ask the reflection LM to smooth it.
             Evaluate on full valset.
  Phase 4 — Joint refinement: Run a final GEPA pass (budget = total_budget // 5)
             on the smoothed prompt.

Budget: 4*(budget/5) + (budget/5) = budget  (same total as vanilla GEPA)
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
from gepa_mutations.metrics.standalone_eval import evaluate_prompt
from gepa_mutations.metrics.tracked_lm import TrackedLM
from gepa_mutations.runner.callbacks import MetricsCallback, ProgressStreamerCallback
from gepa_mutations.runner.experiment import (
    BENCHMARK_SEED_PROMPTS,
    SEED_PROMPT,
    ExperimentResult,
)
from gepa_mutations.storage.local import save_result

from modular.composer import compose_modules, smooth_composition
from modular.decomposer import MODULE_NAMES, decompose

console = Console()


def _uses_dspy(benchmark: str) -> bool:
    """Check if benchmark uses dspy (AIME only) vs direct LM calls."""
    return benchmark == "aime"


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
        trainset: Training examples for this phase.
        valset: Validation examples.
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


def run_modular(
    benchmark: str,
    seed: int,
    subset: int | None = None,
    max_metric_calls: int | None = None,
    settings: Settings | None = None,
) -> ExperimentResult:
    """Run the PDMO (Prompt Decomposition and Modular Optimization) experiment.

    Decomposes the seed prompt into modules, optimizes each independently via
    sequential mini-GEPA runs, smooths the composition, then runs a final joint
    refinement pass.

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
    tracked_reflection = TrackedLM(reflection_lm, collector, role="reflection")

    # =========================================================================
    # 3. Seed prompt and budget
    # =========================================================================
    seed_prompt = BENCHMARK_SEED_PROMPTS.get(benchmark, SEED_PROMPT)
    total_budget = max_metric_calls or PAPER_ROLLOUTS["gepa"].get(benchmark, 5000)
    module_budget = max(1, total_budget // 5)
    refinement_budget = max(1, total_budget // 5)

    console.print(f"[bold]Running PDMO (Modular Decomposition)[/bold]")
    console.print(f"  Benchmark: {benchmark}, Seed: {seed}")
    console.print(f"  Total GEPA rollout budget: {total_budget}")
    console.print(f"  Per-module budget: {module_budget}, Refinement budget: {refinement_budget}")
    console.print(f"  Modules: {MODULE_NAMES}")

    # =========================================================================
    # 4. Phase 1: Decompose seed prompt into modules
    # =========================================================================
    console.print(f"\n[bold]Phase 1: Decomposing seed prompt into {len(MODULE_NAMES)} modules...[/bold]")
    modules = decompose(seed_prompt, trainset[:5], tracked_reflection)

    for name, text in modules.items():
        if text:
            console.print(f"  [{name}]: {len(text)} chars")
        else:
            console.print(f"  [{name}]: (empty — will use seed prompt)")

    # Compose initial prompt from decomposed modules
    current_prompt = compose_modules(modules)
    if not current_prompt.strip():
        # Fallback if decomposition produced nothing useful
        current_prompt = seed_prompt
    current_candidate = {"system_prompt": current_prompt}

    # =========================================================================
    # 5. Phase 2: Optimize each module independently (4 sequential mini-GEPA runs)
    # =========================================================================
    console.print(f"\n[bold]Phase 2: Sequential module optimization ({len(MODULE_NAMES)} mini-GEPA runs)...[/bold]")
    module_scores: list[float] = []
    module_rollouts: list[int] = []
    _mtag = get_model_tag(settings)

    for module_idx, module_name in enumerate(MODULE_NAMES):
        console.print(f"\n  [bold]Module {module_idx + 1}/{len(MODULE_NAMES)}: {module_name} (budget={module_budget})...[/bold]")

        module_run_dir = f"runs/{_mtag + '/' if _mtag else ''}{benchmark}/modular/{seed}/module_{module_idx}/gepa_state"
        module_metrics_cb = MetricsCallback(benchmark=benchmark, seed=seed, run_dir=module_run_dir)
        module_progress_cb = ProgressStreamerCallback(benchmark=benchmark, seed=seed, run_dir=module_run_dir)

        # Run mini-GEPA seeded with current composed prompt
        optimized_candidate, module_val_score, module_result = _run_gepa_phase(
            adapter=adapter,
            reflection_lm=tracked_reflection,
            trainset=trainset,
            valset=valset,
            seed_candidate=current_candidate,
            seed=seed,
            budget=module_budget,
            run_dir=module_run_dir,
            callbacks=[module_metrics_cb, module_progress_cb],
            use_merge=True,
        )
        module_rollouts_used = module_result.total_metric_calls or 0
        collector.record_rollouts(n=module_rollouts_used)
        collector.record_val_score(iteration=module_idx + 1, score=module_val_score)
        module_scores.append(module_val_score)
        module_rollouts.append(module_rollouts_used)

        console.print(f"    Val score after {module_name} optimization: {module_val_score:.4f}")
        console.print(f"    Rollouts used: {module_rollouts_used}")

        # The optimized prompt becomes the seed for the next module
        current_candidate = optimized_candidate

    # =========================================================================
    # 6. Phase 3: Composition + smoothing
    # =========================================================================
    console.print(f"\n[bold]Phase 3: Smoothing composed prompt...[/bold]")
    composed_prompt_text = current_candidate.get("system_prompt", seed_prompt)

    smoothed_text = smooth_composition(composed_prompt_text, tracked_reflection)
    smoothed_candidate = {"system_prompt": smoothed_text}

    # Evaluate smoothed prompt on full valset
    composition_val_score, _ = evaluate_prompt(adapter, valset, smoothed_candidate, collector)
    collector.record_val_score(iteration=len(MODULE_NAMES) + 1, score=composition_val_score)
    console.print(f"  Composition val score (after smoothing): {composition_val_score:.4f}")

    # =========================================================================
    # 7. Phase 4: Joint refinement GEPA
    # =========================================================================
    console.print(f"\n[bold]Phase 4: Joint refinement GEPA (budget={refinement_budget})...[/bold]")
    refinement_run_dir = f"runs/{_mtag + '/' if _mtag else ''}{benchmark}/modular/{seed}/refinement/gepa_state"
    refinement_metrics_cb = MetricsCallback(benchmark=benchmark, seed=seed, run_dir=refinement_run_dir)
    refinement_progress_cb = ProgressStreamerCallback(benchmark=benchmark, seed=seed, run_dir=refinement_run_dir)

    final_prompt, final_val_score, refinement_result = _run_gepa_phase(
        adapter=adapter,
        reflection_lm=tracked_reflection,
        trainset=trainset,
        valset=valset,
        seed_candidate=smoothed_candidate,
        seed=seed,
        budget=refinement_budget,
        run_dir=refinement_run_dir,
        callbacks=[refinement_metrics_cb, refinement_progress_cb],
        use_merge=True,
    )
    refinement_rollouts = refinement_result.total_metric_calls or 0
    collector.record_rollouts(n=refinement_rollouts)
    collector.record_val_score(iteration=len(MODULE_NAMES) + 2, score=final_val_score)

    refinement_delta = final_val_score - composition_val_score
    console.print(f"  Final val score: {final_val_score:.4f} (delta from composition: {refinement_delta:+.4f})")
    console.print(f"  Refinement rollouts used: {refinement_rollouts}")

    # =========================================================================
    # 8. Populate method-specific metrics
    # =========================================================================
    collector.method_specific.update({
        "module_scores": module_scores,
        "module_rollouts": module_rollouts,
        "composition_score": composition_val_score,
        "refinement_delta": refinement_delta,
        "modules_decomposed": MODULE_NAMES,
        "refinement_rollouts": refinement_rollouts,
        "per_module_budget": module_budget,
        "refinement_budget": refinement_budget,
    })

    # =========================================================================
    # 9. Evaluate on test set
    # =========================================================================
    console.print(f"\n[bold]Evaluating on test set ({len(testset)} examples)...[/bold]")
    test_eval = evaluate_on_test(benchmark, final_prompt, testset, settings)
    console.print(f"  Test score: {test_eval.score:.4f} ({test_eval.score * 100:.2f}%)")

    # =========================================================================
    # 10. Save results
    # =========================================================================
    metrics_data = collector.finalize(
        test_score=test_eval.score,
        best_prompt=final_prompt,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
        model=model_id(settings),
        model_tag=get_model_tag(settings),
        benchmark=benchmark,
        seed=seed,
        method="modular",
    )

    config_snap = {
        "benchmark": benchmark,
        "seed": seed,
        "subset": subset,
        "method": "modular",
        "model": model_id(settings),
        "temperature": settings.gepa_temperature,
        "top_p": settings.gepa_top_p,
        "top_k": settings.gepa_top_k,
        "rollout_budget": total_budget,
        "module_names": MODULE_NAMES,
        "per_module_budget": module_budget,
        "refinement_budget": refinement_budget,
        "use_merge": True,
    }

    exp_result = ExperimentResult(
        benchmark=benchmark,
        seed=seed,
        test_score=test_eval.score,
        val_score=final_val_score,
        best_prompt=final_prompt,
        rollout_count=collector.rollout_count,
        config_snapshot=config_snap,
        wall_clock_seconds=time.time() - start_time,
        method="modular",
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
        method="modular",
        model_tag=mtagval,
    )
    console.print(f"  Results saved to runs/{mtagval + '/' if mtagval else ''}{benchmark}/modular/{seed}/")

    return exp_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PDMO (Prompt Decomposition and Modular Optimization)"
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
    run_modular(args.benchmark, args.seed, args.subset, args.max_metric_calls)
