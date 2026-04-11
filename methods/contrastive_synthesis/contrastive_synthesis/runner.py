"""Runner for the Contrastive Synthesis Reflection (CSR) experiment.

Constructs GEPAEngine directly (bypassing optimize()) to inject
ContrastiveSynthesisProposer. Replicates the setup from
gepa-contrastive-reflection/contrastive_reflection/runner.py, but uses
ContrastiveSynthesisProposer instead of ContrastiveReflectionProposer.

The key difference: after finding contrastive pairs, calls reflection_lm once
per iteration to synthesize an abstract improvement principle, which is then
injected as side_info into the reflective dataset (instead of raw snippets).
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
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.component_selector import (
    AllReflectionComponentSelector,
    RoundRobinReflectionComponentSelector,
)
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.utils import CompositeStopper, FileStopper, MaxMetricCallsStopper
from rich.console import Console

from contrastive_reflection.config import ContrastiveReflectionConfig
from contrastive_synthesis.callbacks import ContrastiveSynthesisCallback
from contrastive_synthesis.proposer import ContrastiveSynthesisProposer
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
from gepa_mutations.config import PAPER_ROLLOUTS, Settings, model_tag as get_model_tag
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


def run_contrastive_synthesis(
    benchmark: str = "hotpotqa",
    seed: int = 42,
    subset: int | None = None,
    max_metric_calls: int | None = None,
    num_contrastive_pairs: int = 3,
    min_score_gap: float = 0.1,
    include_full_text: bool = False,
    max_snippet_length: int = 500,
    use_merge: bool = True,
    module_selector: str = "round_robin",
    reflection_minibatch_size: int = 3,
    frontier_type: str = "instance",
    max_merge_invocations: int = 5,
    merge_val_overlap_floor: int = 5,
    settings: Settings | None = None,
) -> ExperimentResult:
    """Run the Contrastive Synthesis Reflection (CSR) mutation experiment.

    Constructs the GEPAEngine directly with ContrastiveSynthesisProposer
    injected in place of the default ReflectiveMutationProposer.

    Budget is approximately GEPA + 1 extra LLM call per iteration for synthesis
    (~500 tokens per synthesis call).

    Args:
        benchmark: Benchmark name (hotpotqa, ifbench, aime, etc.).
        seed: Random seed for reproducibility.
        subset: If set, use only this many train/val examples.
        max_metric_calls: Rollout budget override (defaults to GEPA paper budget).
        num_contrastive_pairs: Number of contrastive pairs to search for.
        min_score_gap: Minimum score gap to include a contrastive pair.
        include_full_text: Include full candidate text vs snippet in pairs.
        max_snippet_length: Max characters for contrastive snippets (used in
            ContrastiveReflectionConfig; not directly used in synthesis).
        use_merge: Whether to use merge strategy.
        module_selector: Component selector strategy ("round_robin" or "all").
        reflection_minibatch_size: Minibatch size for reflection.
        frontier_type: Frontier type for GEPAEngine ("instance" or "aggregate").
        settings: Environment settings (loaded from .env if not provided).

    Returns:
        ExperimentResult with scores, best prompt, and diagnostics.
    """
    settings = settings or Settings()
    start_time = time.time()
    collector = MetricsCollector()

    mutation_name = "contrastive_synthesis"

    # =========================================================================
    # 1. Contrastive-specific config
    # =========================================================================
    contrastive_config = ContrastiveReflectionConfig(
        num_contrastive_pairs=num_contrastive_pairs,
        min_score_gap=min_score_gap,
        include_full_text=include_full_text,
        max_snippet_length=max_snippet_length,
    )

    # =========================================================================
    # 2. Load benchmark data
    # =========================================================================
    console.print(f"[bold]Loading benchmark: {benchmark}[/bold]")
    data = load_benchmark(benchmark, seed=0)
    console.print(f"  Train: {len(data.train)}, Val: {len(data.val)}, Test: {len(data.test)}")

    # Apply subset if specified
    trainset = data.train[:subset] if subset is not None else data.train
    valset = data.val[:subset] if subset is not None else data.val
    testset = data.test

    # =========================================================================
    # 3. Build LMs and adapter
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
    # 4. Rollout budget
    # =========================================================================
    if max_metric_calls is None:
        max_metric_calls = PAPER_ROLLOUTS["gepa"].get(benchmark, 5000)

    # =========================================================================
    # 5. Run directory
    # =========================================================================
    _mtag = get_model_tag(settings)
    run_dir = f"runs/{_mtag + '/' if _mtag else ''}{benchmark}/{mutation_name}/{seed}/gepa_state"

    # =========================================================================
    # 6. Callbacks
    # =========================================================================
    metrics_cb = MetricsCallback(benchmark=benchmark, seed=seed, run_dir=run_dir)
    progress_cb = ProgressStreamerCallback(benchmark=benchmark, seed=seed, run_dir=run_dir)
    synthesis_cb = ContrastiveSynthesisCallback()
    callbacks: list[Any] = [metrics_cb, progress_cb, synthesis_cb]

    # =========================================================================
    # 7. Seed candidate
    # =========================================================================
    seed_prompt = BENCHMARK_SEED_PROMPTS.get(benchmark, SEED_PROMPT)
    seed_candidate = {"system_prompt": seed_prompt}

    console.print(f"[bold]Running mutation: {mutation_name}[/bold]")
    console.print(f"  Benchmark: {benchmark}, Seed: {seed}")
    console.print(f"  Rollout budget: {max_metric_calls}")
    console.print(f"  Merge: {use_merge}")
    console.print(f"  Contrastive pairs: {num_contrastive_pairs}, min gap: {min_score_gap}")
    console.print(f"  Synthesis: 1 extra reflection LM call per iteration")

    # =========================================================================
    # 8. Construct engine components (replicating gepa/src/gepa/api.py)
    # =========================================================================
    train_loader = ensure_loader(trainset)
    val_loader = ensure_loader(valset)

    rng = random.Random(seed)

    # Candidate selector (paper default: pareto)
    candidate_selector = ParetoCandidateSelector(rng=rng)

    # Module selector
    if module_selector == "all":
        module_selector_instance = AllReflectionComponentSelector()
    else:
        module_selector_instance = RoundRobinReflectionComponentSelector()

    # Batch sampler (paper default: epoch_shuffled, minibatch_size=3)
    batch_sampler = EpochShuffledBatchSampler(
        minibatch_size=reflection_minibatch_size, rng=rng
    )

    # Experiment tracker
    experiment_tracker = create_experiment_tracker(
        use_wandb=False,
        use_mlflow=False,
    )

    # Logger
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(os.path.join(run_dir, "run_log.txt"))

    # Stop callbacks
    stop_callbacks_list: list[Any] = []
    stop_file_path = os.path.join(run_dir, "gepa.stop")
    stop_callbacks_list.append(FileStopper(stop_file_path))
    stop_callbacks_list.append(MaxMetricCallsStopper(max_metric_calls))

    if len(stop_callbacks_list) == 1:
        stop_callback = stop_callbacks_list[0]
    else:
        stop_callback = CompositeStopper(*stop_callbacks_list)

    # Evaluation cache
    evaluation_cache: EvaluationCache = EvaluationCache()

    # Validation evaluation policy
    val_evaluation_policy = FullEvaluationPolicy()

    # =========================================================================
    # 9. Build ContrastiveSynthesisProposer
    # =========================================================================
    reflective_proposer = ContrastiveSynthesisProposer(
        logger=logger,
        trainset=train_loader,
        adapter=adapter,
        candidate_selector=candidate_selector,
        module_selector=module_selector_instance,
        batch_sampler=batch_sampler,
        perfect_score=1.0,
        skip_perfect_score=True,
        experiment_tracker=experiment_tracker,
        contrastive_config=contrastive_config,
        reflection_lm=tracked_reflection_lm,
        reflection_prompt_template=None,
        custom_candidate_proposer=None,
        callbacks=callbacks,
    )

    # =========================================================================
    # 10. Build evaluator for merge
    # =========================================================================
    def evaluator_fn(inputs, prog):
        eval_out = adapter.evaluate(inputs, prog, capture_traces=False)
        return eval_out.outputs, eval_out.scores, eval_out.objective_scores

    # =========================================================================
    # 11. MergeProposer if use_merge=True
    # =========================================================================
    merge_proposer: MergeProposer | None = None
    if use_merge:
        merge_proposer = MergeProposer(
            logger=logger,
            valset=val_loader,
            evaluator=evaluator_fn,
            use_merge=use_merge,
            max_merge_invocations=max_merge_invocations,
            rng=rng,
            val_overlap_floor=merge_val_overlap_floor,
            callbacks=callbacks,
        )

    # =========================================================================
    # 12. Build GEPAEngine
    # =========================================================================
    engine = GEPAEngine(
        adapter=adapter,
        run_dir=run_dir,
        valset=val_loader,
        seed_candidate=seed_candidate,
        perfect_score=1.0,
        seed=seed,
        reflective_proposer=reflective_proposer,
        merge_proposer=merge_proposer,
        frontier_type=frontier_type,
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
    # 13. Run optimization
    # =========================================================================
    with experiment_tracker:
        with logger:
            state = engine.run()

    result = GEPAResult.from_state(state, run_dir=run_dir, seed=seed)

    # =========================================================================
    # 14. Extract results
    # =========================================================================
    best_prompt = result.best_candidate
    if isinstance(best_prompt, str):
        best_prompt = {"system_prompt": best_prompt}
    val_score = result.val_aggregate_scores[result.best_idx]

    console.print(f"\n[bold green]Optimization complete![/bold green]")
    console.print(f"  Best val score: {val_score:.4f}")
    console.print(f"  Candidates explored: {result.num_candidates}")

    # =========================================================================
    # 15. Evaluate best prompt on test set
    # =========================================================================
    console.print(f"\n[bold]Evaluating on test set ({len(testset)} examples)...[/bold]")
    test_eval = evaluate_on_test(benchmark, best_prompt, testset, settings)
    console.print(f"  Test score: {test_eval.score:.4f} ({test_eval.score * 100:.2f}%)")

    wall_clock = time.time() - start_time

    # =========================================================================
    # 16. Build MutationConfig for config snapshot
    # =========================================================================
    mutation_config = MutationConfig(
        mutation_name=mutation_name,
        description=(
            "Distill contrastive pairs into abstract principles via synthesis, "
            "then inject principle as side_info into reflective dataset"
        ),
        benchmark=benchmark,
        seed=seed,
        subset=subset,
        max_metric_calls=max_metric_calls,
        use_merge=use_merge,
    )

    config_data = config_snapshot(mutation_config, settings)
    config_data["contrastive_config"] = contrastive_config.model_dump()

    # =========================================================================
    # 17. Build experiment result
    # =========================================================================
    exp_result = ExperimentResult(
        benchmark=benchmark,
        seed=seed,
        test_score=test_eval.score,
        val_score=val_score,
        best_prompt=best_prompt,
        rollout_count=result.total_metric_calls or 0,
        config_snapshot=config_data,
        wall_clock_seconds=wall_clock,
        method=mutation_name,
        metrics=metrics_cb.metrics.to_dict(),
        all_candidates=result.candidates,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
    )

    # =========================================================================
    # 18. Save results
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
    combined_metrics["contrastive_synthesis"] = synthesis_cb.to_dict()

    mtagval = get_model_tag(settings)
    save_result(
        benchmark=benchmark,
        seed=seed,
        result_data=exp_result.to_dict(),
        config_data=exp_result.config_snapshot,
        metrics_data=combined_metrics,
        method=mutation_name,
        model_tag=mtagval,
    )
    console.print(f"  Results saved to runs/{mtagval + '/' if mtagval else ''}{benchmark}/{mutation_name}/{seed}/")

    return exp_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Contrastive Synthesis Reflection (CSR) experiment"
    )
    parser.add_argument("--benchmark", "-b", required=True, help="Benchmark name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subset", "-s", type=int, default=None, help="Use only N train/val examples")
    parser.add_argument("--max-metric-calls", "-m", type=int, default=None, help="Rollout budget override")
    parser.add_argument("--num-contrastive-pairs", "-k", type=int, default=3, help="Number of contrastive pairs")
    parser.add_argument("--min-score-gap", type=float, default=0.1, help="Minimum score gap for contrastive pairs")
    parser.add_argument("--use-merge", action=argparse.BooleanOptionalAction, default=True, help="Use merge strategy")
    args = parser.parse_args()

    run_contrastive_synthesis(
        benchmark=args.benchmark,
        seed=args.seed,
        subset=args.subset,
        max_metric_calls=args.max_metric_calls,
        num_contrastive_pairs=args.num_contrastive_pairs,
        min_score_gap=args.min_score_gap,
        use_merge=args.use_merge,
    )
