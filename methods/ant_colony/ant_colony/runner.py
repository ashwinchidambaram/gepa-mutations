"""Runner for the ACPCO (Ant Colony Prompt Component Optimization) experiment.

ACPCO decomposes the prompt search space into atomic components and uses
pheromone-based reinforcement to identify the most effective combinations:

1. Generate ~50 components (5 categories × 10 each) via reflection LM.
2. Run 50 rounds of ant colony optimization:
   - Each round: 3 ants × 10 examples = 30 rollouts
   - Total: ~1500 rollouts
3. Polish the top-15 components by pheromone into a final prompt.
4. Evaluate on full val set, then test set, and save results.

Budget: ~1500 rollouts + ~6 LLM calls.
"""

from __future__ import annotations

import argparse
import random
import time

from rich.console import Console

from gepa_mutations.base import build_qa_task_lm, build_reflection_lm, evaluate_on_test
from gepa_mutations.benchmarks.evaluators import get_adapter
from gepa_mutations.benchmarks.loader import load_benchmark
from gepa_mutations.config import PAPER_ROLLOUTS, Settings, model_tag as get_model_tag
from gepa_mutations.metrics.collector import MetricsCollector
from gepa_mutations.metrics.standalone_eval import evaluate_prompt
from gepa_mutations.metrics.tracked_lm import TrackedLM
from gepa_mutations.runner.experiment import BENCHMARK_SEED_PROMPTS, SEED_PROMPT, ExperimentResult
from gepa_mutations.storage.local import save_result

from ant_colony.colony import AntColony
from ant_colony.components import CATEGORIES, ComponentLibrary

console = Console()

METHOD_NAME = "ant_colony"


def run_ant_colony(
    benchmark: str = "hotpotqa",
    seed: int = 42,
    subset: int | None = None,
    max_metric_calls: int | None = None,
    settings: Settings | None = None,
) -> ExperimentResult:
    """Run the ACPCO (Ant Colony Prompt Component Optimization) experiment.

    Args:
        benchmark: Benchmark name (hotpotqa, ifbench, hover, pupa, etc.).
        seed: Random seed for reproducibility.
        subset: If set, limit train/val to this many examples.
        max_metric_calls: Rollout budget override (defaults to paper budget).
        settings: Environment settings (loaded from .env if not provided).

    Returns:
        ExperimentResult with test/val scores, best prompt, and diagnostics.
    """
    settings = settings or Settings()
    start_time = time.time()
    rng = random.Random(seed)
    collector = MetricsCollector()

    # =========================================================================
    # 1. Load benchmark data
    # =========================================================================
    console.print(f"[bold]Loading benchmark: {benchmark}[/bold]")
    data = load_benchmark(benchmark, seed=0)
    console.print(f"  Train: {len(data.train)}, Val: {len(data.val)}, Test: {len(data.test)}")

    trainset = data.train[:subset] if subset is not None else data.train
    valset = data.val[:subset] if subset is not None else data.val
    testset = data.test

    # =========================================================================
    # 2. Build LMs and adapter
    # =========================================================================
    qa_lm = build_qa_task_lm(settings)
    tracked_task_lm = TrackedLM(qa_lm, collector, role="task")
    adapter = get_adapter(benchmark, task_lm=tracked_task_lm)

    reflection_lm = build_reflection_lm(settings)
    tracked_reflection = TrackedLM(reflection_lm, collector, role="reflection")

    # =========================================================================
    # 3. Budget
    # =========================================================================
    if max_metric_calls is None:
        max_metric_calls = PAPER_ROLLOUTS["gepa"].get(benchmark, 5000)

    # =========================================================================
    # 4. Seed prompt
    # =========================================================================
    seed_prompt = BENCHMARK_SEED_PROMPTS.get(benchmark, SEED_PROMPT)

    console.print(f"\n[bold]Running ACPCO optimization[/bold]")
    console.print(f"  Benchmark: {benchmark}, Seed: {seed}")
    console.print(f"  Train: {len(trainset)}, Val: {len(valset)}")
    console.print(f"  Rollout budget: {max_metric_calls}")

    # =========================================================================
    # 5. Generate component library (~50 components: 5 categories × 10 each)
    # =========================================================================
    console.print(f"\n[bold cyan]Generating component library...[/bold cyan]")
    library = ComponentLibrary()
    library.generate(
        reflection_lm=tracked_reflection,
        seed_prompt=seed_prompt,
        categories=CATEGORIES,
        n_per_category=10,
    )
    console.print(f"  Library size: {len(library)} components across {len(CATEGORIES)} categories")

    # =========================================================================
    # 6. Run 50 rounds of ant colony optimization
    # =========================================================================
    console.print(f"\n[bold cyan]Running 50 rounds of ant colony optimization...[/bold cyan]")
    colony = AntColony(
        library=library,
        n_ants=3,
        n_components=10,
        n_rounds=50,
        evaporation_rate=0.1,
    )
    best_raw_prompt = colony.run(
        adapter=adapter,
        trainset=trainset,
        collector=collector,
        rng=rng,
        budget=max_metric_calls,
    )
    console.print(f"  Best raw score: {colony.best_score:.4f}")
    console.print(f"  Rollouts so far: {collector.rollout_count}")

    # =========================================================================
    # 7. Polish: compose final prompt from top-15 components
    # =========================================================================
    console.print(f"\n[bold cyan]Polishing top components into final prompt...[/bold cyan]")
    top_15 = library.top_components(15)
    polished_prompt = colony.polish_prompt(
        reflection_lm=tracked_reflection,
        best_components=top_15,
    )
    if not polished_prompt.strip():
        polished_prompt = best_raw_prompt or seed_prompt

    console.print(f"  Polished prompt length: {len(polished_prompt)} chars")

    # =========================================================================
    # 8. Evaluate polished prompt on full val set
    # =========================================================================
    console.print(f"\n[bold]Evaluating on full val set ({len(valset)} examples)...[/bold]")
    best_prompt_dict = {"system_prompt": polished_prompt}
    val_score, _ = evaluate_prompt(adapter, valset, best_prompt_dict, collector)
    collector.record_val_score(iteration=50, score=val_score)
    console.print(f"  Val score: {val_score:.4f} ({val_score * 100:.2f}%)")

    # =========================================================================
    # 9. Evaluate on test set
    # =========================================================================
    console.print(f"\n[bold]Evaluating on test set ({len(testset)} examples)...[/bold]")
    test_eval = evaluate_on_test(benchmark, best_prompt_dict, testset, settings)
    console.print(f"  Test score: {test_eval.score:.4f} ({test_eval.score * 100:.2f}%)")

    wall_clock = time.time() - start_time

    # =========================================================================
    # 10. Build and save result
    # =========================================================================
    from gepa_mutations.config import model_id
    mtagval = get_model_tag(settings)

    config_snapshot = {
        "benchmark": benchmark,
        "seed": seed,
        "subset": subset,
        "method_name": METHOD_NAME,
        "model": model_id(settings),
        "max_metric_calls": max_metric_calls,
        "n_categories": len(CATEGORIES),
        "n_per_category": 10,
        "n_rounds": 50,
        "n_ants": 3,
        "n_components_per_ant": 10,
        "evaporation_rate": 0.1,
    }

    # Build method-specific metrics
    final_pheromone_ranking = [
        c.to_dict() for c in sorted(library.components, key=lambda c: -c.pheromone)
    ]
    method_specific = {
        "pheromone_history": colony.pheromone_history,
        "component_library_size": len(library),
        "final_pheromone_ranking": final_pheromone_ranking,
    }
    collector.method_specific.update(method_specific)

    metrics_data = collector.finalize(
        test_score=test_eval.score,
        best_prompt=best_prompt_dict,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
        model=model_id(settings),
        model_tag=get_model_tag(settings),
        benchmark=benchmark,
        seed=seed,
        method=METHOD_NAME,
    )

    exp_result = ExperimentResult(
        benchmark=benchmark,
        seed=seed,
        test_score=test_eval.score,
        val_score=val_score,
        best_prompt=best_prompt_dict,
        rollout_count=collector.rollout_count,
        config_snapshot=config_snapshot,
        wall_clock_seconds=wall_clock,
        method=METHOD_NAME,
        metrics=metrics_data,
        all_candidates=[{"system_prompt": polished_prompt}],
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
    )

    save_result(
        benchmark=benchmark,
        seed=seed,
        result_data=exp_result.to_dict(),
        config_data=config_snapshot,
        metrics_data=metrics_data,
        method=METHOD_NAME,
        model_tag=mtagval,
    )
    console.print(f"  Results saved to runs/{mtagval + '/' if mtagval else ''}{benchmark}/{METHOD_NAME}/{seed}/")

    console.print(f"\n[bold green]ACPCO complete![/bold green]")
    console.print(f"  Val score:  {val_score:.4f}")
    console.print(f"  Test score: {test_eval.score:.4f}")
    console.print(f"  Rollouts:   {collector.rollout_count}")
    console.print(f"  LLM calls:  {collector.reflection_call_count}")
    console.print(f"  Wall clock: {wall_clock:.1f}s")

    return exp_result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ACPCO (Ant Colony Prompt Component Optimization) experiment"
    )
    parser.add_argument(
        "--benchmark", "-b", required=True,
        help="Benchmark name (hotpotqa, ifbench, hover, pupa, aime, livebench)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--subset", "-s", type=int, default=None,
        help="Limit train/val to this many examples (for quick testing)"
    )
    parser.add_argument(
        "--max-metric-calls", "-m", type=int, default=None,
        help="Rollout budget override (defaults to paper budget)"
    )
    args = parser.parse_args()

    run_ant_colony(
        benchmark=args.benchmark,
        seed=args.seed,
        subset=args.subset,
        max_metric_calls=args.max_metric_calls,
    )
