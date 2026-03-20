"""Parameter sweep for best-of-K mutation.

Runs K=[1,3,5,7] across benchmarks and seeds, collecting results
for comparison against vanilla GEPA baselines.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from best_of_k.runner import run_best_of_k
from gepa_mutations.base import MutationConfig
from gepa_mutations.config import Settings
from gepa_mutations.runner.experiment import DEFAULT_SEEDS, ExperimentResult

console = Console()

# Default sweep parameters
DEFAULT_K_VALUES = [1, 3, 5, 7]
DEFAULT_BENCHMARKS = ["hotpotqa", "ifbench", "hover", "pupa", "aime", "livebench"]


def run_sweep(
    benchmarks: list[str] | None = None,
    k_values: list[int] | None = None,
    seeds: list[int] | None = None,
    subset: int | None = None,
    max_metric_calls: int | None = None,
    use_merge: bool = True,
    settings: Settings | None = None,
) -> dict[str, dict[int, list[ExperimentResult]]]:
    """Run a parameter sweep over K values, benchmarks, and seeds.

    Args:
        benchmarks: List of benchmark names to sweep. Defaults to all.
        k_values: List of K values to test. Defaults to [1, 3, 5, 7].
        seeds: List of random seeds. Defaults to DEFAULT_SEEDS.
        subset: Optional subset size for quick testing.
        max_metric_calls: Override rollout budget. None = paper budget.
        use_merge: Whether to use merge strategy.
        settings: Environment settings.

    Returns:
        Nested dict: results[benchmark][k] = list of ExperimentResult (one per seed).
    """
    benchmarks = benchmarks or DEFAULT_BENCHMARKS
    k_values = k_values or DEFAULT_K_VALUES
    seeds = seeds or DEFAULT_SEEDS
    settings = settings or Settings()

    results: dict[str, dict[int, list[ExperimentResult]]] = {}

    total_runs = len(benchmarks) * len(k_values) * len(seeds)
    run_idx = 0

    for benchmark in benchmarks:
        results[benchmark] = {}
        for k in k_values:
            results[benchmark][k] = []
            for seed in seeds:
                run_idx += 1
                console.print(f"\n{'=' * 70}")
                console.print(
                    f"[bold]Run {run_idx}/{total_runs}: "
                    f"benchmark={benchmark}, K={k}, seed={seed}[/bold]"
                )
                console.print(f"{'=' * 70}")

                config = MutationConfig(
                    mutation_name=f"best_of_k_K{k}",
                    description=f"Best-of-K mutation with K={k}",
                    benchmark=benchmark,
                    seed=seed,
                    subset=subset,
                    use_merge=use_merge,
                    max_metric_calls=max_metric_calls,
                    mutation_candidates=k,
                )

                try:
                    result = run_best_of_k(config=config, k=k, settings=settings)
                    results[benchmark][k].append(result)
                except Exception as e:
                    console.print(f"[bold red]Run failed: {e}[/bold red]")
                    continue

    # Print summary table
    _print_sweep_summary(results, k_values)

    return results


def _print_sweep_summary(
    results: dict[str, dict[int, list[ExperimentResult]]],
    k_values: list[int],
) -> None:
    """Print a summary table of sweep results."""
    table = Table(title="Best-of-K Sweep Results (Test Scores)")
    table.add_column("Benchmark", style="bold")
    for k in k_values:
        table.add_column(f"K={k}", justify="center")

    for benchmark in sorted(results.keys()):
        row = [benchmark]
        for k in k_values:
            k_results = results[benchmark].get(k, [])
            if k_results:
                scores = [r.test_score for r in k_results]
                mean = sum(scores) / len(scores)
                row.append(f"{mean:.4f}")
            else:
                row.append("--")
        table.add_row(*row)

    console.print()
    console.print(table)


def main() -> None:
    """CLI entry point for the best-of-K sweep."""
    parser = argparse.ArgumentParser(
        description="Run best-of-K parameter sweep across benchmarks and seeds."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Benchmarks to sweep (default: all).",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=None,
        help="K values to test (default: 1 3 5 7).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds (default: 42 123 456 789 1024).",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use only this many examples per split (for quick testing).",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="Override rollout budget (default: paper budget).",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable merge strategy.",
    )

    args = parser.parse_args()

    run_sweep(
        benchmarks=args.benchmarks,
        k_values=args.k_values,
        seeds=args.seeds,
        subset=args.subset,
        max_metric_calls=args.max_metric_calls,
        use_merge=not args.no_merge,
    )


if __name__ == "__main__":
    main()
