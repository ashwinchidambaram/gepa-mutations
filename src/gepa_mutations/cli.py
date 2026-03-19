"""CLI entry point for gepa-mutations."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="gepa-mutations", help="Run and mutate GEPA experiments.")
console = Console()


@app.command()
def run(
    benchmark: str = typer.Argument(
        help="Benchmark to run (hotpotqa, ifbench, aime, livebench, hover, pupa)"
    ),
    seed: int = typer.Option(42, help="Random seed for the experiment"),
    seeds: Optional[str] = typer.Option(
        None, help="Comma-separated seeds for multi-seed runs (e.g. 42,123,456,789,1024)"
    ),
    subset: Optional[int] = typer.Option(None, help="Use only N examples (for quick testing)"),
    no_merge: bool = typer.Option(False, help="Run without merge strategy (for ablation)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running"),
    max_metric_calls: Optional[int] = typer.Option(
        None, help="Override rollout budget (defaults to paper budget)"
    ),
):
    """Run a GEPA experiment on a benchmark."""
    from gepa_mutations.runner.experiment import ExperimentRunner

    runner = ExperimentRunner()

    if seeds:
        seed_list = [int(s.strip()) for s in seeds.split(",")]
        results = runner.run_multi_seed(
            benchmark=benchmark,
            seeds=seed_list,
            subset=subset,
            use_merge=not no_merge,
        )
        _print_results_table(results)
    else:
        result = runner.run(
            benchmark=benchmark,
            seed=seed,
            subset=subset,
            use_merge=not no_merge,
            max_metric_calls=max_metric_calls,
            dry_run=dry_run,
        )
        if not dry_run:
            console.print(f"\nTest score: {result.test_score * 100:.2f}%")


@app.command()
def status():
    """Check status of running/completed experiments."""
    from gepa_mutations.storage.local import list_runs

    runs = list_runs()
    if not runs:
        console.print("No completed runs found.")
        return

    table = Table(title="Completed Experiment Runs")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Method", style="green")
    table.add_column("Seed", style="yellow")
    table.add_column("Path", style="dim")

    for run in runs:
        table.add_row(run["benchmark"], run["method"], str(run["seed"]), run["path"])

    console.print(table)


@app.command()
def compare(
    benchmark: Optional[str] = typer.Argument(None, help="Benchmark to compare (or all)"),
    method: str = typer.Option("gepa", help="Method to compare"),
):
    """Compare experiment results against paper baselines."""
    from gepa_mutations.config import PAPER_BASELINES
    from gepa_mutations.storage.local import list_runs, load_result

    runs = list_runs(benchmark=benchmark, method=method)
    if not runs:
        console.print("No results found to compare.")
        return

    # Group by benchmark
    by_benchmark: dict[str, list[dict]] = {}
    for run in runs:
        bm = run["benchmark"]
        by_benchmark.setdefault(bm, []).append(run)

    paper = PAPER_BASELINES.get("qwen3-8b", {})

    table = Table(title="Results vs Paper Baselines")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Seeds", style="yellow")
    table.add_column("Our Mean", style="bold green")
    table.add_column("Paper GEPA", style="blue")
    table.add_column("Paper Baseline", style="dim")
    table.add_column("Diff", style="magenta")

    for bm, bm_runs in sorted(by_benchmark.items()):
        scores = []
        seed_strs = []
        for r in bm_runs:
            result = load_result(r["benchmark"], r["seed"], r["method"])
            scores.append(result.get("test_score", 0.0))
            seed_strs.append(str(r["seed"]))

        mean_score = sum(scores) / len(scores) * 100

        paper_gepa = paper.get("gepa", {}).get(bm, 0.0)
        paper_base = paper.get("baseline", {}).get(bm, 0.0)
        diff = mean_score - paper_gepa

        table.add_row(
            bm,
            ",".join(seed_strs),
            f"{mean_score:.2f}%",
            f"{paper_gepa:.2f}%",
            f"{paper_base:.2f}%",
            f"{diff:+.2f}pp",
        )

    console.print(table)


@app.command()
def upload(
    results_dir: str = typer.Argument(help="Local results directory to upload"),
    bucket: Optional[str] = typer.Option(None, help="S3 bucket (defaults to S3_BUCKET env var)"),
):
    """Upload experiment results to S3."""
    console.print(f"Uploading {results_dir} to S3...")
    from gepa_mutations.storage.s3 import upload_results

    upload_results(results_dir, bucket)


def _print_results_table(results) -> None:
    """Print a summary table for multi-seed results."""
    table = Table(title="Multi-Seed Results")
    table.add_column("Seed", style="yellow")
    table.add_column("Test Score", style="green")
    table.add_column("Val Score", style="blue")
    table.add_column("Rollouts", style="dim")
    table.add_column("Time (s)", style="dim")

    for r in results:
        table.add_row(
            str(r.seed),
            f"{r.test_score * 100:.2f}%",
            f"{r.val_score * 100:.2f}%",
            str(r.rollout_count),
            f"{r.wall_clock_seconds:.0f}",
        )

    # Summary row
    test_scores = [r.test_score for r in results]
    mean = sum(test_scores) / len(test_scores)
    table.add_row("---", "---", "---", "---", "---")
    table.add_row("Mean", f"{mean * 100:.2f}%", "", "", "")

    console.print(table)


if __name__ == "__main__":
    app()
