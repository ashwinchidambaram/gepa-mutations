"""Visualization utilities for comparing results against paper baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gepa_mutations.analysis.statistics import (
    BenchmarkStats,
    ReproductionReport,
    analyze_benchmark,
    reproduction_verdict,
)
from gepa_mutations.config import PAPER_BASELINES


def plot_comparison_bar(
    report: ReproductionReport,
    output_path: str | Path = "reports/comparison.png",
) -> None:
    """Bar chart: our scores (mean + 95% CI) vs paper methods.

    Plots baseline, GRPO, MIPROv2, GEPA (paper), and our reproduction for each benchmark.
    """
    paper = PAPER_BASELINES.get("qwen3-8b", {})
    benchmarks = [s.benchmark for s in report.benchmarks]
    n = len(benchmarks)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n)
    width = 0.15

    # Paper methods
    methods = ["baseline", "grpo", "miprov2", "gepa"]
    colors = ["#d4d4d4", "#94a3b8", "#60a5fa", "#3b82f6"]
    labels = ["Baseline", "GRPO", "MIPROv2", "GEPA (paper)"]

    for i, (method, color, label) in enumerate(zip(methods, colors, labels)):
        values = [paper.get(method, {}).get(bm, 0.0) for bm in benchmarks]
        ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.8)

    # Our reproduction
    our_means = [s.mean * 100 for s in report.benchmarks]
    our_ci_lower = [s.ci_lower * 100 for s in report.benchmarks]
    our_ci_upper = [s.ci_upper * 100 for s in report.benchmarks]
    yerr_lower = [m - l for m, l in zip(our_means, our_ci_lower)]
    yerr_upper = [u - m for m, u in zip(our_means, our_ci_upper)]

    ax.bar(
        x + 4 * width,
        our_means,
        width,
        label="Ours",
        color="#ef4444",
        alpha=0.9,
        yerr=[yerr_lower, yerr_upper],
        capsize=3,
    )

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Test Score (%)")
    ax.set_title(f"Reproduction Results vs Paper — Verdict: {report.verdict}")
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(benchmarks, rotation=45, ha="right")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_convergence_curve(
    metrics_data: dict[str, Any],
    output_path: str | Path = "reports/convergence.png",
) -> None:
    """Score-vs-rollout convergence curve from metrics callback data."""
    iterations = metrics_data.get("iterations", [])
    if not iterations:
        return

    rollouts = []
    scores = []
    cumulative_calls = 0

    for it in iterations:
        cumulative_calls += it.get("metric_calls_delta", 0)
        rollouts.append(cumulative_calls)
        # Track best score seen so far
        if it.get("new_score") is not None:
            scores.append(it["new_score"])
        elif it.get("candidate_score") is not None:
            scores.append(it["candidate_score"])
        elif scores:
            scores.append(scores[-1])
        else:
            scores.append(0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rollouts, scores, "-o", markersize=2, linewidth=1)
    ax.set_xlabel("Metric Calls (Rollouts)")
    ax.set_ylabel("Best Validation Score")
    ax.set_title(
        f"Convergence — {metrics_data.get('benchmark', '')} "
        f"(seed {metrics_data.get('seed', '')})"
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_reproduction_report(report: ReproductionReport) -> None:
    """Print a formatted reproduction report to console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Verdict banner
    verdict_colors = {
        "STRONG_MATCH": "bold green",
        "ACCEPTABLE": "bold yellow",
        "FAILED": "bold red",
    }
    color = verdict_colors.get(report.verdict, "bold white")
    console.print(f"\n[{color}]Reproduction Verdict: {report.verdict}[/{color}]")
    console.print(
        f"Aggregate: {report.aggregate_mean:.2f}% vs paper {report.aggregate_paper:.2f}% "
        f"(diff: {report.aggregate_diff_pp:+.2f}pp)"
    )
    console.print(
        f"Within tolerance: {report.num_within_tolerance}/{report.total_benchmarks}\n"
    )

    # Per-benchmark table
    table = Table(title="Per-Benchmark Analysis")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Our Mean", style="green")
    table.add_column("95% CI", style="dim")
    table.add_column("Paper", style="blue")
    table.add_column("Diff (pp)", style="magenta")
    table.add_column("Tolerance", style="dim")
    table.add_column("Status")

    for s in report.benchmarks:
        status = "[green]PASS[/green]" if s.within_tolerance else "[red]FAIL[/red]"
        table.add_row(
            s.benchmark,
            f"{s.mean * 100:.2f}%",
            f"[{s.ci_lower * 100:.2f}, {s.ci_upper * 100:.2f}]",
            f"{s.paper_score:.2f}%",
            f"{s.diff_pp:+.2f}",
            f"±{s.tolerance:.1f}pp",
            status,
        )

    console.print(table)
