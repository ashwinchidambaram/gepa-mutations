#!/usr/bin/env python3
"""Live dashboard for the Inductive Strategy Discovery experiment."""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TIER1_METHODS = [
    "slime_mold",
    "slime_mold_prescribed8",
    "slime_mold_inductive_k5",
    "slime_mold_inductive_k5_crosspollin",
    "slime_mold_inductive_k5_refresh_expand",
]
BENCHMARKS = ["hotpotqa", "hover", "pupa", "ifbench"]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
HOTPOTQA_EXTRA_SEEDS = [2048, 4096, 8192, 16384, 32768]


def discover_results(runs_dir: Path) -> dict:
    """Walk runs_dir and collect result data per (benchmark, method, seed)."""
    results = defaultdict(lambda: defaultdict(list))  # [benchmark][method] = [seed_data,...]
    for metrics_path in runs_dir.rglob("metrics.json"):
        # Expected path: {runs_dir}/{benchmark}/{method}/{seed}/metrics.json
        parts = metrics_path.relative_to(runs_dir).parts
        if len(parts) < 4:
            continue
        benchmark, method, seed = parts[0], parts[1], parts[2]
        try:
            data = json.loads(metrics_path.read_text())
            results[benchmark][method].append({
                "seed": int(seed),
                "test_score": data.get("test_score", 0.0),
                "val_score": data.get("val_score", 0.0),
                "rollout_count": data.get("rollout_count", 0),
                "reflection_call_count": data.get("reflection_call_count", 0),
                "holdout_trajectory": data.get("holdout_trajectory", []),
            })
        except Exception as e:
            print(f"Warning: failed to parse {metrics_path}: {e}", file=sys.stderr)
    return results


def expected_seed_count(benchmark: str, method: str) -> int:
    """HotpotQA has 10 seeds for Tier 1; everything else has 5."""
    if benchmark == "hotpotqa":
        return 10
    return 5


def plot_progress_table(results: dict, output_dir: Path):
    """Grid of method x benchmark showing completed/total runs."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    rows = TIER1_METHODS
    cols = BENCHMARKS

    cell_text = []
    cell_colors = []
    for method in rows:
        row_text = []
        row_colors = []
        for bench in cols:
            n_done = len(results.get(bench, {}).get(method, []))
            n_expected = expected_seed_count(bench, method)
            row_text.append(f"{n_done}/{n_expected}")
            pct = n_done / max(n_expected, 1)
            if pct >= 0.9:
                row_colors.append("#a8e6a8")
            elif pct >= 0.5:
                row_colors.append("#fff3a0")
            else:
                row_colors.append("#ffb3b3")
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=cols,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title("Experiment Progress: Tier 1 Methods x Benchmarks", pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / "progress_table.png", dpi=120, bbox_inches="tight")
    plt.close()


def plot_money_plot(results: dict, output_dir: Path):
    """Convergence curves per benchmark with +-1 sigma shading."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    colors = plt.cm.tab10.colors

    for ax_idx, bench in enumerate(BENCHMARKS):
        ax = axes[ax_idx]
        bench_data = results.get(bench, {})
        has_data = False
        for m_idx, method in enumerate(TIER1_METHODS):
            runs = bench_data.get(method, [])
            all_trajectories = [r["holdout_trajectory"] for r in runs if r["holdout_trajectory"]]
            if not all_trajectories:
                continue

            max_iter = max(len(t) for t in all_trajectories)
            x_vals = []
            y_vals = []
            y_stds = []
            for it in range(max_iter):
                points = [t[it] for t in all_trajectories if it < len(t)]
                if not points:
                    continue
                x_vals.append(np.mean([p.get("cumulative_rollouts", 0) for p in points]))
                ys = [p.get("best_so_far", 0.0) for p in points]
                y_vals.append(np.mean(ys))
                y_stds.append(np.std(ys))

            if not x_vals:
                continue
            has_data = True
            x_arr = np.array(x_vals)
            y_arr = np.array(y_vals)
            std_arr = np.array(y_stds)
            color = colors[m_idx % len(colors)]
            ax.plot(x_arr, y_arr, label=method, color=color, marker="o", markersize=4)
            ax.fill_between(x_arr, y_arr - std_arr, y_arr + std_arr, alpha=0.15, color=color)

        if not has_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                    fontsize=14, color="gray")

        ax.set_title(bench)
        ax.set_xlabel("Cumulative rollouts")
        ax.set_ylabel("Best hold-out score")
        ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Convergence Curves: Hold-out Score vs Cumulative Rollouts (+-1 sigma across seeds)")
    plt.tight_layout()
    plt.savefig(output_dir / "money_plot.png", dpi=120, bbox_inches="tight")
    plt.close()


def plot_mean_scores(results: dict, output_dir: Path):
    """Bar chart per benchmark: mean test_score +- std across seeds."""
    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(4 * len(BENCHMARKS), 5), sharey=True)
    colors = plt.cm.tab10.colors

    for ax_idx, bench in enumerate(BENCHMARKS):
        ax = axes[ax_idx]
        bench_data = results.get(bench, {})
        means = []
        stds = []
        labels = []
        for m_idx, method in enumerate(TIER1_METHODS):
            runs = bench_data.get(method, [])
            scores = [r["test_score"] for r in runs]
            if not scores:
                means.append(0.0)
                stds.append(0.0)
            else:
                means.append(np.mean(scores))
                stds.append(np.std(scores))
            labels.append(method)

        x = np.arange(len(labels))
        bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
        ax.bar(x, means, yerr=stds, capsize=3, color=bar_colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("slime_mold_", "sm_").replace("slime_mold", "baseline")
             for m in labels],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_title(bench)
        if ax_idx == 0:
            ax.set_ylabel("Test score (mean +- std)")
        ax.grid(axis="y", alpha=0.3)

        if all(m == 0.0 for m in means):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="gray")

    fig.suptitle("Test Score per Method per Benchmark")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_scores.png", dpi=120, bbox_inches="tight")
    plt.close()


def plot_variance_tracker(results: dict, output_dir: Path):
    """Bar chart: std dev of test_score per method per benchmark. Lower = better."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10.colors

    x = np.arange(len(BENCHMARKS))
    width = 0.8 / len(TIER1_METHODS)
    has_data = False

    for m_idx, method in enumerate(TIER1_METHODS):
        stds = []
        for bench in BENCHMARKS:
            runs = results.get(bench, {}).get(method, [])
            scores = [r["test_score"] for r in runs]
            stds.append(np.std(scores) if len(scores) >= 2 else 0.0)
            if scores:
                has_data = True
        ax.bar(
            x + (m_idx - len(TIER1_METHODS) / 2) * width + width / 2,
            stds,
            width,
            label=method.replace("slime_mold_", "sm_").replace("slime_mold", "baseline"),
            color=colors[m_idx % len(colors)],
            alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS)
    ax.set_ylabel("Std dev of test_score across seeds")
    ax.set_title("Variance Tracker (lower = more consistent, the key metric)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    if not has_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                fontsize=14, color="gray")

    plt.tight_layout()
    plt.savefig(output_dir / "variance_tracker.png", dpi=120, bbox_inches="tight")
    plt.close()


def generate_all_plots(runs_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = discover_results(runs_dir)
    plot_progress_table(results, output_dir)
    plot_money_plot(results, output_dir)
    plot_mean_scores(results, output_dir)
    plot_variance_tracker(results, output_dir)
    print(f"[{time.strftime('%H:%M:%S')}] Generated 4 PNGs in {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="Live dashboard for experiment progress")
    ap.add_argument("--runs-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    if not args.runs_dir.exists():
        print(f"Error: runs-dir does not exist: {args.runs_dir}", file=sys.stderr)
        sys.exit(1)

    if args.once:
        generate_all_plots(args.runs_dir, args.output_dir)
    else:
        print(f"Live dashboard: refreshing every {args.interval}s. Ctrl-C to stop.")
        while True:
            generate_all_plots(args.runs_dir, args.output_dir)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
