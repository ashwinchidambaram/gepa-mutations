#!/usr/bin/env python3
"""
Comprehensive visualization of GEPA-mutations experimental results.

Loads all result.json, metrics.json, and run_events.jsonl across:
  - runs/ (qwen3-1.7b, qwen3-8b)
  - runs_archive_slurm_sweep/archive/ (qwen3-27b-awq, qwen3-4b, ifbench archive)

Generates:
  1. Convergence curves (best_val_score vs rollouts) per benchmark, grouped by method
  2. Final test score comparison bar charts per model
  3. Summary heatmaps of test scores (method x benchmark)
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
RUNS_DIRS = [
    PROJECT / "runs",
    PROJECT / "runs_archive_slurm_sweep" / "archive",
]
OUT = PROJECT / "analysis"
OUT.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]

# Consistent ordering and colors
METHOD_ORDER = [
    "baseline",
    "gepa",
    "contrastive_reflection",
    "synaptic_pruning",
    "slime_mold",
    "tournament",
    "best_of_k_K3",
    "failure_stratified_k_K3",
    "contrastive_synthesis",
    "ecological_succession",
]
METHOD_COLORS = {
    "baseline": "#888888",
    "gepa": "#1f77b4",
    "contrastive_reflection": "#2ca02c",
    "synaptic_pruning": "#d62728",
    "slime_mold": "#9467bd",
    "tournament": "#ff7f0e",
    "best_of_k_K3": "#e377c2",
    "failure_stratified_k_K3": "#8c564b",
    "contrastive_synthesis": "#17becf",
    "ecological_succession": "#bcbd22",
}
METHOD_LABELS = {
    "baseline": "Baseline",
    "gepa": "GEPA",
    "contrastive_reflection": "Contrastive Refl.",
    "synaptic_pruning": "Synaptic Pruning",
    "slime_mold": "Slime Mold",
    "tournament": "Tournament",
    "best_of_k_K3": "Best-of-K (K=3)",
    "failure_stratified_k_K3": "Fail-Strat K (K=3)",
    "contrastive_synthesis": "Contrastive Synth.",
    "ecological_succession": "Ecological Succ.",
}

BENCHMARK_ORDER = ["aime", "hover", "hotpotqa", "ifbench", "livebench", "pupa"]


# ── Data Loading ─────────────────────────────────────────────────────────────

def infer_structure(root_dir: Path, rj_path: Path):
    """Infer (model, benchmark, method, seed) from path structure."""
    parts = rj_path.relative_to(root_dir).parts
    # Handle archive/ifbench/synaptic_pruning/42/... (no model prefix)
    if str(root_dir).endswith("archive") and parts[0] in ("ifbench", "hover", "aime", "livebench", "hotpotqa", "pupa"):
        return "unknown", parts[0], parts[1], parts[2]
    # Normal: model/benchmark/method/seed/...
    return parts[0], parts[1], parts[2], parts[3]


def load_all_results():
    """Load all result.json files into a structured dict."""
    results = {}  # (model, benchmark, method, seed) -> result_dict
    for root_dir in RUNS_DIRS:
        if not root_dir.exists():
            continue
        for rj in root_dir.rglob("result.json"):
            try:
                d = json.load(open(rj))
                model, bm, method, seed = infer_structure(root_dir, rj)
                results[(model, bm, method, str(seed))] = d
                # Also store path for finding related files
                d["_path"] = rj.parent
                d["_model"] = model
            except Exception as e:
                pass
    return results


def load_convergence_curve(run_dir: Path, method: str):
    """
    Load convergence data: (rollouts_list, best_val_scores_list).
    Sources (tried in order):
      1. run_events.jsonl (gepa, contrastive_reflection)
      2. metrics.json rollout_trajectory + best_val_trajectory
      3. metrics.json iterations array (archive format)
    """
    # Try run_events.jsonl first
    events_file = run_dir / "gepa_state" / "run_events.jsonl"
    if events_file.exists():
        rollouts, scores = [], []
        try:
            for line in events_file.read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    if e.get("event") == "iteration_end":
                        rollouts.append(e["rollouts"])
                        scores.append(e["best_val_score"])
            if rollouts:
                return rollouts, scores
        except Exception:
            pass

    # Try metrics.json trajectories (tournament, slime_mold, synaptic_pruning)
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        try:
            d = json.load(open(metrics_file))
            rt = d.get("rollout_trajectory", [])
            bt = d.get("best_val_trajectory", [])
            if rt and bt and len(rt) > 1:
                rollouts = [x[1] for x in rt]
                scores = [x[1] for x in bt]
                if rollouts:
                    return rollouts, scores
        except Exception:
            pass

    # Try metrics.json or metrics_checkpoint.json iterations (archive gepa format)
    for fname in ["metrics_checkpoint.json", "metrics.json"]:
        mf = run_dir / "gepa_state" / fname
        if not mf.exists():
            mf = run_dir / fname
        if mf.exists():
            try:
                d = json.load(open(mf))
                iters = d.get("iterations", [])
                if iters:
                    rollouts = [it["metric_calls_used"] for it in iters]
                    # best_val_score = running max of candidate_score
                    best = 0
                    scores = []
                    for it in iters:
                        sc = it.get("new_score") if it.get("proposal_accepted") else it.get("candidate_score", 0)
                        if sc is not None:
                            best = max(best, sc)
                        scores.append(best)
                    if rollouts:
                        return rollouts, scores
            except Exception:
                pass

    return None, None


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_convergence_curves(all_results: dict):
    """
    Plot convergence curves: best_val_score vs cumulative rollouts.
    One figure per model, subplots per benchmark, lines per method.
    """
    # Group by model
    models = sorted(set(k[0] for k in all_results))

    for model in models:
        # Get benchmarks for this model
        benchmarks = sorted(set(k[1] for k in all_results if k[0] == model))
        benchmarks = [b for b in BENCHMARK_ORDER if b in benchmarks]
        if not benchmarks:
            continue

        methods_present = sorted(set(k[2] for k in all_results if k[0] == model and k[2] != "baseline"),
                                  key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)

        if not methods_present:
            continue

        ncols = min(3, len(benchmarks))
        nrows = (len(benchmarks) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
        fig.suptitle(f"Convergence Curves — {model}", fontsize=16, fontweight="bold", y=1.02)

        for idx, bm in enumerate(benchmarks):
            ax = axes[idx // ncols][idx % ncols]
            ax.set_title(bm, fontsize=13)
            ax.set_xlabel("Cumulative Rollouts (LLM calls)")
            ax.set_ylabel("Best Val Score")
            ax.grid(True, alpha=0.3)

            # Draw baseline as horizontal line
            baseline_scores = []
            for seed in SEEDS:
                key = (model, bm, "baseline", str(seed))
                if key in all_results:
                    vs = all_results[key].get("val_score")
                    if vs is not None:
                        baseline_scores.append(vs)
            if baseline_scores:
                mean_bl = np.mean(baseline_scores)
                ax.axhline(mean_bl, color=METHOD_COLORS["baseline"], linestyle="--",
                           linewidth=1.5, alpha=0.7, label=f"Baseline ({mean_bl:.3f})")

            for method in methods_present:
                all_rollouts = []
                all_scores = []
                for seed in SEEDS:
                    key = (model, bm, method, str(seed))
                    if key not in all_results:
                        continue
                    run_dir = all_results[key]["_path"]
                    rollouts, scores = load_convergence_curve(run_dir, method)
                    if rollouts is not None and scores is not None and len(rollouts) > 1:
                        all_rollouts.append(rollouts)
                        all_scores.append(scores)

                if not all_rollouts:
                    continue

                color = METHOD_COLORS.get(method, "gray")
                label = METHOD_LABELS.get(method, method)

                if len(all_rollouts) == 1:
                    ax.plot(all_rollouts[0], all_scores[0], color=color, linewidth=1.5,
                            label=label, alpha=0.8)
                else:
                    # Interpolate to common grid and show mean +/- std
                    max_r = max(c[-1] for c in all_rollouts)
                    grid = np.linspace(0, max_r, 300)
                    interp = []
                    for r, s in zip(all_rollouts, all_scores):
                        interp.append(np.interp(grid, r, s))
                    mean_curve = np.mean(interp, axis=0)
                    std_curve = np.std(interp, axis=0)
                    ax.plot(grid, mean_curve, color=color, linewidth=2, label=f"{label} (n={len(all_rollouts)})")
                    ax.fill_between(grid, mean_curve - std_curve, mean_curve + std_curve,
                                    alpha=0.15, color=color)

            ax.legend(fontsize=7, loc="lower right")

        # Hide unused axes
        for idx in range(len(benchmarks), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        outpath = OUT / f"convergence_{model}.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {outpath}")


def plot_final_scores(all_results: dict):
    """
    Bar chart of final test scores: grouped by benchmark, bars per method.
    One figure per model. Error bars from seed variance.
    """
    models = sorted(set(k[0] for k in all_results))

    for model in models:
        benchmarks = sorted(set(k[1] for k in all_results if k[0] == model))
        benchmarks = [b for b in BENCHMARK_ORDER if b in benchmarks]
        methods_present = sorted(set(k[2] for k in all_results if k[0] == model),
                                  key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
        if not benchmarks or not methods_present:
            continue

        fig, ax = plt.subplots(figsize=(max(10, 1.5 * len(benchmarks) * len(methods_present)), 6))
        fig.suptitle(f"Final Test Scores — {model}", fontsize=16, fontweight="bold")

        x = np.arange(len(benchmarks))
        n_methods = len(methods_present)
        bar_width = 0.8 / n_methods

        for i, method in enumerate(methods_present):
            means, stds = [], []
            for bm in benchmarks:
                scores = []
                for seed in SEEDS:
                    key = (model, bm, method, str(seed))
                    if key in all_results and all_results[key].get("test_score") is not None:
                        scores.append(all_results[key]["test_score"])
                if scores:
                    means.append(np.mean(scores))
                    stds.append(np.std(scores))
                else:
                    means.append(0)
                    stds.append(0)

            color = METHOD_COLORS.get(method, "gray")
            label = METHOD_LABELS.get(method, method)
            offset = (i - n_methods / 2 + 0.5) * bar_width
            ax.bar(x + offset, means, bar_width * 0.9, yerr=stds, label=label,
                   color=color, alpha=0.85, capsize=3, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, fontsize=11)
        ax.set_ylabel("Test Score", fontsize=12)
        ax.set_ylim(0, min(1.0, max(
            all_results[k].get("test_score", 0) or 0
            for k in all_results if k[0] == model
        ) * 1.25))
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        outpath = OUT / f"test_scores_{model}.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {outpath}")


def plot_heatmaps(all_results: dict):
    """
    Heatmap of mean test scores: method (rows) x benchmark (cols).
    One heatmap per model.
    """
    models = sorted(set(k[0] for k in all_results))

    for model in models:
        benchmarks = sorted(set(k[1] for k in all_results if k[0] == model))
        benchmarks = [b for b in BENCHMARK_ORDER if b in benchmarks]
        methods_present = sorted(set(k[2] for k in all_results if k[0] == model),
                                  key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
        if not benchmarks or not methods_present:
            continue

        matrix = np.full((len(methods_present), len(benchmarks)), np.nan)
        counts = np.zeros((len(methods_present), len(benchmarks)), dtype=int)

        for mi, method in enumerate(methods_present):
            for bi, bm in enumerate(benchmarks):
                scores = []
                for seed in SEEDS:
                    key = (model, bm, method, str(seed))
                    if key in all_results and all_results[key].get("test_score") is not None:
                        scores.append(all_results[key]["test_score"])
                if scores:
                    matrix[mi, bi] = np.mean(scores)
                    counts[mi, bi] = len(scores)

        fig, ax = plt.subplots(figsize=(max(8, len(benchmarks) * 1.8), max(4, len(methods_present) * 0.7)))
        fig.suptitle(f"Test Score Heatmap — {model}", fontsize=14, fontweight="bold")

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0,
                        vmax=np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1)
        plt.colorbar(im, ax=ax, label="Mean Test Score")

        ax.set_xticks(range(len(benchmarks)))
        ax.set_xticklabels(benchmarks, fontsize=10)
        ax.set_yticks(range(len(methods_present)))
        ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods_present], fontsize=10)

        # Annotate cells
        for mi in range(len(methods_present)):
            for bi in range(len(benchmarks)):
                val = matrix[mi, bi]
                n = counts[mi, bi]
                if not np.isnan(val):
                    text_color = "white" if val < np.nanmax(matrix) * 0.4 else "black"
                    ax.text(bi, mi, f"{val:.3f}\n(n={n})", ha="center", va="center",
                            fontsize=8, color=text_color)
                else:
                    ax.text(bi, mi, "—", ha="center", va="center", fontsize=10, color="gray")

        plt.tight_layout()
        outpath = OUT / f"heatmap_{model}.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {outpath}")


def plot_cross_model_comparison(all_results: dict):
    """
    Compare the same methods across different model sizes, for benchmarks
    that have data for multiple models.
    """
    # Find benchmarks with data across multiple models
    bm_models = defaultdict(set)
    for (model, bm, method, seed) in all_results:
        if all_results[(model, bm, method, seed)].get("test_score") is not None:
            bm_models[bm].add(model)

    multi_model_bms = {bm: sorted(models) for bm, models in bm_models.items() if len(models) > 1}
    if not multi_model_bms:
        print("No cross-model data to compare.")
        return

    for bm, models in multi_model_bms.items():
        methods_present = sorted(set(
            k[2] for k in all_results
            if k[1] == bm and k[0] in models and all_results[k].get("test_score") is not None
        ), key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)

        if len(methods_present) < 2:
            continue

        fig, ax = plt.subplots(figsize=(max(10, len(methods_present) * len(models) * 0.8), 6))
        fig.suptitle(f"Cross-Model Comparison — {bm}", fontsize=14, fontweight="bold")

        x = np.arange(len(methods_present))
        n_models = len(models)
        bar_width = 0.8 / n_models
        model_colors = plt.cm.Set2(np.linspace(0, 1, n_models))

        for mi, model in enumerate(models):
            means, stds = [], []
            for method in methods_present:
                scores = []
                for seed in SEEDS:
                    key = (model, bm, method, str(seed))
                    if key in all_results and all_results[key].get("test_score") is not None:
                        scores.append(all_results[key]["test_score"])
                means.append(np.mean(scores) if scores else 0)
                stds.append(np.std(scores) if scores else 0)

            offset = (mi - n_models / 2 + 0.5) * bar_width
            ax.bar(x + offset, means, bar_width * 0.9, yerr=stds, label=model,
                   color=model_colors[mi], alpha=0.85, capsize=3, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods_present], fontsize=9, rotation=30, ha="right")
        ax.set_ylabel("Test Score")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        outpath = OUT / f"cross_model_{bm}.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {outpath}")


def print_summary(all_results: dict):
    """Print a text summary of results."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    models = sorted(set(k[0] for k in all_results))
    for model in models:
        print(f"\n{'─' * 60}")
        print(f"Model: {model}")
        print(f"{'─' * 60}")

        benchmarks = sorted(set(k[1] for k in all_results if k[0] == model))
        benchmarks = [b for b in BENCHMARK_ORDER if b in benchmarks]

        for bm in benchmarks:
            print(f"\n  Benchmark: {bm}")
            methods = sorted(set(k[2] for k in all_results if k[0] == model and k[1] == bm),
                              key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
            for method in methods:
                test_scores, val_scores = [], []
                for seed in SEEDS:
                    key = (model, bm, method, str(seed))
                    if key in all_results:
                        ts = all_results[key].get("test_score")
                        vs = all_results[key].get("val_score")
                        if ts is not None:
                            test_scores.append(ts)
                        if vs is not None:
                            val_scores.append(vs)
                if test_scores:
                    label = METHOD_LABELS.get(method, method)
                    print(f"    {label:25s}  test={np.mean(test_scores):.4f} +/- {np.std(test_scores):.4f}  "
                          f"val={np.mean(val_scores):.4f}  (n={len(test_scores)} seeds)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    all_results = load_all_results()
    print(f"Loaded {len(all_results)} result files.")

    print("\n--- Printing Summary ---")
    print_summary(all_results)

    print("\n--- Plotting Convergence Curves ---")
    plot_convergence_curves(all_results)

    print("\n--- Plotting Final Test Scores ---")
    plot_final_scores(all_results)

    print("\n--- Plotting Heatmaps ---")
    plot_heatmaps(all_results)

    print("\n--- Plotting Cross-Model Comparisons ---")
    plot_cross_model_comparison(all_results)

    print(f"\nAll plots saved to: {OUT}")


if __name__ == "__main__":
    main()
