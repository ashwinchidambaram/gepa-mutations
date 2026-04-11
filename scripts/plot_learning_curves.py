#!/usr/bin/env python3
"""
Plot learning curves (best_val_score vs rollouts) for all completed/active runs.
Reads run_events.jsonl from every run directory.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

ROOT = Path(__file__).parent.parent / "runs"
OUT = Path(__file__).parent.parent / "plots"
OUT.mkdir(exist_ok=True)

METHODS = ["gepa", "best_of_k_K3", "contrastive_reflection", "failure_stratified_k_K3"]
BENCHMARKS = ["livebench", "hover", "pupa", "ifbench", "aime", "hotpotqa"]
SEEDS = [42, 123, 456, 789, 1024]
METHOD_COLORS = {
    "gepa": "#1f77b4",
    "best_of_k_K3": "#ff7f0e",
    "contrastive_reflection": "#2ca02c",
    "failure_stratified_k_K3": "#d62728",
}


def load_events(path: Path):
    events = []
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))
    except Exception:
        pass
    return events


def plot_benchmark(benchmark: str, axs_row, methods=METHODS):
    """Plot one benchmark across all methods, one subplot per method."""
    for ax, method in zip(axs_row, methods):
        ax.set_title(f"{method.replace('_', ' ')}", fontsize=8)
        ax.set_xlabel("Rollouts", fontsize=7)
        ax.set_ylabel("Best val score", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

        colors = cm.tab10(np.linspace(0, 1, len(SEEDS)))
        any_data = False
        for seed, color in zip(SEEDS, colors):
            evfile = ROOT / benchmark / method / str(seed) / "gepa_state" / "run_events.jsonl"
            events = load_events(evfile)
            if not events:
                continue
            rollouts = [e["rollouts"] for e in events]
            scores = [e["best_val_score"] for e in events]
            ax.plot(rollouts, scores, color=color, linewidth=1.2, label=f"seed={seed}")
            # Mark accepted (improved) iterations
            accepted_r = [e["rollouts"] for e in events if e.get("accepted")]
            accepted_s = [e["best_val_score"] for e in events if e.get("accepted")]
            if accepted_r:
                ax.scatter(accepted_r, accepted_s, color=color, s=15, zorder=5)
            any_data = True

        if any_data:
            ax.legend(fontsize=5, loc="lower right")
        else:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")


# --- Figure 1: per-benchmark grid (benchmark x method) ---
fig, axes = plt.subplots(
    len(BENCHMARKS), len(METHODS),
    figsize=(5 * len(METHODS), 3.5 * len(BENCHMARKS)),
    squeeze=False,
)
fig.suptitle("GEPA-Mutations — Learning Curves (best_val_score vs rollouts)", fontsize=13, y=1.01)

for i, bm in enumerate(BENCHMARKS):
    axes[i][0].set_ylabel(f"{bm}\nbest val score", fontsize=8)
    plot_benchmark(bm, axes[i])

plt.tight_layout()
out1 = OUT / "learning_curves_grid.png"
fig.savefig(out1, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out1}")


# --- Figure 2: per-benchmark summary (all methods overlaid, averaged across seeds) ---
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
fig2.suptitle("GEPA-Mutations — Average Learning Curves per Benchmark", fontsize=13)

for idx, bm in enumerate(BENCHMARKS):
    ax = axes2[idx // 3][idx % 3]
    ax.set_title(bm, fontsize=11)
    ax.set_xlabel("Rollouts")
    ax.set_ylabel("Best val score")
    ax.grid(True, alpha=0.3)

    for method in METHODS:
        all_curves = []
        for seed in SEEDS:
            evfile = ROOT / bm / method / str(seed) / "gepa_state" / "run_events.jsonl"
            events = load_events(evfile)
            if events:
                all_curves.append([(e["rollouts"], e["best_val_score"]) for e in events])

        if not all_curves:
            continue

        # Interpolate all curves to a common rollout grid and average
        max_r = max(c[-1][0] for c in all_curves)
        grid = np.linspace(0, max_r, 300)
        interp_curves = []
        for curve in all_curves:
            rs, ss = zip(*curve)
            interp_curves.append(np.interp(grid, rs, ss))

        mean = np.mean(interp_curves, axis=0)
        std = np.std(interp_curves, axis=0) if len(interp_curves) > 1 else np.zeros_like(mean)

        color = METHOD_COLORS.get(method, "gray")
        label = method.replace("_", " ")
        ax.plot(grid, mean, color=color, linewidth=2, label=label)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.15, color=color)

    ax.legend(fontsize=7)

plt.tight_layout()
out2 = OUT / "learning_curves_summary.png"
fig2.savefig(out2, dpi=120, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")

print("\nDone. Open plots/ to view.")
