#!/usr/bin/env python3
"""
Generate one large, readable convergence curve plot per benchmark.

Focuses on qwen3-1.7b (most complete data). Each plot:
  - figsize=(14, 8)
  - x-axis: cumulative rollouts (LLM calls)
  - y-axis: best val score so far
  - One line per method: mean across seeds with +/- 1 std shading
  - Large fonts, clear legend
  - Saved as convergence_detail_{benchmark}_{model}.png
"""
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
RUNS_DIRS = [
    PROJECT / "runs",
    PROJECT / "runs_archive_slurm_sweep" / "archive",
]
OUT = PROJECT / "analysis"
OUT.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]

METHOD_ORDER = [
    "baseline",
    "gepa",
    "contrastive_reflection",
    "synaptic_pruning",
    "iso",
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
    "iso": "#9467bd",
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
    "iso": "Slime Mold",
    "tournament": "Tournament",
    "best_of_k_K3": "Best-of-K (K=3)",
    "failure_stratified_k_K3": "Fail-Strat K (K=3)",
    "contrastive_synthesis": "Contrastive Synth.",
    "ecological_succession": "Ecological Succ.",
}

BENCHMARK_ORDER = ["aime", "hover", "hotpotqa", "ifbench", "livebench", "pupa"]

# Focus model
FOCUS_MODEL = "qwen3-1.7b"


# ── Data Loading (reused from visualize_results.py) ─────────────────────────

def infer_structure(root_dir: Path, rj_path: Path):
    parts = rj_path.relative_to(root_dir).parts
    if str(root_dir).endswith("archive") and parts[0] in BENCHMARK_ORDER:
        return "unknown", parts[0], parts[1], parts[2]
    return parts[0], parts[1], parts[2], parts[3]


def load_all_results():
    results = {}
    for root_dir in RUNS_DIRS:
        if not root_dir.exists():
            continue
        for rj in root_dir.rglob("result.json"):
            try:
                d = json.load(open(rj))
                model, bm, method, seed = infer_structure(root_dir, rj)
                results[(model, bm, method, str(seed))] = d
                d["_path"] = rj.parent
                d["_model"] = model
            except Exception:
                pass
    return results


def load_convergence_curve(run_dir: Path, method: str):
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

    # Try metrics.json trajectories
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

    # Try iterations format
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


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_detail(all_results: dict, model: str):
    """Generate one large plot per benchmark for the given model."""
    benchmarks = sorted(set(k[1] for k in all_results if k[0] == model))
    benchmarks = [b for b in BENCHMARK_ORDER if b in benchmarks]

    generated = []

    for bm in benchmarks:
        methods_present = sorted(
            set(k[2] for k in all_results if k[0] == model and k[1] == bm and k[2] != "baseline"),
            key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99,
        )
        if not methods_present:
            print(f"  [{bm}] No non-baseline methods found, skipping.")
            continue

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_title(f"Convergence — {bm} ({model})", fontsize=18, fontweight="bold", pad=12)
        ax.set_xlabel("Cumulative Rollouts (LLM calls)", fontsize=14)
        ax.set_ylabel("Best Val Score", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

        # Baseline horizontal line
        baseline_scores = []
        for seed in SEEDS:
            key = (model, bm, "baseline", str(seed))
            if key in all_results:
                vs = all_results[key].get("val_score")
                if vs is not None:
                    baseline_scores.append(vs)
        if baseline_scores:
            mean_bl = np.mean(baseline_scores)
            std_bl = np.std(baseline_scores)
            ax.axhline(mean_bl, color=METHOD_COLORS["baseline"], linestyle="--",
                       linewidth=2, alpha=0.7, label=f"Baseline ({mean_bl:.3f})")
            if std_bl > 0:
                ax.axhspan(mean_bl - std_bl, mean_bl + std_bl,
                           color=METHOD_COLORS["baseline"], alpha=0.07)

        methods_plotted = 0
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
                ax.plot(all_rollouts[0], all_scores[0], color=color, linewidth=2,
                        label=f"{label} (n=1)", alpha=0.85)
            else:
                max_r = max(c[-1] for c in all_rollouts)
                grid = np.linspace(0, max_r, 500)
                interp = []
                for r, s in zip(all_rollouts, all_scores):
                    interp.append(np.interp(grid, r, s))
                mean_curve = np.mean(interp, axis=0)
                std_curve = np.std(interp, axis=0)
                ax.plot(grid, mean_curve, color=color, linewidth=2.5,
                        label=f"{label} (n={len(all_rollouts)})")
                ax.fill_between(grid, mean_curve - std_curve, mean_curve + std_curve,
                                alpha=0.15, color=color)
            methods_plotted += 1

        if methods_plotted == 0:
            plt.close(fig)
            print(f"  [{bm}] No convergence data found, skipping.")
            continue

        ax.legend(fontsize=12, loc="best", framealpha=0.9)
        plt.tight_layout()

        outpath = OUT / f"convergence_detail_{bm}_{model}.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outpath}")
        generated.append(outpath)

    return generated


def main():
    print("Loading results...")
    all_results = load_all_results()
    print(f"Loaded {len(all_results)} result files.")

    # Report available data for focus model
    focus_keys = [k for k in all_results if k[0] == FOCUS_MODEL]
    benchmarks = sorted(set(k[1] for k in focus_keys))
    methods = sorted(set(k[2] for k in focus_keys))
    print(f"\n{FOCUS_MODEL}: {len(focus_keys)} runs across {len(benchmarks)} benchmarks, {len(methods)} methods")
    print(f"  Benchmarks: {benchmarks}")
    print(f"  Methods: {methods}")

    print(f"\nGenerating detail plots for {FOCUS_MODEL}...")
    generated = plot_detail(all_results, FOCUS_MODEL)

    # Also generate for other models if they have data
    other_models = sorted(set(k[0] for k in all_results if k[0] != FOCUS_MODEL))
    for model in other_models:
        n = len([k for k in all_results if k[0] == model])
        print(f"\nAlso generating for {model} ({n} runs)...")
        generated.extend(plot_detail(all_results, model))

    print(f"\nDone. Generated {len(generated)} plots in {OUT}")


if __name__ == "__main__":
    main()
