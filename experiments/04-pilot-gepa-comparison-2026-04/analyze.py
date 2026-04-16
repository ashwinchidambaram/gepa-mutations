#!/usr/bin/env python3
"""Analyze the GEPA-vs-Slime-Mold pilot.

Reads the 6 result.json + metrics.json files from the pilot run, computes the
quantities pre-registered in README.md, writes outputs for inspection, and
prints a summary against the pre-registered decision matrix.

Usage:
    python experiments/04-pilot-gepa-comparison-2026-04/analyze.py \\
        --runs-dir experiments/04-pilot-gepa-comparison-2026-04/runs/qwen3-8b/hotpotqa
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# Optional plotting (skipped silently if matplotlib not installed)
try:
    import matplotlib.pyplot as plt  # noqa: F401
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


METHODS = ["gepa", "slime_mold"]
EXPECTED_SEEDS = [555, 999, 1337]


def _load_run(run_dir: Path) -> dict | None:
    """Load result.json + metrics.json for a single run. Returns None on failure."""
    res_path = run_dir / "result.json"
    met_path = run_dir / "metrics.json"
    if not res_path.exists():
        return None
    try:
        result = json.loads(res_path.read_text())
    except json.JSONDecodeError:
        return None
    metrics = {}
    if met_path.exists():
        try:
            metrics = json.loads(met_path.read_text())
        except json.JSONDecodeError:
            pass
    return {"result": result, "metrics": metrics, "run_dir": str(run_dir)}


def _trajectory_points(metrics: dict) -> list[tuple[int, float]]:
    """Extract (rollouts, val_score) trajectory from metrics.

    Handles both new (`val_score_trajectory`) and old (`convergence_curve`)
    schemas. Returns sorted list, possibly empty.
    """
    points: list[tuple[int, float]] = []

    # New schema
    for entry in metrics.get("val_score_trajectory", []) or []:
        if isinstance(entry, dict) and "rollouts" in entry and "val_score" in entry:
            points.append((int(entry["rollouts"]), float(entry["val_score"])))

    # Old schema
    if not points:
        for entry in metrics.get("convergence_curve", []) or []:
            if isinstance(entry, dict) and "rollouts" in entry and "score" in entry:
                points.append((int(entry["rollouts"]), float(entry["score"])))

    points.sort(key=lambda p: p[0])
    return points


def _rollouts_to_beat(trajectory: list[tuple[int, float]], threshold: float) -> int | None:
    """Return the first rollout count where val_score > threshold, or None."""
    for rollouts, score in trajectory:
        if score > threshold:
            return rollouts
    return None


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], 0.0)
    return (statistics.mean(values), statistics.stdev(values))


def _classify_outcome(slime_mean: float, gepa_mean: float, slime_std: float, gepa_std: float, gepa_seed_score: float) -> tuple[str, str]:
    """Apply the pre-registered decision matrix from README.md.

    Returns (outcome_tag, recommendation).
    """
    delta = slime_mean - gepa_mean

    # GEPA-fails-to-converge check first
    if gepa_mean <= gepa_seed_score:
        return (
            "gepa_failed_to_converge",
            "GEPA's mean test score did not exceed its seed prompt baseline. "
            "This is a finding in itself: GEPA's single-prompt behavior on "
            "Qwen3-8B is weak (the paper never tested this). Report Slime Mold "
            "standalone. Consider a follow-up with a stronger reflection LM.",
        )

    # Variance check
    if slime_std > 0.05 or gepa_std > 0.05:
        return (
            "too_noisy",
            f"Variance flag: slime_std={slime_std:.4f}, gepa_std={gepa_std:.4f}. "
            "Either method's std exceeds 0.05. Recommend adding 2 more seeds "
            "before deciding.",
        )

    if delta >= 0.05:
        return (
            "slime_mold_clearly_wins",
            "Slime Mold beats GEPA by ≥5% absolute with comparable variance. "
            "Proceed to full sweep across 4 benchmarks × 5 seeds × 2 methods.",
        )
    if delta >= 0.02:
        return (
            "comparable_with_cost_advantage",
            "Slime Mold is between 2–5% above GEPA. Proceed to full sweep but "
            "TEMPER claims: report as 'comparable performance with cost "
            "advantage' — NOT 'beats'.",
        )
    return (
        "inconclusive_or_gepa_wins",
        "Δ < 2% (or Slime Mold loses). Do NOT claim we beat GEPA. Reframe "
        "experiment 03 as a pure Slime-Mold ablation study (inductive vs "
        "personality vs prescribed8).",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=str,
        required=True,
        help="Path to runs directory (typically experiments/04-.../runs/qwen3-8b/hotpotqa)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write summary + plots (defaults to experiment dir parent of runs)",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"ERROR: runs-dir not found: {runs_dir}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else runs_dir.parent.parent.parent
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load runs
    by_method: dict[str, list[dict]] = {m: [] for m in METHODS}
    missing: list[str] = []
    for method in METHODS:
        method_dir = runs_dir / method
        if not method_dir.is_dir():
            missing.append(f"{method}/ (entire method dir missing)")
            continue
        for seed in EXPECTED_SEEDS:
            run = _load_run(method_dir / str(seed))
            if run is None:
                missing.append(f"{method}/{seed}")
            else:
                by_method[method].append(run)

    if missing:
        print(f"WARNING: {len(missing)} expected runs missing:")
        for m in missing:
            print(f"  - {m}")
        print()

    # Compute summary
    summary: dict = {
        "runs_dir": str(runs_dir),
        "expected_seeds": EXPECTED_SEEDS,
        "methods": {},
        "missing_runs": missing,
    }

    for method in METHODS:
        runs = by_method[method]
        test_scores = [r["result"]["test_score"] for r in runs if "test_score" in r["result"]]
        val_scores = [r["result"].get("val_score", float("nan")) for r in runs]
        seed_test_scores = [r["result"].get("seed_prompt_test_score") for r in runs]
        seed_test_scores = [s for s in seed_test_scores if s is not None]
        seed_val_scores = [r["result"].get("seed_prompt_val_score") for r in runs]
        seed_val_scores = [s for s in seed_val_scores if s is not None]
        rollouts_used = [r["result"].get("rollout_count", 0) for r in runs]
        wall_seconds = [r["result"].get("wall_clock_seconds", 0) for r in runs]

        test_mean, test_std = _mean_std(test_scores)
        val_mean, val_std = _mean_std(val_scores)
        seed_test_mean, _ = _mean_std(seed_test_scores)
        seed_val_mean, _ = _mean_std(seed_val_scores)
        best_test = max(test_scores) if test_scores else float("nan")

        # Rollouts-to-beat-seed per run, then median
        beat_rollouts = []
        for r in runs:
            traj = _trajectory_points(r["metrics"])
            seed_val = r["result"].get("seed_prompt_val_score", 0.0)
            n = _rollouts_to_beat(traj, seed_val)
            if n is not None:
                beat_rollouts.append(n)

        summary["methods"][method] = {
            "n_runs": len(runs),
            "test_score_mean": test_mean,
            "test_score_std": test_std,
            "test_score_best_of_n": best_test,
            "test_score_per_seed": {
                r["result"]["seed"]: r["result"]["test_score"] for r in runs
            },
            "val_score_mean": val_mean,
            "val_score_std": val_std,
            "seed_prompt_test_mean": seed_test_mean,
            "seed_prompt_val_mean": seed_val_mean,
            "rollouts_used_mean": _mean_std(rollouts_used)[0],
            "wall_clock_seconds_mean": _mean_std(wall_seconds)[0],
            "rollouts_to_beat_seed_median": (
                statistics.median(beat_rollouts) if beat_rollouts else None
            ),
            "rollouts_to_beat_seed_per_run": beat_rollouts,
        }

    # Apply pre-registered decision matrix
    gepa = summary["methods"]["gepa"]
    slime = summary["methods"]["slime_mold"]
    if gepa["n_runs"] > 0 and slime["n_runs"] > 0:
        outcome, recommendation = _classify_outcome(
            slime_mean=slime["test_score_mean"],
            gepa_mean=gepa["test_score_mean"],
            slime_std=slime["test_score_std"],
            gepa_std=gepa["test_score_std"],
            gepa_seed_score=gepa["seed_prompt_test_mean"],
        )
        delta = slime["test_score_mean"] - gepa["test_score_mean"]
        summary["delta_slime_minus_gepa"] = delta
        summary["pre_registered_outcome"] = outcome
        summary["recommendation"] = recommendation
    else:
        summary["pre_registered_outcome"] = "incomplete"
        summary["recommendation"] = "Cannot evaluate — runs missing."

    # Write JSON summary
    summary_path = output_dir / "analyze_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Write final prompts dump
    prompts_path = output_dir / "final_prompts.md"
    lines = ["# Final prompts from pilot\n"]
    lines.append("Each section shows the seed prompt, test score, and the optimized prompt.\n")
    for method in METHODS:
        lines.append(f"\n## Method: `{method}`\n")
        for r in sorted(by_method[method], key=lambda x: x["result"]["seed"]):
            seed = r["result"]["seed"]
            score = r["result"]["test_score"]
            seed_score = r["result"].get("seed_prompt_test_score", "n/a")
            best = r["result"].get("best_prompt", {})
            text = best.get("system_prompt", str(best)) if isinstance(best, dict) else str(best)
            lines.append(f"\n### Seed {seed}")
            lines.append(f"- **test_score:** {score:.4f}")
            lines.append(f"- **seed_prompt_test_score:** {seed_score}")
            lines.append(f"- **rollouts_used:** {r['result'].get('rollout_count', 'n/a')}")
            lines.append(f"- **wall_clock_seconds:** {r['result'].get('wall_clock_seconds', 'n/a')}")
            lines.append("\n```\n" + text + "\n```\n")
    prompts_path.write_text("\n".join(lines))

    # Convergence plot
    if plt is not None:
        _plt = plt  # local alias for type narrowing
        fig, ax = _plt.subplots(figsize=(10, 6))
        colors = {"gepa": "tab:blue", "slime_mold": "tab:orange"}
        for method in METHODS:
            for r in by_method[method]:
                traj = _trajectory_points(r["metrics"])
                if traj:
                    xs = [p[0] for p in traj]
                    ys = [p[1] for p in traj]
                    label = f"{method} seed={r['result']['seed']}"
                    ax.plot(xs, ys, marker="o", color=colors.get(method, "gray"),
                            alpha=0.6, label=label)
        ax.set_xlabel("Rollouts used")
        ax.set_ylabel("Val score")
        ax.set_title("HotpotQA pilot: convergence per run (Qwen3-8B, paper budget 6871)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        out = plots_dir / "convergence.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        _plt.close(fig)
        print(f"Wrote convergence plot: {out}")

    # Pre-registered stdout summary
    print()
    print("=" * 72)
    print("HotpotQA pilot results (paper budget 6871, Qwen3-8B)")
    print("=" * 72)
    for method in METHODS:
        m = summary["methods"][method]
        if m["n_runs"] == 0:
            print(f"{method:12s}  NO RUNS COMPLETED")
            continue
        print(
            f"{method:12s}  test={m['test_score_mean']:.3f} ± {m['test_score_std']:.3f}  "
            f"(best: {m['test_score_best_of_n']:.3f}, n={m['n_runs']}, "
            f"seed_baseline={m['seed_prompt_test_mean']:.3f})"
        )
    if "delta_slime_minus_gepa" in summary:
        delta = summary["delta_slime_minus_gepa"]
        sign = "+" if delta >= 0 else ""
        print(f"\nGap (Slime Mold − GEPA): {sign}{delta:.3f} absolute")
        print(f"\nPre-registered outcome: {summary['pre_registered_outcome']}")
        print(f"Recommendation:        {summary['recommendation']}")
    print()
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {prompts_path}")
    print()
    print("DECISION POINT — see experiments/04-pilot-gepa-comparison-2026-04/README.md")
    print("for the pre-registered decision matrix. Do NOT proceed to full sweep")
    print("without explicit user go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
