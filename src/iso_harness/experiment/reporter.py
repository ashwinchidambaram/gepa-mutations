"""Per-run report generation: summary.json, report.md, experiment_results.csv.

Generated at the end of each experiment run by the orchestrator.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "run_id", "optimizer", "benchmark", "seed",
    "final_score_val", "final_score_test",
    "rollouts_consumed", "tokens_consumed",
    "wall_clock_seconds", "cost_estimate_usd",
    "git_sha", "timestamp",
]

_csv_lock = threading.Lock()


def write_summary_json(run_dir: Path, summary: dict[str, Any]) -> Path:
    """Write summary.json to the run directory (atomic write).

    Args:
        run_dir: Path to runs/{run_id}/.
        summary: Dict with RunSummary-compatible fields.

    Returns:
        Path to the written summary.json file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    path = run_dir / "summary.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))

    logger.info("Wrote summary.json to %s", path)
    return path


def write_report_md(run_dir: Path, summary: dict[str, Any]) -> Path:
    """Write a human-readable report.md to the run directory.

    Args:
        run_dir: Path to runs/{run_id}/.
        summary: Dict with RunSummary-compatible fields.

    Returns:
        Path to the written report.md file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    optimizer = summary.get("optimizer", "unknown")
    benchmark = summary.get("benchmark", "unknown")
    seed = summary.get("seed", "?")
    val_score = summary.get("final_score_val", 0.0)
    test_score = summary.get("final_score_test", 0.0)
    rollouts = summary.get("rollouts_consumed_total", 0)
    tokens = summary.get("tokens_consumed_total", 0)
    duration = summary.get("duration_seconds", 0.0)
    cost = summary.get("cost_estimate_usd", 0.0)
    git_sha = summary.get("git_sha", "unknown")
    model_task = summary.get("model_task", "unknown")
    model_refl = summary.get("model_reflection", "unknown")
    prompts = summary.get("final_candidate_prompts", {})

    lines = [
        f"# Run Report: {optimizer}/{benchmark}/seed={seed}",
        "",
        f"**Run ID:** `{summary.get('run_id', 'unknown')}`",
        f"**Git SHA:** `{git_sha}`",
        f"**Timestamp:** {summary.get('start_time', 'unknown')}",
        "",
        "## Scores",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Validation score | {val_score:.4f} |",
        f"| Test score | {test_score:.4f} |",
        "",
        "## Resource Usage",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Rollouts consumed | {rollouts:,} |",
        f"| Tokens consumed | {tokens:,} |",
        f"| Wall-clock time | {duration:.1f}s ({duration/3600:.2f}h) |",
        f"| Estimated cost | ${cost:.2f} |",
        "",
        "## Models",
        "",
        f"- **Task model:** {model_task}",
        f"- **Reflection model:** {model_refl}",
        "",
        "## Final Prompts",
        "",
    ]

    if prompts:
        for module_name, prompt_text in prompts.items():
            lines.append(f"### {module_name}")
            lines.append("")
            lines.append("```")
            lines.append(str(prompt_text))
            lines.append("```")
            lines.append("")
    else:
        lines.append("_No prompts recorded._")
        lines.append("")

    path = run_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Wrote report.md to %s", path)
    return path


def append_to_results_csv(
    csv_path: Path,
    summary: dict[str, Any],
) -> None:
    """Append a row to the experiment results CSV (thread-safe).

    Creates the CSV with headers if it doesn't exist.

    Args:
        csv_path: Path to experiment_results.csv at the experiment root.
        summary: Dict with RunSummary-compatible fields.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "run_id": summary.get("run_id", ""),
        "optimizer": summary.get("optimizer", ""),
        "benchmark": summary.get("benchmark", ""),
        "seed": summary.get("seed", ""),
        "final_score_val": summary.get("final_score_val", ""),
        "final_score_test": summary.get("final_score_test", ""),
        "rollouts_consumed": summary.get("rollouts_consumed_total", ""),
        "tokens_consumed": summary.get("tokens_consumed_total", ""),
        "wall_clock_seconds": summary.get("duration_seconds", ""),
        "cost_estimate_usd": summary.get("cost_estimate_usd", ""),
        "git_sha": summary.get("git_sha", ""),
        "timestamp": summary.get("start_time", datetime.now(timezone.utc).isoformat()),
    }

    with _csv_lock:
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    logger.info("Appended result row to %s", csv_path)


def generate_run_report(
    run_dir: Path,
    summary: dict[str, Any],
    csv_path: Path | None = None,
) -> None:
    """Generate all per-run reports: summary.json, report.md, and CSV row.

    Args:
        run_dir: Path to runs/{run_id}/.
        summary: Dict with RunSummary-compatible fields.
        csv_path: Path to experiment_results.csv. If None, CSV is skipped.
    """
    write_summary_json(run_dir, summary)
    write_report_md(run_dir, summary)
    if csv_path is not None:
        append_to_results_csv(csv_path, summary)
