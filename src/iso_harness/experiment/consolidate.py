"""JSONL -> Parquet consolidation pipeline.

Converts append-only JSONL files from experiment runs into columnar Parquet
for efficient analysis. Idempotent -- safe to re-run. Builds a global
experiment_index.parquet for cross-run queries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def consolidate_jsonl(jsonl_path: Path, parquet_path: Path | None = None) -> Path | None:
    """Convert a single JSONL file to Parquet.

    Args:
        jsonl_path: Path to the input JSONL file.
        parquet_path: Output path. Defaults to same name with .parquet extension.

    Returns:
        Path to the written Parquet file, or None if JSONL is empty/missing.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        return None

    if parquet_path is None:
        parquet_path = jsonl_path.with_suffix(".parquet")

    records: list[dict] = []
    errors = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                errors += 1
                logger.warning("Skipping corrupt line %d in %s", line_num, jsonl_path)

    if not records:
        return None

    if errors > 0:
        logger.warning("%d corrupt lines skipped in %s", errors, jsonl_path)

    # Convert to PyArrow table (infer schema from data)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, parquet_path)

    logger.info(
        "Consolidated %s -> %s (%d records, %d errors)",
        jsonl_path.name,
        parquet_path.name,
        len(records),
        errors,
    )
    return parquet_path


def consolidate_run(run_dir: Path) -> dict[str, Path | None]:
    """Consolidate all JSONL files in a run directory to Parquet.

    Processes: rollouts.jsonl, reflections.jsonl, candidates.jsonl,
    rounds.jsonl, telemetry.jsonl. Skips files that don't exist.
    Also converts summary.json to summary.parquet.

    Args:
        run_dir: Path to runs/{run_id}/.

    Returns:
        Dict mapping filename stem to output Parquet path (or None if skipped).
    """
    run_dir = Path(run_dir)
    results: dict[str, Path | None] = {}

    # JSONL files to consolidate
    jsonl_names = [
        "rollouts.jsonl",
        "reflections.jsonl",
        "candidates.jsonl",
        "rounds.jsonl",
        "telemetry.jsonl",
    ]

    for name in jsonl_names:
        jsonl_path = run_dir / name
        parquet_path = run_dir / name.replace(".jsonl", ".parquet")

        # Skip if Parquet already exists and is newer than JSONL
        if parquet_path.exists() and jsonl_path.exists():
            if parquet_path.stat().st_mtime >= jsonl_path.stat().st_mtime:
                logger.debug("Skipping %s (Parquet is up to date)", name)
                results[name.replace(".jsonl", "")] = parquet_path
                continue

        results[name.replace(".jsonl", "")] = consolidate_jsonl(jsonl_path, parquet_path)

    # Handle summary.json -> summary.parquet
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            table = pa.Table.from_pylist([summary])
            summary_parquet = run_dir / "summary.parquet"
            pq.write_table(table, summary_parquet)
            results["summary"] = summary_parquet
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Failed to consolidate summary.json: %s", e)
            results["summary"] = None
    else:
        results["summary"] = None

    return results


def consolidate_all(experiment_dir: Path) -> int:
    """Consolidate all completed runs in an experiment directory.

    Walks runs/*/ directories, processes only those with a COMPLETE marker.
    Builds/updates a global experiment_index.parquet.

    Args:
        experiment_dir: Root experiment directory containing runs/.

    Returns:
        Number of runs consolidated.
    """
    experiment_dir = Path(experiment_dir)
    runs_dir = experiment_dir / "runs"
    if not runs_dir.exists():
        logger.warning("No runs directory found at %s", runs_dir)
        return 0

    consolidated_count = 0
    index_rows: list[dict] = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "COMPLETE").exists():
            logger.debug("Skipping incomplete run: %s", run_dir.name)
            continue

        consolidate_run(run_dir)
        consolidated_count += 1

        # Build index row from summary.json if available
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                summary["_run_dir"] = str(run_dir)
                index_rows.append(summary)
            except json.JSONDecodeError:
                pass

    # Write experiment index
    if index_rows:
        index_path = experiment_dir / "experiment_index.parquet"
        table = pa.Table.from_pylist(index_rows)
        pq.write_table(table, index_path)
        logger.info(
            "Updated experiment index: %d runs -> %s", len(index_rows), index_path
        )

    logger.info("Consolidated %d runs in %s", consolidated_count, experiment_dir)
    return consolidated_count
