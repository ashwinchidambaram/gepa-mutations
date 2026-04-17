#!/usr/bin/env python3
"""Validate Parquet log schema integrity for experiment runs.

Checks:
- Schema conformance (all required columns present, correct types)
- No null values in required fields
- run_id consistency across tables
- param_count_b populated in reflections (for cost model)

Usage:
    python scripts/validate_logs.py runs/smoke/hotpotqa/*/42
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

_REQUIRED_ROLLOUT_COLS = {
    "run_id", "method", "seed", "phase", "round", "candidate_id",
    "example_id", "score", "prompt_tokens", "completion_tokens",
    "wall_clock_ms", "timestamp",
}

_REQUIRED_REFLECTION_COLS = {
    "run_id", "method", "seed", "phase", "round", "call_type",
    "input_tokens", "output_tokens", "wall_clock_ms",
    "input_hash", "output_hash", "model_id", "param_count_b", "timestamp",
}

_REQUIRED_CANDIDATE_COLS = {
    "run_id", "candidate_id", "generation", "parent_ids",
    "strategy_name", "strategy_technique", "prompt_text", "created_at",
}


def validate_run(run_dir: Path) -> list[str]:
    """Validate a single run directory. Returns list of error strings."""
    errors = []

    rollouts_path = run_dir / "rollouts.parquet"
    reflections_path = run_dir / "reflections.parquet"
    candidates_path = run_dir / "candidates.parquet"

    # Check rollouts
    if rollouts_path.exists():
        table = pq.read_table(rollouts_path)
        missing = _REQUIRED_ROLLOUT_COLS - set(table.column_names)
        if missing:
            errors.append(f"rollouts.parquet missing columns: {missing}")
        # Check for nulls in critical fields
        for col in ["run_id", "method", "score"]:
            if col in table.column_names:
                nulls = table.column(col).null_count
                if nulls > 0:
                    errors.append(f"rollouts.parquet has {nulls} nulls in '{col}'")
    else:
        errors.append("rollouts.parquet not found")

    # Check reflections
    if reflections_path.exists():
        table = pq.read_table(reflections_path)
        missing = _REQUIRED_REFLECTION_COLS - set(table.column_names)
        if missing:
            errors.append(f"reflections.parquet missing columns: {missing}")
        # Check param_count_b populated
        if "param_count_b" in table.column_names:
            nulls = table.column("param_count_b").null_count
            if nulls > 0:
                errors.append(f"reflections.parquet has {nulls} nulls in 'param_count_b' (needed for cost model)")
        if "model_id" in table.column_names:
            nulls = table.column("model_id").null_count
            if nulls > 0:
                errors.append(f"reflections.parquet has {nulls} nulls in 'model_id'")

    # Check run_id consistency
    run_ids = set()
    for path, required in [
        (rollouts_path, _REQUIRED_ROLLOUT_COLS),
        (reflections_path, _REQUIRED_REFLECTION_COLS),
    ]:
        if path.exists():
            table = pq.read_table(path)
            if "run_id" in table.column_names:
                run_ids.update(table.column("run_id").to_pylist())

    if len(run_ids) > 1:
        errors.append(f"Multiple run_ids across tables: {run_ids}")

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_logs.py <run_dir> [run_dir ...]")
        sys.exit(1)

    all_ok = True
    for path_str in sys.argv[1:]:
        run_dir = Path(path_str)
        if not run_dir.is_dir():
            print(f"SKIP {run_dir} (not a directory)")
            continue

        errors = validate_run(run_dir)
        if errors:
            print(f"FAIL {run_dir}")
            for err in errors:
                print(f"  - {err}")
            all_ok = False
        else:
            print(f"OK   {run_dir}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
