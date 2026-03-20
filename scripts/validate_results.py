#!/usr/bin/env python3
"""Validate all downloaded result.json files for completeness and sanity.

Checks every result.json under runs/ to ensure required fields are present,
scores are in valid ranges, and per-example data is non-empty.

Usage:
    python scripts/validate_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS_DIR = Path("runs")

REQUIRED_FIELDS = [
    "benchmark",
    "method",
    "seed",
    "test_score",
    "val_score",
    "best_prompt",
    "rollout_count",
    "wall_clock_seconds",
    "test_example_scores",
    "test_example_ids",
]


def validate() -> None:
    result_files = sorted(RUNS_DIR.rglob("result.json"))

    if not result_files:
        print("No result.json files found under runs/.")
        return

    errors: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []

    for rf in result_files:
        rel = str(rf.relative_to(RUNS_DIR))
        try:
            with open(rf) as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            errors.append((rel, f"Invalid JSON: {exc}"))
            continue

        # Required fields
        missing = [fld for fld in REQUIRED_FIELDS if fld not in data]
        if missing:
            errors.append((rel, f"Missing fields: {missing}"))

        # Score range
        for key in ("test_score", "val_score"):
            val = data.get(key)
            if val is not None and not (0 <= val <= 1):
                errors.append((rel, f"{key} out of range [0,1]: {val}"))

        # Per-example data
        example_scores = data.get("test_example_scores", [])
        example_ids = data.get("test_example_ids", [])
        if len(example_scores) == 0:
            warnings.append((rel, "Empty test_example_scores"))
        if len(example_ids) == 0:
            warnings.append((rel, "Empty test_example_ids"))
        if example_scores and example_ids and len(example_scores) != len(example_ids):
            errors.append(
                (rel, f"Length mismatch: {len(example_scores)} scores "
                      f"vs {len(example_ids)} ids")
            )

        # Wall clock sanity
        wc = data.get("wall_clock_seconds", 0)
        if wc < 60:
            warnings.append((rel, f"Very short wall clock: {wc:.0f}s"))
        if wc > 86400:
            warnings.append((rel, f"Very long wall clock: {wc/3600:.1f}h"))

        # Rollout count
        rc = data.get("rollout_count", 0)
        if rc == 0:
            warnings.append((rel, "rollout_count is 0"))

    # Report
    print(f"\nValidated {len(result_files)} result files.\n")

    if errors:
        print(f"ERRORS ({len(errors)}):")
        for path, msg in errors:
            print(f"  {path}: {msg}")
        print()

    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for path, msg in warnings:
            print(f"  {path}: {msg}")
        print()

    if not errors and not warnings:
        print("All files passed validation with no issues.")
    elif not errors:
        print("No errors found (warnings only).")
    else:
        print(f"FAILED: {len(errors)} errors found.")


if __name__ == "__main__":
    validate()
