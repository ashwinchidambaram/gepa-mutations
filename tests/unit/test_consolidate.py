"""Tests for JSONL -> Parquet consolidation pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from iso_harness.experiment.consolidate import (
    consolidate_all,
    consolidate_jsonl,
    consolidate_run,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Helper: write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


class TestConsolidateJsonl:
    def test_basic_conversion(self, tmp_path: Path):
        jsonl = tmp_path / "data.jsonl"
        _write_jsonl(jsonl, [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}])

        result = consolidate_jsonl(jsonl)
        assert result is not None
        assert result.suffix == ".parquet"

        table = pq.read_table(result)
        assert len(table) == 2
        assert "a" in table.column_names
        assert "b" in table.column_names

    def test_missing_file_returns_none(self, tmp_path: Path):
        assert consolidate_jsonl(tmp_path / "nope.jsonl") is None

    def test_empty_file_returns_none(self, tmp_path: Path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        assert consolidate_jsonl(empty) is None

    def test_corrupt_lines_skipped(self, tmp_path: Path):
        jsonl = tmp_path / "mixed.jsonl"
        with open(jsonl, "w") as f:
            f.write('{"valid": true}\n')
            f.write("NOT VALID JSON\n")
            f.write('{"also_valid": true}\n')

        result = consolidate_jsonl(jsonl)
        assert result is not None
        table = pq.read_table(result)
        assert len(table) == 2

    def test_custom_output_path(self, tmp_path: Path):
        jsonl = tmp_path / "input.jsonl"
        out = tmp_path / "custom_output.parquet"
        _write_jsonl(jsonl, [{"x": 1}])

        result = consolidate_jsonl(jsonl, parquet_path=out)
        assert result == out
        assert out.exists()


class TestConsolidateRun:
    def test_consolidates_available_files(self, tmp_path: Path):
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        _write_jsonl(run_dir / "rollouts.jsonl", [{"score": 0.5}, {"score": 0.7}])
        _write_jsonl(run_dir / "reflections.jsonl", [{"tokens": 100}])
        # candidates.jsonl doesn't exist -- should be skipped

        summary = {
            "optimizer": "iso",
            "benchmark": "hotpotqa",
            "final_score_test": 0.8,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))

        results = consolidate_run(run_dir)
        assert results["rollouts"] is not None
        assert results["reflections"] is not None
        assert results["candidates"] is None  # didn't exist
        assert results["summary"] is not None

    def test_skips_up_to_date_parquet(self, tmp_path: Path):
        run_dir = tmp_path / "run_002"
        run_dir.mkdir()

        jsonl = run_dir / "rollouts.jsonl"
        parquet = run_dir / "rollouts.parquet"
        _write_jsonl(jsonl, [{"score": 0.5}])
        consolidate_jsonl(jsonl)  # Create initial parquet

        # Parquet should be up to date now
        time.sleep(0.01)  # Ensure mtime difference
        results = consolidate_run(run_dir)
        assert results["rollouts"] == parquet

    def test_idempotent(self, tmp_path: Path):
        run_dir = tmp_path / "run_003"
        run_dir.mkdir()
        _write_jsonl(run_dir / "rollouts.jsonl", [{"x": 1}, {"x": 2}])

        r1 = consolidate_run(run_dir)
        r2 = consolidate_run(run_dir)
        assert r1["rollouts"] == r2["rollouts"]

        table = pq.read_table(r2["rollouts"])
        assert len(table) == 2  # No duplicates


class TestConsolidateAll:
    def test_processes_completed_runs(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"

        # Complete run
        r1 = runs_dir / "run_a"
        r1.mkdir(parents=True)
        _write_jsonl(r1 / "rollouts.jsonl", [{"score": 0.9}])
        (r1 / "summary.json").write_text(
            json.dumps({"run_id": "a", "benchmark": "hotpotqa"})
        )
        (r1 / "COMPLETE").write_text("{}")

        # Incomplete run (no COMPLETE marker)
        r2 = runs_dir / "run_b"
        r2.mkdir()
        _write_jsonl(r2 / "rollouts.jsonl", [{"score": 0.3}])

        count = consolidate_all(tmp_path)
        assert count == 1  # Only completed run

        # Check experiment index
        index = tmp_path / "experiment_index.parquet"
        assert index.exists()
        table = pq.read_table(index)
        assert len(table) == 1
        assert table.column("benchmark")[0].as_py() == "hotpotqa"

    def test_no_runs_dir(self, tmp_path: Path):
        assert consolidate_all(tmp_path) == 0

    def test_empty_runs_dir(self, tmp_path: Path):
        (tmp_path / "runs").mkdir()
        assert consolidate_all(tmp_path) == 0
