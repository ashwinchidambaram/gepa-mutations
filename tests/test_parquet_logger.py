"""Tests for the unified Parquet logging schema."""

import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from gepa_mutations.logging.parquet_logger import ParquetLogger


def test_rollout_schema():
    with tempfile.TemporaryDirectory() as d:
        logger = ParquetLogger(Path(d))
        logger.log_rollout(
            run_id="test-001", method="iso_base", seed=42, phase="pruning",
            round_num=1, candidate_id=3, example_id=7, score=0.85,
            prompt_tokens=150, completion_tokens=80, wall_clock_ms=234.5,
        )
        logger.flush()
        table = pq.read_table(Path(d) / "rollouts.parquet")
        assert table.num_rows == 1
        assert set(table.column_names) >= {
            "run_id", "method", "seed", "phase", "round", "candidate_id",
            "example_id", "score", "prompt_tokens", "completion_tokens",
            "wall_clock_ms", "timestamp",
        }
        row = table.to_pydict()
        assert row["run_id"][0] == "test-001"
        assert row["score"][0] == pytest.approx(0.85)


def test_reflection_schema_has_model_fields():
    with tempfile.TemporaryDirectory() as d:
        logger = ParquetLogger(Path(d))
        logger.log_reflection(
            run_id="test-001", method="iso_base", seed=42, phase="discovery",
            round_num=0, call_type="discovery", input_tokens=500,
            output_tokens=300, wall_clock_ms=1200.0,
            input_hash="abc123", output_hash="def456",
            model_id="qwen3-32b", param_count_b=32.0,
        )
        logger.flush()
        table = pq.read_table(Path(d) / "reflections.parquet")
        assert table.num_rows == 1
        assert "model_id" in table.column_names
        assert "param_count_b" in table.column_names
        row = table.to_pydict()
        assert row["model_id"][0] == "qwen3-32b"
        assert row["param_count_b"][0] == pytest.approx(32.0)


def test_candidate_schema():
    with tempfile.TemporaryDirectory() as d:
        logger = ParquetLogger(Path(d))
        logger.log_candidate(
            run_id="test-001", candidate_id=5, generation=2,
            parent_ids=[1, 3], strategy_name="decomposition",
            strategy_technique="Decomposition", prompt_text="You are...",
        )
        logger.flush()
        table = pq.read_table(Path(d) / "candidates.parquet")
        assert table.num_rows == 1
        row = table.to_pydict()
        assert row["candidate_id"][0] == 5
        assert row["prompt_text"][0] == "You are..."


def test_multiple_rows_buffered():
    with tempfile.TemporaryDirectory() as d:
        logger = ParquetLogger(Path(d))
        for i in range(10):
            logger.log_rollout(
                run_id="test-001", method="gepa", seed=42, phase="eval",
                round_num=i, candidate_id=0, example_id=i, score=0.5 + i * 0.05,
                prompt_tokens=100, completion_tokens=50, wall_clock_ms=100.0,
            )
        logger.flush()
        table = pq.read_table(Path(d) / "rollouts.parquet")
        assert table.num_rows == 10


def test_context_manager_flushes():
    with tempfile.TemporaryDirectory() as d:
        with ParquetLogger(Path(d)) as logger:
            logger.log_rollout(
                run_id="test-001", method="miprov2", seed=42, phase="trial",
                round_num=0, candidate_id=1, example_id=0, score=0.7,
                prompt_tokens=200, completion_tokens=100, wall_clock_ms=500.0,
            )
        # After context manager exit, file should exist
        table = pq.read_table(Path(d) / "rollouts.parquet")
        assert table.num_rows == 1


def test_no_file_when_no_data():
    with tempfile.TemporaryDirectory() as d:
        logger = ParquetLogger(Path(d))
        logger.flush()
        assert not (Path(d) / "rollouts.parquet").exists()
        assert not (Path(d) / "reflections.parquet").exists()
        assert not (Path(d) / "candidates.parquet").exists()
