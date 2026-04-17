"""Tests for parameter-weighted and token-weighted cost models."""

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gepa_mutations.cost_model import (
    parameter_weighted_cost,
    token_weighted_cost,
    parameter_weighted_cost_from_parquet,
    token_weighted_cost_from_parquet,
)


# --- Parameter-weighted cost ---

def test_self_analyzer_cost_equals_raw():
    """When task and reflection models are the same size, weight = 1."""
    cost = parameter_weighted_cost(
        rollouts=1000, reflection_calls=50,
        task_param_b=8.0, reflection_param_b=8.0,
    )
    assert cost == pytest.approx(1050.0)


def test_external_analyzer_weights_reflections():
    """32B reflection calls cost 4x 8B rollout calls."""
    cost = parameter_weighted_cost(
        rollouts=1000, reflection_calls=50,
        task_param_b=8.0, reflection_param_b=32.0,
    )
    assert cost == pytest.approx(1200.0)


def test_zero_reflections():
    cost = parameter_weighted_cost(
        rollouts=500, reflection_calls=0,
        task_param_b=8.0, reflection_param_b=32.0,
    )
    assert cost == pytest.approx(500.0)


# --- Token-weighted cost ---

def test_token_weighted_self_analyzer():
    """Self analyzer: all tokens weighted by task_param_b."""
    cost = token_weighted_cost(
        rollout_tokens=10000, reflection_tokens=5000,
        task_param_b=8.0, reflection_param_b=8.0,
    )
    # Both at 8B: 10000*8 + 5000*8 = 120000
    assert cost == pytest.approx(120000.0)


def test_token_weighted_external_analyzer():
    """External analyzer: reflection tokens weighted by 32B."""
    cost = token_weighted_cost(
        rollout_tokens=10000, reflection_tokens=5000,
        task_param_b=8.0, reflection_param_b=32.0,
    )
    # 10000*8 + 5000*32 = 80000 + 160000 = 240000
    assert cost == pytest.approx(240000.0)


# --- From Parquet ---

def _write_test_parquet(tmp_dir: Path):
    """Write minimal test Parquet files."""
    from gepa_mutations.logging.parquet_logger import (
        _ROLLOUT_SCHEMA,
        _REFLECTION_SCHEMA,
    )

    rollouts = pa.Table.from_pylist([
        {"run_id": "r1", "method": "gepa", "seed": 42, "phase": "eval",
         "round": 0, "candidate_id": 0, "example_id": i, "score": 0.5,
         "prompt_tokens": 100, "completion_tokens": 50, "wall_clock_ms": 10.0,
         "timestamp": 0.0}
        for i in range(10)
    ], schema=_ROLLOUT_SCHEMA)
    pq.write_table(rollouts, tmp_dir / "rollouts.parquet")

    reflections = pa.Table.from_pylist([
        {"run_id": "r1", "method": "gepa", "seed": 42, "phase": "mutation",
         "round": 0, "call_type": "proposal", "input_tokens": 200,
         "output_tokens": 100, "wall_clock_ms": 500.0,
         "input_hash": "a", "output_hash": "b",
         "model_id": "qwen3-32b", "param_count_b": 32.0, "timestamp": 0.0}
        for _ in range(5)
    ], schema=_REFLECTION_SCHEMA)
    pq.write_table(reflections, tmp_dir / "reflections.parquet")


def test_parameter_weighted_from_parquet():
    with tempfile.TemporaryDirectory() as d:
        _write_test_parquet(Path(d))
        cost = parameter_weighted_cost_from_parquet(Path(d), task_param_b=8.0)
        # 10 rollouts + 5 reflections * (32/8) = 10 + 20 = 30
        assert cost == pytest.approx(30.0)


def test_token_weighted_from_parquet():
    with tempfile.TemporaryDirectory() as d:
        _write_test_parquet(Path(d))
        cost = token_weighted_cost_from_parquet(Path(d), task_param_b=8.0)
        # rollouts: 10 * (100+50) * 8 = 12000
        # reflections: 5 * (200+100) * 32 = 48000
        # total = 60000
        assert cost == pytest.approx(60000.0)


def test_no_reflections_parquet():
    """When there's no reflections.parquet, cost = rollout count (pw) or rollout tokens (tw)."""
    with tempfile.TemporaryDirectory() as d:
        from gepa_mutations.logging.parquet_logger import _ROLLOUT_SCHEMA
        rollouts = pa.Table.from_pylist([
            {"run_id": "r1", "method": "gepa", "seed": 42, "phase": "eval",
             "round": 0, "candidate_id": 0, "example_id": 0, "score": 0.5,
             "prompt_tokens": 100, "completion_tokens": 50, "wall_clock_ms": 10.0,
             "timestamp": 0.0}
        ], schema=_ROLLOUT_SCHEMA)
        pq.write_table(rollouts, Path(d) / "rollouts.parquet")

        pw = parameter_weighted_cost_from_parquet(Path(d), task_param_b=8.0)
        assert pw == pytest.approx(1.0)  # 1 rollout, no reflections

        tw = token_weighted_cost_from_parquet(Path(d), task_param_b=8.0)
        assert tw == pytest.approx(1200.0)  # (100+50)*8 = 1200
