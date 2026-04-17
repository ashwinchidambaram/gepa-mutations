"""Unified Parquet logging for rollouts, reflections, and candidates.

All three methods (GEPA, MIPROv2, ISO) emit logs in these schemas,
enabling cross-method cost analysis and Pareto plotting.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_ROLLOUT_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("method", pa.string()),
    ("seed", pa.int32()),
    ("phase", pa.string()),
    ("round", pa.int16()),
    ("candidate_id", pa.int32()),
    ("example_id", pa.int32()),
    ("score", pa.float64()),
    ("prompt_tokens", pa.int32()),
    ("completion_tokens", pa.int32()),
    ("wall_clock_ms", pa.float64()),
    ("timestamp", pa.float64()),
])

_REFLECTION_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("method", pa.string()),
    ("seed", pa.int32()),
    ("phase", pa.string()),
    ("round", pa.int16()),
    ("call_type", pa.string()),       # discovery | mutation | proposal | merge
    ("input_tokens", pa.int32()),
    ("output_tokens", pa.int32()),
    ("wall_clock_ms", pa.float64()),
    ("input_hash", pa.string()),
    ("output_hash", pa.string()),
    ("model_id", pa.string()),        # which model handled this call
    ("param_count_b", pa.float32()),   # active params in billions
    ("timestamp", pa.float64()),
])

_CANDIDATE_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("candidate_id", pa.int32()),
    ("generation", pa.int16()),
    ("parent_ids", pa.string()),       # JSON-encoded list
    ("strategy_name", pa.string()),
    ("strategy_technique", pa.string()),
    ("prompt_text", pa.string()),
    ("created_at", pa.float64()),
])


class ParquetLogger:
    """Buffered Parquet logger for experiment data.

    Usage::

        with ParquetLogger(output_dir) as logger:
            logger.log_rollout(...)
            logger.log_reflection(...)
        # files written on exit
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._rollouts: list[dict[str, Any]] = []
        self._reflections: list[dict[str, Any]] = []
        self._candidates: list[dict[str, Any]] = []

    def __enter__(self) -> ParquetLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.flush()

    def log_rollout(
        self,
        *,
        run_id: str,
        method: str,
        seed: int,
        phase: str,
        round_num: int,
        candidate_id: int,
        example_id: int,
        score: float,
        prompt_tokens: int,
        completion_tokens: int,
        wall_clock_ms: float,
    ) -> None:
        self._rollouts.append({
            "run_id": run_id,
            "method": method,
            "seed": seed,
            "phase": phase,
            "round": round_num,
            "candidate_id": candidate_id,
            "example_id": example_id,
            "score": score,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "wall_clock_ms": wall_clock_ms,
            "timestamp": time.time(),
        })

    def log_reflection(
        self,
        *,
        run_id: str,
        method: str,
        seed: int,
        phase: str,
        round_num: int,
        call_type: str,
        input_tokens: int,
        output_tokens: int,
        wall_clock_ms: float,
        input_hash: str,
        output_hash: str,
        model_id: str,
        param_count_b: float,
    ) -> None:
        self._reflections.append({
            "run_id": run_id,
            "method": method,
            "seed": seed,
            "phase": phase,
            "round": round_num,
            "call_type": call_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "wall_clock_ms": wall_clock_ms,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "model_id": model_id,
            "param_count_b": param_count_b,
            "timestamp": time.time(),
        })

    def log_candidate(
        self,
        *,
        run_id: str,
        candidate_id: int,
        generation: int,
        parent_ids: list[int],
        strategy_name: str,
        strategy_technique: str,
        prompt_text: str,
    ) -> None:
        import json
        self._candidates.append({
            "run_id": run_id,
            "candidate_id": candidate_id,
            "generation": generation,
            "parent_ids": json.dumps(parent_ids),
            "strategy_name": strategy_name,
            "strategy_technique": strategy_technique,
            "prompt_text": prompt_text,
            "created_at": time.time(),
        })

    def flush(self) -> None:
        """Write buffered data to Parquet files."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._flush_table(self._rollouts, _ROLLOUT_SCHEMA, "rollouts.parquet")
        self._flush_table(self._reflections, _REFLECTION_SCHEMA, "reflections.parquet")
        self._flush_table(self._candidates, _CANDIDATE_SCHEMA, "candidates.parquet")

    def _flush_table(
        self, rows: list[dict[str, Any]], schema: pa.Schema, filename: str
    ) -> None:
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=schema)
        path = self._output_dir / filename
        if path.exists():
            existing = pq.read_table(path)
            table = pa.concat_tables([existing, table])
        pq.write_table(table, path)
        rows.clear()
