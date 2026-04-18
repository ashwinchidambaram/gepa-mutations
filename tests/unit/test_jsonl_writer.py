"""Comprehensive tests for JSONLWriter and write_complete_marker.

Tests cover:
- Roundtrip: write N records, read back, verify integrity
- Concurrent writes: multiple threads writing simultaneously
- Partial read safety: reader never sees corrupt lines during writes
- Validation errors: invalid records logged separately, not written
- Empty file: read_all on non-existent file returns empty list
- COMPLETE marker: atomic marker file with expected contents
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from iso_harness.experiment.jsonl_writer import JSONLWriter, write_complete_marker
from iso_harness.experiment.schemas import RolloutRecord


# ---------------------------------------------------------------------------
# Helper — create a valid RolloutRecord with sensible defaults
# ---------------------------------------------------------------------------

_NOW = datetime.now(timezone.utc)


def _make_rollout(**overrides) -> RolloutRecord:
    """Return a valid RolloutRecord with sensible defaults.

    Any field can be overridden via keyword arguments.
    """
    defaults = dict(
        rollout_id=str(uuid.uuid4()),
        run_id="run-test",
        round_num=0,
        candidate_id="cand-1",
        module_name=None,
        example_id="ex-42",
        prompt="What is 2+2?",
        response="4",
        score=0.85,
        feedback="Correct.",
        metadata={"key": "val"},
        tokens_input=100,
        tokens_output=20,
        latency_ms=123.4,
        timestamp=_NOW,
    )
    defaults.update(overrides)
    return RolloutRecord(**defaults)


# ======================================================================
# a) Roundtrip test — write 100 records, read back, verify all parse
# ======================================================================


class TestRoundtrip:
    """Write 100 RolloutRecord instances, read back, verify integrity."""

    def test_roundtrip_100_records(self, tmp_path: Path) -> None:
        path = tmp_path / "rollouts.jsonl"
        writer = JSONLWriter(path)

        originals = [_make_rollout(round_num=i) for i in range(100)]

        for record in originals:
            writer.append(record)

        # Read back
        records = writer.read_all()
        assert len(records) == 100

        # Every record is valid JSON and can be parsed as RolloutRecord
        for i, rec in enumerate(records):
            assert isinstance(rec, dict)
            parsed = RolloutRecord.model_validate(rec)
            assert parsed.round_num == i
            assert parsed.score == 0.85
            assert parsed.run_id == "run-test"

    def test_roundtrip_dict_records(self, tmp_path: Path) -> None:
        """Dicts without schema are written as-is."""
        path = tmp_path / "dicts.jsonl"
        writer = JSONLWriter(path)

        for i in range(50):
            writer.append({"index": i, "value": f"item-{i}"})

        records = writer.read_all()
        assert len(records) == 50
        for i, rec in enumerate(records):
            assert rec["index"] == i
            assert rec["value"] == f"item-{i}"

    def test_len_matches_read_all(self, tmp_path: Path) -> None:
        """__len__ returns the same count as read_all()."""
        path = tmp_path / "len_test.jsonl"
        writer = JSONLWriter(path)

        for _ in range(25):
            writer.append(_make_rollout())

        assert len(writer) == 25
        assert len(writer) == len(writer.read_all())


# ======================================================================
# b) Concurrent write test — 4 threads x 50 records = 200 total
# ======================================================================


class TestConcurrentWrites:
    """4 threads write 50 records each to the SAME writer simultaneously."""

    def test_concurrent_200_records(self, tmp_path: Path) -> None:
        path = tmp_path / "concurrent.jsonl"
        writer = JSONLWriter(path)

        errors: list[Exception] = []

        def write_batch(thread_id: int) -> None:
            try:
                for i in range(50):
                    writer.append(
                        _make_rollout(
                            run_id=f"thread-{thread_id}",
                            round_num=i,
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_batch, args=(t,))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions in any thread
        assert errors == [], f"Thread errors: {errors}"

        # All 200 records present and valid JSON
        records = writer.read_all()
        assert len(records) == 200

        # Every record parses as valid JSON (already parsed by read_all)
        for rec in records:
            assert "run_id" in rec
            assert rec["run_id"].startswith("thread-")

        # Verify per-thread counts
        from collections import Counter
        thread_counts = Counter(r["run_id"] for r in records)
        for t in range(4):
            assert thread_counts[f"thread-{t}"] == 50


# ======================================================================
# c) Partial read safety — reader never sees corrupt lines
# ======================================================================


class TestPartialReadSafety:
    """Background writer + foreground reader: reader never sees corrupt data."""

    def test_concurrent_read_during_writes(self, tmp_path: Path) -> None:
        path = tmp_path / "partial_safety.jsonl"
        writer = JSONLWriter(path)

        write_done = threading.Event()
        reader_errors: list[str] = []

        def background_writer() -> None:
            for i in range(100):
                writer.append(_make_rollout(round_num=i))
                time.sleep(0.001)
            write_done.set()

        writer_thread = threading.Thread(target=background_writer)
        writer_thread.start()

        # Repeatedly read while writes are happening
        read_count = 0
        while not write_done.is_set() or read_count < 5:
            records = writer.read_all()
            # Every record returned must be valid JSON (already parsed)
            for rec in records:
                if not isinstance(rec, dict):
                    reader_errors.append(f"Non-dict record: {rec!r}")
                elif "rollout_id" not in rec:
                    reader_errors.append(f"Missing rollout_id: {rec!r}")
            read_count += 1
            time.sleep(0.005)

        writer_thread.join()

        assert reader_errors == [], f"Reader saw corrupt data: {reader_errors}"

        # Final read should have all 100 records
        final_records = writer.read_all()
        assert len(final_records) == 100


# ======================================================================
# d) Validation error test — invalid records go to error file
# ======================================================================


class TestValidationErrors:
    """Invalid dict records are rejected and logged to a sibling file."""

    def test_invalid_score_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "validated.jsonl"
        writer = JSONLWriter(path, schema=RolloutRecord)

        # Write a valid record first (as a Pydantic model — bypasses validation)
        valid = _make_rollout(score=0.5)
        writer.append(valid)

        # Write an invalid dict (score > 1.0)
        invalid_dict = {
            "rollout_id": str(uuid.uuid4()),
            "run_id": "run-test",
            "round_num": 0,
            "candidate_id": "cand-1",
            "example_id": "ex-42",
            "prompt": "What is 2+2?",
            "response": "4",
            "score": 2.0,  # Invalid: > 1.0
            "feedback": "Correct.",
            "tokens_input": 100,
            "tokens_output": 20,
            "latency_ms": 123.4,
            "timestamp": _NOW.isoformat(),
        }
        writer.append(invalid_dict)

        # Main file should only have the valid record
        records = writer.read_all()
        assert len(records) == 1
        assert records[0]["score"] == 0.5

        # Validation error count should be 1
        assert writer.validation_error_count == 1

        # Error file should exist and contain the invalid record
        error_path = path.with_name("validated_validation_errors.jsonl")
        assert error_path.exists()

        error_writer = JSONLWriter(error_path)
        error_records = error_writer.read_all()
        assert len(error_records) == 1
        assert error_records[0]["original_record"]["score"] == 2.0
        assert "errors" in error_records[0]
        assert "timestamp" in error_records[0]

    def test_valid_dict_passes_validation(self, tmp_path: Path) -> None:
        """A valid dict should pass schema validation and be written."""
        path = tmp_path / "valid_dict.jsonl"
        writer = JSONLWriter(path, schema=RolloutRecord)

        valid_dict = {
            "rollout_id": str(uuid.uuid4()),
            "run_id": "run-test",
            "round_num": 0,
            "candidate_id": "cand-1",
            "example_id": "ex-42",
            "prompt": "What is 2+2?",
            "response": "4",
            "score": 0.75,
            "feedback": "Correct.",
            "tokens_input": 100,
            "tokens_output": 20,
            "latency_ms": 123.4,
            "timestamp": _NOW.isoformat(),
        }
        writer.append(valid_dict)

        records = writer.read_all()
        assert len(records) == 1
        assert writer.validation_error_count == 0

    def test_pydantic_model_bypasses_validation(self, tmp_path: Path) -> None:
        """Pydantic model instances are already validated; no re-check."""
        path = tmp_path / "model_bypass.jsonl"
        writer = JSONLWriter(path, schema=RolloutRecord)

        record = _make_rollout()
        writer.append(record)

        records = writer.read_all()
        assert len(records) == 1
        assert writer.validation_error_count == 0

    def test_multiple_validation_errors(self, tmp_path: Path) -> None:
        """Multiple invalid records are all counted and logged."""
        path = tmp_path / "multi_err.jsonl"
        writer = JSONLWriter(path, schema=RolloutRecord)

        for i in range(5):
            writer.append({"bad_field": i})  # Missing required fields

        assert writer.validation_error_count == 5
        assert len(writer.read_all()) == 0  # No valid records written

        error_path = path.with_name("multi_err_validation_errors.jsonl")
        error_writer = JSONLWriter(error_path)
        assert len(error_writer.read_all()) == 5


# ======================================================================
# e) Empty file test — read_all on non-existent file
# ======================================================================


class TestEmptyFile:
    """read_all on non-existent or empty files returns empty list."""

    def test_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "does_not_exist.jsonl"
        writer = JSONLWriter(path)
        assert writer.read_all() == []

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.touch()
        writer = JSONLWriter(path)
        assert writer.read_all() == []

    def test_len_nonexistent_returns_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "no_file.jsonl"
        writer = JSONLWriter(path)
        assert len(writer) == 0


# ======================================================================
# f) COMPLETE marker test
# ======================================================================


class TestCompleteMarker:
    """write_complete_marker creates an atomic marker file."""

    def test_marker_file_created(self, tmp_path: Path) -> None:
        write_complete_marker(tmp_path, run_id="test-123")

        marker_path = tmp_path / "COMPLETE"
        assert marker_path.exists()

        # No temp file remains
        tmp_marker = tmp_path / "COMPLETE.tmp"
        assert not tmp_marker.exists()

        # Contents are valid JSON with expected fields
        with open(marker_path) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "git_sha" in data
        assert data["run_id"] == "test-123"

        # Timestamp is ISO format
        datetime.fromisoformat(data["timestamp"])

    def test_marker_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        write_complete_marker(nested, run_id="nested-run")

        assert (nested / "COMPLETE").exists()

    def test_marker_default_run_id(self, tmp_path: Path) -> None:
        write_complete_marker(tmp_path)

        with open(tmp_path / "COMPLETE") as f:
            data = json.load(f)

        assert data["run_id"] == ""


# ======================================================================
# Additional edge cases
# ======================================================================


class TestEdgeCases:
    """Edge cases for writer behavior."""

    def test_parent_dir_created_on_init(self, tmp_path: Path) -> None:
        """JSONLWriter creates parent directories if they don't exist."""
        deep_path = tmp_path / "a" / "b" / "c" / "output.jsonl"
        writer = JSONLWriter(deep_path)
        assert deep_path.parent.exists()

        writer.append({"test": True})
        assert deep_path.exists()

    def test_datetime_serialization(self, tmp_path: Path) -> None:
        """datetime objects in dicts are serialized via default=str."""
        path = tmp_path / "dates.jsonl"
        writer = JSONLWriter(path)

        now = datetime.now(timezone.utc)
        writer.append({"timestamp": now, "value": 42})

        records = writer.read_all()
        assert len(records) == 1
        # datetime was serialized to string
        assert isinstance(records[0]["timestamp"], str)

    def test_no_schema_allows_any_dict(self, tmp_path: Path) -> None:
        """Without a schema, any dict is written without validation."""
        path = tmp_path / "any.jsonl"
        writer = JSONLWriter(path)

        writer.append({"anything": "goes"})
        writer.append({"nested": {"a": [1, 2, 3]}})

        records = writer.read_all()
        assert len(records) == 2
        assert writer.validation_error_count == 0
