"""V7: Logging validation — MLflow, LoggingLM JSONL, COMPLETE marker."""

from __future__ import annotations

from pathlib import Path

from iso_harness.experiment.jsonl_writer import JSONLWriter, write_complete_marker
from iso_harness.experiment.schemas import RolloutRecord


class TestLoggingValidation:
    def test_jsonl_writer_produces_valid_output(self, tmp_path: Path):
        """LoggingLM-style JSONL output is valid."""
        import uuid
        from datetime import datetime, timezone

        writer = JSONLWriter(tmp_path / "rollouts.jsonl")
        record = RolloutRecord(
            rollout_id=str(uuid.uuid4()),
            run_id="test-run",
            round_num=0,
            candidate_id="cand-1",
            module_name=None,
            example_id="ex-1",
            prompt="test prompt",
            response="test response",
            score=0.5,
            feedback="",
            metadata={},
            tokens_input=10,
            tokens_output=5,
            latency_ms=100.0,
            timestamp=datetime.now(timezone.utc),
        )
        writer.append(record)

        records = writer.read_all()
        assert len(records) == 1
        assert records[0]["run_id"] == "test-run"
        assert records[0]["score"] == 0.5

    def test_complete_marker_written(self, tmp_path: Path):
        """COMPLETE marker file created with expected fields."""
        import json

        run_dir = tmp_path / "test_run"
        write_complete_marker(run_dir, run_id="test-123")

        marker = run_dir / "COMPLETE"
        assert marker.exists()

        data = json.loads(marker.read_text())
        assert "timestamp" in data
        assert "git_sha" in data
        assert data["run_id"] == "test-123"
        # No .tmp file should remain
        assert not (run_dir / "COMPLETE.tmp").exists()

    def test_jsonl_concurrent_integrity(self, tmp_path: Path):
        """Multiple writes produce valid JSONL (no corruption)."""
        import threading
        import uuid
        from datetime import datetime, timezone

        writer = JSONLWriter(tmp_path / "concurrent.jsonl")

        def write_batch(n: int):
            for _ in range(n):
                record = RolloutRecord(
                    rollout_id=str(uuid.uuid4()),
                    run_id="concurrent-test",
                    round_num=0,
                    candidate_id="c",
                    module_name=None,
                    example_id="e",
                    prompt="p",
                    response="r",
                    score=0.5,
                    feedback="",
                    metadata={},
                    tokens_input=1,
                    tokens_output=1,
                    latency_ms=1.0,
                    timestamp=datetime.now(timezone.utc),
                )
                writer.append(record)

        threads = [threading.Thread(target=write_batch, args=(25,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        records = writer.read_all()
        assert len(records) == 100

    def test_mlflow_importable(self):
        """MLflow is importable and autolog function exists."""
        import mlflow
        assert hasattr(mlflow, "dspy")
        assert hasattr(mlflow.dspy, "autolog")
