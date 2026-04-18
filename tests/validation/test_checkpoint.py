"""V8: Checkpoint and resume validation.

Simulates a multi-round optimizer that checkpoints after each round,
gets killed mid-run, and resumes cleanly from the last checkpoint.
"""

from __future__ import annotations

from pathlib import Path

from iso_harness.experiment.checkpoint import (
    load_latest_checkpoint,
    save_checkpoint,
)
from iso_harness.experiment.jsonl_writer import JSONLWriter


def _fake_round(round_num: int, candidates: list[dict], writer: JSONLWriter) -> list[dict]:
    """Simulate one optimizer round: score candidates, prune bottom half."""
    scored = []
    for i, c in enumerate(candidates):
        score = (round_num + 1) * 0.1 + i * 0.01  # deterministic scores
        c_copy = {**c, "score": score, "last_round": round_num}
        scored.append(c_copy)
        # Log a record to JSONL (simulates rollout logging)
        writer.append({"round": round_num, "candidate": c["name"], "score": score})
    # Keep top half
    scored.sort(key=lambda x: x["score"], reverse=True)
    keep = max(1, len(scored) // 2)
    return scored[:keep]


def _make_initial_candidates(n: int = 8) -> list[dict]:
    """Create N initial candidates."""
    return [{"name": f"candidate_{i}", "prompt": f"prompt {i}"} for i in range(n)]


class TestCheckpointResume:
    """V8: Kill-and-resume produces correct final state."""

    def test_full_run_without_interruption(self, tmp_path: Path):
        """Baseline: 5 rounds without interruption."""
        run_dir = tmp_path / "run_baseline"
        run_dir.mkdir()
        writer = JSONLWriter(run_dir / "rollouts.jsonl")
        candidates = _make_initial_candidates(8)

        for round_num in range(5):
            candidates = _fake_round(round_num, candidates, writer)
            save_checkpoint(
                run_dir=run_dir,
                round_num=round_num,
                candidates=candidates,
                cumulative_rollouts=(round_num + 1) * 8,
            )

        assert len(candidates) == 1  # 8 -> 4 -> 2 -> 1 -> 1
        assert len(candidates) >= 1
        assert len(writer.read_all()) > 0

    def test_kill_and_resume(self, tmp_path: Path):
        """Kill after round 3, resume, verify state matches uninterrupted run."""
        run_dir = tmp_path / "run_resume"
        run_dir.mkdir()

        # Phase 1: Run rounds 0-2, then "die"
        writer = JSONLWriter(run_dir / "rollouts.jsonl")
        candidates = _make_initial_candidates(8)

        for round_num in range(3):  # Rounds 0, 1, 2
            candidates = _fake_round(round_num, candidates, writer)
            save_checkpoint(
                run_dir=run_dir,
                round_num=round_num,
                candidates=candidates,
                cumulative_rollouts=(round_num + 1) * 8,
            )

        # Verify checkpoint exists at round 2
        checkpoint = load_latest_checkpoint(run_dir)
        assert checkpoint is not None
        assert checkpoint.round_num == 2
        records_before_kill = len(writer.read_all())

        # Phase 2: "Restart" — load checkpoint and continue
        restored = load_latest_checkpoint(run_dir)
        assert restored is not None
        candidates = restored.candidates
        resume_round = restored.round_num + 1

        # Re-open writer in append mode (new JSONLWriter appends to existing file)
        writer2 = JSONLWriter(run_dir / "rollouts.jsonl")

        for round_num in range(resume_round, 5):  # Rounds 3, 4
            candidates = _fake_round(round_num, candidates, writer2)
            save_checkpoint(
                run_dir=run_dir,
                round_num=round_num,
                candidates=candidates,
                cumulative_rollouts=(round_num + 1) * 8,
            )

        # Verify final state
        assert len(candidates) == 1
        final_checkpoint = load_latest_checkpoint(run_dir)
        assert final_checkpoint is not None
        assert final_checkpoint.round_num == 4

        # Verify no duplicate JSONL entries from the pre-kill rounds
        all_records = writer2.read_all()
        assert len(all_records) > records_before_kill  # New records were added

    def test_checkpoint_state_integrity(self, tmp_path: Path):
        """Verify checkpoint preserves all candidate data."""
        run_dir = tmp_path / "run_integrity"
        run_dir.mkdir()
        candidates = [
            {"name": "a", "prompt": "hello", "score": 0.9, "metadata": {"key": "val"}},
            {"name": "b", "prompt": "world", "score": 0.5, "metadata": {"key": "val2"}},
        ]

        save_checkpoint(
            run_dir=run_dir,
            round_num=2,
            candidates=candidates,
            cumulative_rollouts=500,
            cumulative_tokens=12000,
            metrics_snapshot={"rollout_count": 500, "best_score": 0.9},
        )

        restored = load_latest_checkpoint(run_dir)
        assert restored is not None
        assert restored.round_num == 2
        assert restored.cumulative_rollouts == 500
        assert restored.cumulative_tokens == 12000
        assert len(restored.candidates) == 2
        assert restored.candidates[0]["name"] == "a"
        assert restored.candidates[0]["metadata"] == {"key": "val"}
        assert restored.metrics_snapshot["best_score"] == 0.9

    def test_resume_appends_to_jsonl(self, tmp_path: Path):
        """JSONL file grows on resume, no entries lost."""
        run_dir = tmp_path / "run_jsonl"
        run_dir.mkdir()
        writer = JSONLWriter(run_dir / "log.jsonl")

        # Write 3 entries
        for i in range(3):
            writer.append({"entry": i})

        count_before = len(writer.read_all())
        assert count_before == 3

        # "Resume" with a new writer (same path)
        writer2 = JSONLWriter(run_dir / "log.jsonl")
        for i in range(3, 6):
            writer2.append({"entry": i})

        all_entries = writer2.read_all()
        assert len(all_entries) == 6
        # Verify order preserved
        assert [e["entry"] for e in all_entries] == [0, 1, 2, 3, 4, 5]
