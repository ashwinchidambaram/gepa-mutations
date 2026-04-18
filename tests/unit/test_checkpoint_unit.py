"""Unit tests for round-level checkpoint/resume system.

Tests cover:
- Save and load roundtrip
- Load latest picks highest round
- Corrupt checkpoint fallback
- Empty / missing directory handling
- find_resumable_run with COMPLETE marker logic
- Atomic write safety (no .tmp residue)
- Pydantic validation on CheckpointState
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from iso_harness.experiment.checkpoint import (
    CheckpointState,
    find_resumable_run,
    load_latest_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidates(n: int = 3) -> list[dict]:
    """Create n dummy candidate dicts."""
    return [
        {
            "candidate_id": f"cand-{i}",
            "prompt": f"System prompt variant {i}",
            "score": round(0.5 + i * 0.1, 2),
            "rollouts": 10,
        }
        for i in range(n)
    ]


def _make_config(benchmark: str = "hotpotqa", optimizer: str = "iso", seed: int = 42) -> dict:
    """Create a minimal config dict matching find_resumable_run expectations."""
    return {"benchmark": benchmark, "optimizer": optimizer, "seed": seed}


# ======================================================================
# a) Save and load roundtrip
# ======================================================================


class TestSaveLoadRoundtrip:
    """Save a checkpoint with 3 candidates, load it back, verify all fields."""

    def test_roundtrip_fields_match(self, tmp_path: Path) -> None:
        candidates = _make_candidates(3)
        metrics = {"mean_score": 0.72, "total_rollouts": 50}

        ckpt_dir = save_checkpoint(
            run_dir=tmp_path,
            round_num=1,
            candidates=candidates,
            cumulative_rollouts=50,
            cumulative_tokens=12000,
            metrics_snapshot=metrics,
        )

        # Checkpoint dir was created
        assert ckpt_dir.exists()
        assert (ckpt_dir / "state.json").exists()
        assert (ckpt_dir / "candidates.json").exists()
        assert (ckpt_dir / "metrics.json").exists()

        # Load it back
        state = load_latest_checkpoint(tmp_path)
        assert state is not None
        assert state.round_num == 1
        assert len(state.candidates) == 3
        assert state.cumulative_rollouts == 50
        assert state.cumulative_tokens == 12000
        assert state.metrics_snapshot == metrics
        assert state.git_sha  # non-empty string

        # Candidates content preserved
        assert state.candidates[0]["candidate_id"] == "cand-0"
        assert state.candidates[2]["score"] == 0.7


# ======================================================================
# b) Load latest picks highest round
# ======================================================================


class TestLoadLatestPicksHighest:
    """Save rounds 0, 1, 2. load_latest_checkpoint returns round 2."""

    def test_returns_highest_round(self, tmp_path: Path) -> None:
        for r in range(3):
            save_checkpoint(
                run_dir=tmp_path,
                round_num=r,
                candidates=_make_candidates(3 - r),
                cumulative_rollouts=(r + 1) * 100,
            )

        state = load_latest_checkpoint(tmp_path)
        assert state is not None
        assert state.round_num == 2
        assert len(state.candidates) == 1  # 3 - 2 = 1
        assert state.cumulative_rollouts == 300


# ======================================================================
# c) Corrupt checkpoint fallback
# ======================================================================


class TestCorruptCheckpointFallback:
    """Corrupt the latest checkpoint; loader falls back to previous valid one."""

    def test_falls_back_on_corrupt_json(self, tmp_path: Path) -> None:
        # Save 3 valid rounds
        for r in range(3):
            save_checkpoint(
                run_dir=tmp_path,
                round_num=r,
                candidates=_make_candidates(2),
                cumulative_rollouts=(r + 1) * 50,
            )

        # Corrupt round_002/state.json
        corrupt_path = tmp_path / "checkpoints" / "round_002" / "state.json"
        corrupt_path.write_text("{this is not valid json!!!", encoding="utf-8")

        state = load_latest_checkpoint(tmp_path)
        assert state is not None
        assert state.round_num == 1
        assert state.cumulative_rollouts == 100


# ======================================================================
# d) Empty directory returns None
# ======================================================================


class TestEmptyDirectory:
    """load_latest_checkpoint on empty checkpoints dir returns None."""

    def test_empty_checkpoints_dir(self, tmp_path: Path) -> None:
        (tmp_path / "checkpoints").mkdir()
        result = load_latest_checkpoint(tmp_path)
        assert result is None


# ======================================================================
# e) No checkpoints dir returns None
# ======================================================================


class TestNoCheckpointsDir:
    """load_latest_checkpoint on dir without checkpoints/ returns None."""

    def test_no_checkpoints_subdir(self, tmp_path: Path) -> None:
        result = load_latest_checkpoint(tmp_path)
        assert result is None


# ======================================================================
# f) find_resumable_run — returns incomplete run
# ======================================================================


class TestFindResumableRun:
    """Two run dirs: one complete, one with checkpoints but no COMPLETE."""

    def test_finds_incomplete_run(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        # Run A: completed (has COMPLETE marker)
        run_a = runs_dir / "run-a"
        run_a.mkdir()
        (run_a / "checkpoints").mkdir()
        (run_a / "COMPLETE").touch()
        with open(run_a / "config.json", "w") as f:
            json.dump(_make_config(), f)

        # Run B: incomplete (has checkpoints, no COMPLETE)
        run_b = runs_dir / "run-b"
        run_b.mkdir()
        (run_b / "checkpoints").mkdir()
        with open(run_b / "config.json", "w") as f:
            json.dump(_make_config(), f)

        result = find_resumable_run(runs_dir, benchmark="hotpotqa", optimizer="iso", seed=42)
        assert result is not None
        assert result.name == "run-b"

    def test_skips_runs_without_checkpoints(self, tmp_path: Path) -> None:
        """A run with config but no checkpoints dir is not resumable."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        run_dir = runs_dir / "run-no-ckpt"
        run_dir.mkdir()
        with open(run_dir / "config.json", "w") as f:
            json.dump(_make_config(), f)

        result = find_resumable_run(runs_dir, benchmark="hotpotqa", optimizer="iso", seed=42)
        assert result is None


# ======================================================================
# g) find_resumable_run — no match returns None
# ======================================================================


class TestFindResumableRunNoMatch:
    """When config doesn't match, returns None."""

    def test_no_config_match(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        run_dir = runs_dir / "run-x"
        run_dir.mkdir()
        (run_dir / "checkpoints").mkdir()
        with open(run_dir / "config.json", "w") as f:
            json.dump(_make_config(benchmark="hover", optimizer="iso", seed=42), f)

        # Search for a different benchmark
        result = find_resumable_run(runs_dir, benchmark="hotpotqa", optimizer="iso", seed=42)
        assert result is None

    def test_seed_mismatch(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        run_dir = runs_dir / "run-y"
        run_dir.mkdir()
        (run_dir / "checkpoints").mkdir()
        with open(run_dir / "config.json", "w") as f:
            json.dump(_make_config(seed=99), f)

        result = find_resumable_run(runs_dir, benchmark="hotpotqa", optimizer="iso", seed=42)
        assert result is None

    def test_nonexistent_runs_dir(self, tmp_path: Path) -> None:
        result = find_resumable_run(
            tmp_path / "nonexistent", benchmark="hotpotqa", optimizer="iso", seed=42
        )
        assert result is None


# ======================================================================
# h) Atomic write safety — no .tmp files remain
# ======================================================================


class TestAtomicWriteSafety:
    """Verify no .tmp files remain after save_checkpoint."""

    def test_no_tmp_files_after_save(self, tmp_path: Path) -> None:
        save_checkpoint(
            run_dir=tmp_path,
            round_num=0,
            candidates=_make_candidates(5),
            cumulative_rollouts=200,
            cumulative_tokens=50000,
            metrics_snapshot={"mean_score": 0.65},
        )

        # Recursively check for any .tmp files
        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert tmp_files == [], f"Leftover .tmp files: {tmp_files}"


# ======================================================================
# i) CheckpointState validation — pydantic constraints
# ======================================================================


class TestCheckpointStateValidation:
    """Pydantic validation catches invalid field values."""

    def test_negative_rollouts_raises(self) -> None:
        with pytest.raises(ValidationError):
            CheckpointState(
                round_num=0,
                candidates=[],
                cumulative_rollouts=-1,
                cumulative_tokens=0,
                timestamp="2026-01-01T00:00:00Z",
            )

    def test_negative_round_num_raises(self) -> None:
        with pytest.raises(ValidationError):
            CheckpointState(
                round_num=-1,
                candidates=[],
                cumulative_rollouts=0,
                cumulative_tokens=0,
                timestamp="2026-01-01T00:00:00Z",
            )

    def test_negative_tokens_raises(self) -> None:
        with pytest.raises(ValidationError):
            CheckpointState(
                round_num=0,
                candidates=[],
                cumulative_rollouts=0,
                cumulative_tokens=-100,
                timestamp="2026-01-01T00:00:00Z",
            )

    def test_valid_state_with_empty_candidates(self) -> None:
        state = CheckpointState(
            round_num=0,
            candidates=[],
            cumulative_rollouts=0,
            cumulative_tokens=0,
            timestamp="2026-01-01T00:00:00Z",
        )
        assert state.round_num == 0
        assert state.candidates == []

    def test_missing_required_fields_raises(self) -> None:
        with pytest.raises(ValidationError):
            CheckpointState(round_num=0)  # type: ignore[call-arg]
