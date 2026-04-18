"""Round-level checkpoint/resume for ISO experiments.

Saves optimizer state after each pruning round using atomic JSON writes.
On resume, detects the latest valid checkpoint and restores state.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CheckpointState(BaseModel):
    """Serializable checkpoint of optimizer state at a round boundary."""

    round_num: int = Field(ge=0)
    candidates: list[dict[str, Any]]
    cumulative_rollouts: int = Field(ge=0)
    cumulative_tokens: int = Field(ge=0)
    metrics_snapshot: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
    git_sha: str = "unknown"


def _atomic_json_write(path: Path, data: Any) -> None:
    """Write JSON atomically: write to .tmp, fsync, rename."""
    tmp = path.with_suffix(".tmp")
    content = json.dumps(data, indent=2, default=str)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def _get_git_sha() -> str:
    """Get current git HEAD SHA, or 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_checkpoint(
    run_dir: Path,
    round_num: int,
    candidates: list[dict[str, Any]],
    cumulative_rollouts: int = 0,
    cumulative_tokens: int = 0,
    metrics_snapshot: dict[str, Any] | None = None,
) -> Path:
    """Save a checkpoint after a completed round.

    Args:
        run_dir: Path to the run directory (e.g., runs/{run_id}/).
        round_num: The round that just completed (0-indexed).
        candidates: List of candidate dicts (prompts + scores + metadata).
        cumulative_rollouts: Total rollouts consumed so far.
        cumulative_tokens: Total tokens consumed so far.
        metrics_snapshot: Optional MetricsCollector state as dict.

    Returns:
        Path to the checkpoint directory that was written.
    """
    checkpoint_dir = Path(run_dir) / "checkpoints" / f"round_{round_num:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = CheckpointState(
        round_num=round_num,
        candidates=candidates,
        cumulative_rollouts=cumulative_rollouts,
        cumulative_tokens=cumulative_tokens,
        metrics_snapshot=metrics_snapshot or {},
        timestamp=datetime.now(timezone.utc),
        git_sha=_get_git_sha(),
    )

    # Write state.json (core state)
    _atomic_json_write(checkpoint_dir / "state.json", state.model_dump(mode="json"))

    # Write candidates.json (detailed candidate data, separate for readability)
    _atomic_json_write(checkpoint_dir / "candidates.json", candidates)

    # Write metrics.json (collector snapshot)
    _atomic_json_write(checkpoint_dir / "metrics.json", metrics_snapshot or {})

    logger.info(
        "Checkpoint saved: round=%d, candidates=%d, rollouts=%d, dir=%s",
        round_num,
        len(candidates),
        cumulative_rollouts,
        checkpoint_dir,
    )
    return checkpoint_dir


def load_latest_checkpoint(run_dir: Path) -> CheckpointState | None:
    """Find and load the latest valid checkpoint in a run directory.

    Scans checkpoints/ for the highest round_NNN directory with a valid
    state.json. If the latest is corrupt, falls back to the previous.

    Returns:
        CheckpointState if a valid checkpoint exists, None otherwise.
    """
    checkpoints_dir = Path(run_dir) / "checkpoints"
    if not checkpoints_dir.exists():
        return None

    # Find all round directories, sorted descending
    round_dirs = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("round_")],
        key=lambda d: d.name,
        reverse=True,
    )

    for round_dir in round_dirs:
        state_path = round_dir / "state.json"
        if not state_path.exists():
            logger.warning("Checkpoint dir missing state.json: %s", round_dir)
            continue

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = CheckpointState.model_validate(data)
            logger.info(
                "Loaded checkpoint: round=%d, candidates=%d, rollouts=%d",
                state.round_num,
                len(state.candidates),
                state.cumulative_rollouts,
            )
            return state
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Corrupt checkpoint at %s: %s", round_dir, e)
            continue

    return None


def find_resumable_run(
    runs_dir: Path,
    benchmark: str,
    optimizer: str,
    seed: int,
) -> Path | None:
    """Find an incomplete run matching the given config.

    Looks for run directories that have checkpoints but no COMPLETE marker.
    Matches on config.json fields: benchmark, optimizer, seed.

    Returns:
        Path to the resumable run directory, or None.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return None

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        # Skip completed runs
        if (run_dir / "COMPLETE").exists():
            continue

        # Must have checkpoints
        if not (run_dir / "checkpoints").exists():
            continue

        # Check config match
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if (
                config.get("benchmark") == benchmark
                and config.get("optimizer") == optimizer
                and config.get("seed") == seed
            ):
                logger.info("Found resumable run: %s", run_dir)
                return run_dir
        except (json.JSONDecodeError, KeyError):
            continue

    return None
