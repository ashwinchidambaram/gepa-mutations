"""Atomic append-only JSONL writer with optional Pydantic validation.

Each line is written atomically (flush + fsync after each append) so concurrent
readers (e.g., rsync) never see partial lines. Validation errors are logged to
a sibling file rather than raising, so expensive experiment runs aren't halted
by logging bugs.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class JSONLWriter:
    """Append-only JSONL writer with optional Pydantic schema validation."""

    def __init__(self, path: Path, schema: type[BaseModel] | None = None) -> None:
        self.path = Path(path)
        self.schema = schema
        self._lock = threading.Lock()
        self._validation_error_count = 0
        self._error_writer: JSONLWriter | None = None
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def validation_error_count(self) -> int:
        return self._validation_error_count

    def append(self, record: BaseModel | dict[str, Any]) -> None:
        """Append a single record as one JSON line.

        If a schema is set, validates before writing. Validation failures
        are logged to {stem}_validation_errors.jsonl and counted, not raised.
        """
        # Serialize
        if isinstance(record, BaseModel):
            line = record.model_dump_json()
        else:
            line = json.dumps(record, default=str)

        # Optional validation (only if schema set and record is a dict)
        if self.schema is not None and isinstance(record, dict):
            try:
                self.schema.model_validate(record)
            except ValidationError as e:
                self._validation_error_count += 1
                self._log_validation_error(record, e)
                if self._validation_error_count > 10:
                    logger.warning(
                        "JSONL validation error count exceeded 10 for %s (%d total)",
                        self.path,
                        self._validation_error_count,
                    )
                return  # Don't write invalid records

        # Atomic append: lock -> write line -> flush -> fsync
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    def _log_validation_error(self, record: dict, error: ValidationError) -> None:
        """Log validation error to a sibling file."""
        if self._error_writer is None:
            error_path = self.path.with_name(
                self.path.stem + "_validation_errors" + self.path.suffix
            )
            # No schema on error writer to avoid recursion
            self._error_writer = JSONLWriter(error_path)

        error_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_record": record,
            "errors": error.errors(),
        }
        self._error_writer.append(error_entry)

    def read_all(self) -> list[dict[str, Any]]:
        """Read all complete lines from the JSONL file.

        Skips any trailing incomplete line (e.g., from a crash mid-write).
        Returns empty list if file doesn't exist.
        """
        if not self.path.exists():
            return []

        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip incomplete/corrupt lines (e.g., from crash)
                    logger.warning("Skipping corrupt JSONL line in %s", self.path)
        return records

    def __len__(self) -> int:
        """Count of records written (by reading the file)."""
        return len(self.read_all())


def write_complete_marker(run_dir: Path, run_id: str = "") -> None:
    """Write a COMPLETE marker file indicating successful run completion.

    The marker is used by sync_from_pod.sh to detect finished runs.
    Uses atomic write (tmp + rename) to avoid partial reads.
    """
    import subprocess

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Get git SHA
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_sha = "unknown"

    marker_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "run_id": run_id,
    }

    marker_path = run_dir / "COMPLETE"
    tmp_path = run_dir / "COMPLETE.tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(marker_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    tmp_path.rename(marker_path)
