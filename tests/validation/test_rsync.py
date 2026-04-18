"""V10: Rsync validation — sync script produces expected output.

This test only runs on the researcher's local machine with pod SSH configured.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.mark.local_only
class TestRsyncValidation:
    """V10: Validate sync_from_pod.sh works correctly.

    Requires: SSH access to a running RunPod pod (configured in .env).
    """

    def test_sync_script_exists(self):
        """Sync script exists and is executable."""
        script = Path("scripts/iso_sync_from_pod.sh")
        assert script.exists(), "iso_sync_from_pod.sh not found"
        assert script.stat().st_mode & 0o111, "iso_sync_from_pod.sh not executable"

    def test_sync_script_syntax(self):
        """Sync script passes bash syntax check."""
        result = subprocess.run(
            ["bash", "-n", "scripts/iso_sync_from_pod.sh"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"

    def test_sync_dry_run(self):
        """Sync script dry-run mode (requires pod SSH config)."""
        # Only attempt if POD_SSH_TARGET is configured
        import os
        if not os.environ.get("POD_SSH_TARGET"):
            pytest.skip("POD_SSH_TARGET not configured — skip rsync test")

        result = subprocess.run(
            ["bash", "scripts/iso_sync_from_pod.sh", "--dry-run"],
            capture_output=True, text=True, timeout=60,
        )
        # Dry run should complete without error (even if nothing to sync)
        assert result.returncode == 0, f"Dry run failed: {result.stderr}"
