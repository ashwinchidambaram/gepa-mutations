"""Tests for the CLI entry point: src/iso_harness/experiment/__main__.py.

Covers:
- --help flag
- --dry-run with pilot config
- Missing --config flag
- Invalid (nonexistent) config path
- --smoke-test flag accepted alongside --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iso_harness.experiment.__main__ import main

# Absolute path to the repo root (tests/unit/ -> ../..)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PILOT_YAML = REPO_ROOT / "configs" / "pilot.yaml"


# ---------------------------------------------------------------------------
# test_help_flag
# ---------------------------------------------------------------------------


def test_help_flag(monkeypatch, capsys):
    """--help should print usage information and exit with code 0."""
    monkeypatch.setattr(sys, "argv", ["prog", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--config" in captured.out
    assert "--dry-run" in captured.out


# ---------------------------------------------------------------------------
# test_dry_run_with_pilot_config
# ---------------------------------------------------------------------------


def test_dry_run_with_pilot_config(monkeypatch, capsys):
    """--config configs/pilot.yaml --dry-run should load config, call dry_run, and exit 0."""
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--config", str(PILOT_YAML), "--dry-run"],
    )

    mock_orchestrator = MagicMock()
    mock_orchestrator.build_matrix.return_value = []
    mock_orchestrator.dry_run.return_value = None

    # load_config and Orchestrator are imported lazily inside main(), so patch
    # them at their source module locations rather than on __main__.
    with patch("iso_harness.experiment.config.load_config", return_value=MagicMock()) as mock_load_config, \
         patch("iso_harness.experiment.orchestrator.Orchestrator", return_value=mock_orchestrator):

        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 0
    mock_load_config.assert_called_once_with(str(PILOT_YAML))
    mock_orchestrator.build_matrix.assert_called_once()
    mock_orchestrator.dry_run.assert_called_once()


# ---------------------------------------------------------------------------
# test_dry_run_prints_matrix (integration-style, real config)
# ---------------------------------------------------------------------------


def test_dry_run_prints_matrix(monkeypatch, capsys):
    """--dry-run with the real pilot config prints expected matrix output."""
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--config", str(PILOT_YAML), "--dry-run"],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Dry run" in captured.out
    assert "ifbench" in captured.out


# ---------------------------------------------------------------------------
# test_missing_config_flag
# ---------------------------------------------------------------------------


def test_missing_config_flag(monkeypatch):
    """Omitting --config should cause argparse to exit with code 2."""
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# test_invalid_config_path
# ---------------------------------------------------------------------------


def test_invalid_config_path(monkeypatch):
    """--config pointing at a nonexistent file should exit with code 2."""
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--config", "nonexistent.yaml"],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# test_smoke_test_flag_accepted
# ---------------------------------------------------------------------------


def test_smoke_test_flag_accepted(monkeypatch, capsys):
    """--smoke-test alongside --dry-run should be accepted without error."""
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--config", str(PILOT_YAML), "--dry-run", "--smoke-test"],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
