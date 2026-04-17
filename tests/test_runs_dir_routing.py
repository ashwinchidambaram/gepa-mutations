"""Regression tests for the RUNS_DIR env var fix in storage/local.py.

Bug: Three layers hardcoded "runs/" path, ignoring the RUNS_DIR env var.
     Result: orchestrator couldn't find completed experiments in custom
     directories, causing unnecessary re-runs.
Fix: Committed in 57e4a7174 and b8c624930.

CRITICAL: RUNS_DIR is set at import time via
    RUNS_DIR = Path(os.environ.get("RUNS_DIR") or "runs")
so monkeypatch.setenv is too late.  All tests patch the module attribute
directly: monkeypatch.setattr("gepa_mutations.storage.local.RUNS_DIR", ...)
"""

from __future__ import annotations

import json

import pytest

import gepa_mutations.storage.local as local_mod
from gepa_mutations.storage.local import (
    list_runs,
    load_result,
    save_result,
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_result_uses_custom_runs_dir(monkeypatch, tmp_path):
    """save_result must write into the patched RUNS_DIR, not the default "runs/"."""
    monkeypatch.setattr(local_mod, "RUNS_DIR", tmp_path)

    save_result(
        benchmark="hotpotqa",
        seed=42,
        result_data={"test_score": 0.75},
        config_data={"model": "test"},
    )

    result_file = tmp_path / "hotpotqa" / "gepa" / "42" / "result.json"
    assert result_file.exists(), f"result.json not found at {result_file}"

    data = json.loads(result_file.read_text())
    assert data["test_score"] == 0.75


def test_load_result_uses_custom_runs_dir(monkeypatch, tmp_path):
    """load_result must read from the patched RUNS_DIR (round-trip test)."""
    monkeypatch.setattr(local_mod, "RUNS_DIR", tmp_path)

    original = {"test_score": 0.9, "best_prompt": "Think step by step."}
    save_result(
        benchmark="hotpotqa",
        seed=7,
        result_data=original,
        config_data={"model": "test"},
    )

    loaded = load_result(benchmark="hotpotqa", seed=7)
    assert loaded == original


def test_list_runs_uses_custom_runs_dir(monkeypatch, tmp_path):
    """list_runs must scan the patched RUNS_DIR and return the saved run."""
    monkeypatch.setattr(local_mod, "RUNS_DIR", tmp_path)

    save_result(
        benchmark="hotpotqa",
        seed=0,
        result_data={"test_score": 0.5},
        config_data={"model": "test"},
    )

    runs = list_runs(benchmark="hotpotqa")
    assert len(runs) == 1, f"Expected 1 run, got {len(runs)}: {runs}"

    run = runs[0]
    assert run["benchmark"] == "hotpotqa"
    assert run["method"] == "gepa"
    assert run["seed"] == 0


def test_run_dir_with_model_tag(monkeypatch, tmp_path):
    """When model_tag is provided, path must be RUNS_DIR/<model_tag>/<benchmark>/...."""
    monkeypatch.setattr(local_mod, "RUNS_DIR", tmp_path)

    save_result(
        benchmark="hotpotqa",
        seed=42,
        result_data={"test_score": 0.8},
        config_data={"model": "qwen3-8b"},
        model_tag="qwen3-8b",
    )

    result_file = tmp_path / "qwen3-8b" / "hotpotqa" / "gepa" / "42" / "result.json"
    assert result_file.exists(), (
        f"result.json not found at expected model-tag path {result_file}"
    )

    data = json.loads(result_file.read_text())
    assert data["test_score"] == 0.8


def test_load_nonexistent_raises(monkeypatch, tmp_path):
    """load_result must raise FileNotFoundError for a run that was never saved."""
    monkeypatch.setattr(local_mod, "RUNS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        load_result(benchmark="hotpotqa", seed=99)
