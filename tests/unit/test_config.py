"""Tests for YAML config loading and Pydantic validation.

Covers:
- Loading valid pilot and full configs from configs/ directory
- Rejection of missing required fields, invalid values, unknown phase
- SHA pinning: warning for pilot, error for full
- Default values applied for optional sections
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from iso_harness.experiment.config import ISOExperimentConfig, load_config

# Absolute path to the repo root (tests/unit/ -> ../..)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PILOT_YAML = REPO_ROOT / "configs" / "pilot.yaml"
FULL_YAML = REPO_ROOT / "configs" / "full.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str) -> Path:
    """Write a YAML string to a temp file and return the path."""
    p = tmp_path / "test_config.yaml"
    p.write_text(dedent(content))
    return p



# ======================================================================
# Test: Load valid pilot config
# ======================================================================


class TestLoadPilotConfig:
    def test_phase_is_pilot(self):
        cfg = load_config(PILOT_YAML)
        assert cfg.phase == "pilot"

    def test_task_model_id(self):
        cfg = load_config(PILOT_YAML)
        assert cfg.models.task.model_id == "Qwen/Qwen3-8B"

    def test_smoke_test_present(self):
        cfg = load_config(PILOT_YAML)
        assert cfg.smoke_test is not None

    def test_run_matrix_absent(self):
        cfg = load_config(PILOT_YAML)
        assert cfg.run_matrix is None


# ======================================================================
# Test: Load valid full config
# ======================================================================


class TestLoadFullConfig:
    """Full config has unpinned SHAs so load_config will sys.exit(2).

    We test the structural properties by validating with pinned SHAs.
    """

    def _load_full_pinned(self, tmp_path: Path) -> ISOExperimentConfig:
        """Load full.yaml content but with pinned SHAs so it validates."""
        content = FULL_YAML.read_text()
        content = content.replace("<PIN_AT_IMPLEMENTATION>", "abc123def456")
        p = tmp_path / "full_pinned.yaml"
        p.write_text(content)
        return load_config(p)

    def test_phase_is_full(self, tmp_path):
        cfg = self._load_full_pinned(tmp_path)
        assert cfg.phase == "full"

    def test_run_matrix_present(self, tmp_path):
        cfg = self._load_full_pinned(tmp_path)
        assert cfg.run_matrix is not None

    def test_benchmarks_count(self, tmp_path):
        cfg = self._load_full_pinned(tmp_path)
        assert len(cfg.run_matrix.benchmarks) == 6

    def test_benchmark_budgets(self, tmp_path):
        cfg = self._load_full_pinned(tmp_path)
        bm = cfg.run_matrix.benchmarks
        assert bm["hotpotqa"].budget_rollouts == 6500
        assert bm["aime"].budget_rollouts == 7000
        assert bm["pupa"].budget_rollouts == 3500
        assert bm["ifbench"].budget_rollouts == 3500
        assert bm["hover"].budget_rollouts == 2000
        assert bm["livebench"].budget_rollouts == 1500


# ======================================================================
# Test: Reject missing required fields
# ======================================================================


class TestRejectMissingFields:
    def test_missing_phase(self, tmp_path):
        content = """\
        models:
          task:
            model_id: test/model
            max_model_len: 4096
          reflection:
            model_id: test/reflection
            max_model_len: 4096
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 2

    def test_missing_models(self, tmp_path):
        p = _write_yaml(tmp_path, "phase: pilot\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 2


# ======================================================================
# Test: Reject invalid values
# ======================================================================


class TestRejectInvalidValues:
    def test_negative_max_model_len(self, tmp_path):
        content = """\
        phase: pilot
        models:
          task:
            model_id: test/model
            max_model_len: -1
          reflection:
            model_id: test/reflection
            max_model_len: 4096
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 2

    def test_zero_max_model_len(self, tmp_path):
        content = """\
        phase: pilot
        models:
          task:
            model_id: test/model
            max_model_len: 0
          reflection:
            model_id: test/reflection
            max_model_len: 4096
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 2

    def test_temperature_too_high(self, tmp_path):
        content = """\
        phase: pilot
        models:
          task:
            model_id: test/model
            max_model_len: 4096
            temperature: 3.0
          reflection:
            model_id: test/reflection
            max_model_len: 4096
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 2


# ======================================================================
# Test: SHA warning for pilot (unpinned SHAs allowed)
# ======================================================================


class TestSHAWarningPilot:
    def test_pilot_loads_with_unpinned_sha(self):
        """Pilot config has unpinned SHAs — should load successfully (just warns)."""
        cfg = load_config(PILOT_YAML)
        assert cfg.models.task.hf_sha == "<PIN_AT_IMPLEMENTATION>"
        assert cfg.models.reflection.hf_sha == "<PIN_AT_IMPLEMENTATION>"


# ======================================================================
# Test: SHA error for full phase
# ======================================================================


class TestSHAErrorFull:
    def test_full_phase_rejects_unpinned_sha(self, tmp_path):
        """Full phase with unpinned SHAs must fail with sys.exit(2)."""
        content = """\
        phase: full
        models:
          task:
            model_id: test/model
            hf_sha: "<PIN_AT_IMPLEMENTATION>"
            max_model_len: 4096
          reflection:
            model_id: test/reflection
            hf_sha: "<PIN_AT_IMPLEMENTATION>"
            max_model_len: 4096
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 2

    def test_full_phase_rejects_unpinned_sha_from_file(self):
        """The actual full.yaml has unpinned SHAs and must fail."""
        with pytest.raises(SystemExit) as exc_info:
            load_config(FULL_YAML)
        assert exc_info.value.code == 2


# ======================================================================
# Test: Reject unknown phase
# ======================================================================


class TestRejectUnknownPhase:
    def test_invalid_phase_string(self, tmp_path):
        content = """\
        phase: invalid
        models:
          task:
            model_id: test/model
            max_model_len: 4096
          reflection:
            model_id: test/reflection
            max_model_len: 4096
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 2


# ======================================================================
# Test: Defaults applied for optional sections
# ======================================================================


class TestDefaultsApplied:
    def test_logging_defaults(self, tmp_path):
        content = """\
        phase: pilot
        models:
          task:
            model_id: test/model
            max_model_len: 4096
          reflection:
            model_id: test/reflection
            max_model_len: 4096
        """
        p = _write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.logging.mlflow_tracking_uri == "file:///workspace/mlflow"
        assert cfg.logging.artifacts_dir == "/workspace/iso-experiment/runs"

    def test_monitoring_defaults(self, tmp_path):
        content = """\
        phase: pilot
        models:
          task:
            model_id: test/model
            max_model_len: 4096
          reflection:
            model_id: test/reflection
            max_model_len: 4096
        """
        p = _write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.monitoring.gpu_poll_interval_sec == 30
        assert cfg.monitoring.kv_cache_poll_interval_sec == 60
        assert cfg.monitoring.disk_check_interval_min == 10
        assert cfg.monitoring.disk_min_free_gb == 20

    def test_smoke_test_default_none(self, tmp_path):
        content = """\
        phase: pilot
        models:
          task:
            model_id: test/model
            max_model_len: 4096
          reflection:
            model_id: test/reflection
            max_model_len: 4096
        """
        p = _write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.smoke_test is None

    def test_run_matrix_default_none(self, tmp_path):
        content = """\
        phase: pilot
        models:
          task:
            model_id: test/model
            max_model_len: 4096
          reflection:
            model_id: test/reflection
            max_model_len: 4096
        """
        p = _write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.run_matrix is None

    def test_model_defaults(self, tmp_path):
        content = """\
        phase: pilot
        models:
          task:
            model_id: test/model
            max_model_len: 4096
          reflection:
            model_id: test/reflection
            max_model_len: 4096
        """
        p = _write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.models.task.dtype == "bfloat16"
        assert cfg.models.task.quantization is None
        assert cfg.models.task.temperature == 1.0
        assert cfg.models.task.max_tokens == 8192


# ======================================================================
# Test: Edge cases
# ======================================================================


class TestEdgeCases:
    def test_nonexistent_file(self):
        with pytest.raises(SystemExit) as exc_info:
            load_config("/nonexistent/path/config.yaml")
        assert exc_info.value.code == 2

    def test_non_mapping_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- just\n- a\n- list\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 2

    def test_empty_yaml(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 2
