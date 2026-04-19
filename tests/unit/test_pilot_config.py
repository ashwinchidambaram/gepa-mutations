"""Tests for pilot phase YAML config loading and Pydantic validation.

Covers:
- Load configs/pilot_phase_a.yaml successfully (Track 1)
- Load configs/pilot_phase_b.yaml successfully (Track 2)
- Verify key fields (variants, benchmarks, meta_optimizers)
- Reject missing required fields and invalid track values
- load_pilot_config auto-detects correct config type
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from iso_harness.optimizer.pilot_config import (
    PilotPhaseAConfig,
    PilotPhaseBConfig,
    load_pilot_config,
)

# Absolute path to the repo root (tests/unit/ -> ../..)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE_A_YAML = REPO_ROOT / "configs" / "pilot_phase_a.yaml"
PHASE_B_YAML = REPO_ROOT / "configs" / "pilot_phase_b.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str) -> Path:
    """Write a YAML string to a temp file and return the path."""
    p = tmp_path / "test_pilot.yaml"
    p.write_text(dedent(content))
    return p


# ======================================================================
# Test: Load pilot_phase_a.yaml (Track 1)
# ======================================================================


class TestLoadPhaseA:
    def test_returns_phase_a_config(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert isinstance(cfg, PilotPhaseAConfig)

    def test_phase_is_pilot(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.phase == "pilot"

    def test_track_is_1(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.track == 1

    def test_task_model_name(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.models.task.name == "Qwen/Qwen3-8B"

    def test_task_model_port(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.models.task.port == 8000

    def test_task_model_max_tokens(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.models.task.max_tokens == 8192

    def test_reflection_model_name(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.models.reflection.name == "Qwen/Qwen3-32B-AWQ"

    def test_reflection_model_max_tokens(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.models.reflection.max_tokens == 4096

    def test_meta_model_absent(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.models.meta is None

    def test_variants_list(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.variants == [
            "iso_sprint",
            "iso_grove",
            "iso_tide",
            "iso_lens",
            "iso_storm",
        ]

    def test_baselines_list(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.baselines == ["gepa", "mipro"]

    def test_benchmarks_keys(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert "ifbench" in cfg.benchmarks

    def test_ifbench_budget(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.benchmarks["ifbench"].budget == 3500

    def test_ifbench_seeds(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.benchmarks["ifbench"].seeds == [0, 1, 2]

    def test_ifbench_subset_size_is_none(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.benchmarks["ifbench"].subset_size is None

    def test_smoke_test_present(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.smoke_test is not None

    def test_smoke_test_budget(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.smoke_test.budget == 100

    def test_smoke_test_subset_size(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert cfg.smoke_test.subset_size == 20

    def test_smoke_test_overrides_keys(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        overrides = cfg.smoke_test.overrides
        assert "n_discovery_examples" in overrides
        assert "max_rounds" in overrides
        assert "plateau_rounds_threshold" in overrides

    def test_smoke_test_overrides_values(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        overrides = cfg.smoke_test.overrides
        assert overrides["n_discovery_examples"] == 5
        assert overrides["max_rounds"] == 3
        assert overrides["plateau_rounds_threshold"] == 99


# ======================================================================
# Test: Load pilot_phase_b.yaml (Track 2)
# ======================================================================


class TestLoadPhaseB:
    def test_returns_phase_b_config(self):
        cfg = load_pilot_config(PHASE_B_YAML)
        assert isinstance(cfg, PilotPhaseBConfig)

    def test_phase_is_pilot(self):
        cfg = load_pilot_config(PHASE_B_YAML)
        assert cfg.phase == "pilot"

    def test_track_is_2(self):
        cfg = load_pilot_config(PHASE_B_YAML)
        assert cfg.track == 2

    def test_task_model_name(self):
        cfg = load_pilot_config(PHASE_B_YAML)
        assert cfg.models.task.name == "Qwen/Qwen3-8B"

    def test_meta_model_present(self):
        cfg = load_pilot_config(PHASE_B_YAML)
        assert cfg.models.meta is not None
        assert cfg.models.meta.name == "Qwen/Qwen3-32B-AWQ"

    def test_inner_variant(self):
        cfg = load_pilot_config(PHASE_B_YAML)
        assert cfg.inner_variant == "iso_tide"

    def test_meta_optimizers_keys(self):
        cfg = load_pilot_config(PHASE_B_YAML)
        assert set(cfg.meta_optimizers.keys()) == {"scout", "cartographer", "atlas"}

    def test_scout_config(self):
        cfg = load_pilot_config(PHASE_B_YAML)
        scout = cfg.meta_optimizers["scout"]
        assert scout.n_episodes == 50
        assert scout.surrogate_size == 20
        assert scout.playbook_update_interval is None

    def test_cartographer_config(self):
        cfg = load_pilot_config(PHASE_B_YAML)
        cart = cfg.meta_optimizers["cartographer"]
        assert cart.n_episodes == 50
        assert cart.surrogate_size == 20
        assert cart.playbook_update_interval == 10

    def test_atlas_config(self):
        cfg = load_pilot_config(PHASE_B_YAML)
        atlas = cfg.meta_optimizers["atlas"]
        assert atlas.n_episodes == 80
        assert atlas.surrogate_size == 20

    def test_benchmark(self):
        cfg = load_pilot_config(PHASE_B_YAML)
        assert cfg.benchmark == "ifbench"


# ======================================================================
# Test: Validation failures — missing required fields
# ======================================================================


class TestValidationFailures:
    def test_missing_track(self, tmp_path):
        content = """\
        phase: pilot
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
        variants:
          - iso_sprint
        benchmarks:
          ifbench:
            budget: 100
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_pilot_config(p)
        assert exc_info.value.code == 2

    def test_invalid_track(self, tmp_path):
        content = """\
        phase: pilot
        track: 99
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_pilot_config(p)
        assert exc_info.value.code == 2

    def test_track_1_missing_models(self, tmp_path):
        content = """\
        phase: pilot
        track: 1
        variants:
          - iso_sprint
        benchmarks:
          ifbench:
            budget: 100
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_pilot_config(p)
        assert exc_info.value.code == 2

    def test_track_1_missing_variants(self, tmp_path):
        content = """\
        phase: pilot
        track: 1
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
        benchmarks:
          ifbench:
            budget: 100
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_pilot_config(p)
        assert exc_info.value.code == 2

    def test_track_1_missing_benchmarks(self, tmp_path):
        content = """\
        phase: pilot
        track: 1
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
        variants:
          - iso_sprint
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_pilot_config(p)
        assert exc_info.value.code == 2

    def test_track_2_missing_inner_variant(self, tmp_path):
        content = """\
        phase: pilot
        track: 2
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
        meta_optimizers:
          scout:
            n_episodes: 50
            surrogate_size: 20
        benchmark: ifbench
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_pilot_config(p)
        assert exc_info.value.code == 2

    def test_track_2_missing_meta_optimizers(self, tmp_path):
        content = """\
        phase: pilot
        track: 2
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
        inner_variant: iso_tide
        benchmark: ifbench
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_pilot_config(p)
        assert exc_info.value.code == 2

    def test_track_2_missing_benchmark(self, tmp_path):
        content = """\
        phase: pilot
        track: 2
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
        inner_variant: iso_tide
        meta_optimizers:
          scout:
            n_episodes: 50
            surrogate_size: 20
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_pilot_config(p)
        assert exc_info.value.code == 2

    def test_nonexistent_file(self):
        with pytest.raises(SystemExit) as exc_info:
            load_pilot_config("/nonexistent/path/pilot_phase_a.yaml")
        assert exc_info.value.code == 2

    def test_non_mapping_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- just\n- a\n- list\n")
        with pytest.raises(SystemExit) as exc_info:
            load_pilot_config(p)
        assert exc_info.value.code == 2

    def test_empty_yaml(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        with pytest.raises(SystemExit) as exc_info:
            load_pilot_config(p)
        assert exc_info.value.code == 2

    def test_model_missing_name(self, tmp_path):
        content = """\
        phase: pilot
        track: 1
        models:
          task:
            port: 8000
          reflection:
            name: test/reflection
        variants:
          - iso_sprint
        benchmarks:
          ifbench:
            budget: 100
        """
        p = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            load_pilot_config(p)
        assert exc_info.value.code == 2


# ======================================================================
# Test: Auto-detection of config type
# ======================================================================


class TestAutoDetect:
    def test_phase_a_auto_detected(self):
        cfg = load_pilot_config(PHASE_A_YAML)
        assert type(cfg) is PilotPhaseAConfig

    def test_phase_b_auto_detected(self):
        cfg = load_pilot_config(PHASE_B_YAML)
        assert type(cfg) is PilotPhaseBConfig

    def test_track_1_inline_auto_detected(self, tmp_path):
        content = """\
        phase: pilot
        track: 1
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
        variants:
          - iso_sprint
        benchmarks:
          ifbench:
            budget: 200
        """
        p = _write_yaml(tmp_path, content)
        cfg = load_pilot_config(p)
        assert isinstance(cfg, PilotPhaseAConfig)

    def test_track_2_inline_auto_detected(self, tmp_path):
        content = """\
        phase: pilot
        track: 2
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
          meta:
            name: test/meta
        inner_variant: iso_tide
        meta_optimizers:
          scout:
            n_episodes: 10
            surrogate_size: 5
        benchmark: ifbench
        """
        p = _write_yaml(tmp_path, content)
        cfg = load_pilot_config(p)
        assert isinstance(cfg, PilotPhaseBConfig)


# ======================================================================
# Test: Default values
# ======================================================================


class TestDefaultValues:
    def test_model_port_default(self, tmp_path):
        content = """\
        phase: pilot
        track: 1
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
        variants:
          - iso_sprint
        benchmarks:
          ifbench:
            budget: 100
        """
        p = _write_yaml(tmp_path, content)
        cfg = load_pilot_config(p)
        assert cfg.models.task.port == 8000
        assert cfg.models.reflection.port == 8000

    def test_model_max_tokens_default(self, tmp_path):
        content = """\
        phase: pilot
        track: 1
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
        variants:
          - iso_sprint
        benchmarks:
          ifbench:
            budget: 100
        """
        p = _write_yaml(tmp_path, content)
        cfg = load_pilot_config(p)
        assert cfg.models.task.max_tokens == 8192
        assert cfg.models.reflection.max_tokens == 8192

    def test_smoke_test_default_none(self, tmp_path):
        content = """\
        phase: pilot
        track: 1
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
        variants:
          - iso_sprint
        benchmarks:
          ifbench:
            budget: 100
        """
        p = _write_yaml(tmp_path, content)
        cfg = load_pilot_config(p)
        assert cfg.smoke_test is None

    def test_baselines_default_empty(self, tmp_path):
        content = """\
        phase: pilot
        track: 1
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
        variants:
          - iso_sprint
        benchmarks:
          ifbench:
            budget: 100
        """
        p = _write_yaml(tmp_path, content)
        cfg = load_pilot_config(p)
        assert cfg.baselines == []

    def test_benchmark_seeds_default(self, tmp_path):
        content = """\
        phase: pilot
        track: 1
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
        variants:
          - iso_sprint
        benchmarks:
          ifbench:
            budget: 100
        """
        p = _write_yaml(tmp_path, content)
        cfg = load_pilot_config(p)
        assert cfg.benchmarks["ifbench"].seeds == [0, 1, 2]

    def test_meta_optimizer_playbook_update_default_none(self, tmp_path):
        content = """\
        phase: pilot
        track: 2
        models:
          task:
            name: test/model
          reflection:
            name: test/reflection
        inner_variant: iso_tide
        meta_optimizers:
          scout:
            n_episodes: 10
            surrogate_size: 5
        benchmark: ifbench
        """
        p = _write_yaml(tmp_path, content)
        cfg = load_pilot_config(p)
        assert cfg.meta_optimizers["scout"].playbook_update_interval is None
