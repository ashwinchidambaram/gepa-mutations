"""Tests for orchestrator: budget enforcement, run matrix, execution."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from iso_harness.experiment.orchestrator import (
    BudgetEnforcer,
    BudgetExhaustedError,
    Orchestrator,
    RunSpec,
)
from iso_harness.experiment.config import ISOExperimentConfig, load_config

# Absolute path to the repo root (tests/unit/ -> ../..)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PILOT_YAML = REPO_ROOT / "configs" / "pilot.yaml"
FULL_YAML = REPO_ROOT / "configs" / "full.yaml"


def _load_full_pinned(tmp_path: Path) -> ISOExperimentConfig:
    """Load full.yaml content but with pinned SHAs so it validates."""
    content = FULL_YAML.read_text()
    content = content.replace("<PIN_AT_IMPLEMENTATION>", "abc123def456")
    p = tmp_path / "full_pinned.yaml"
    p.write_text(content)
    return load_config(p)


# -- Budget Enforcer Tests ----------------------------------------------------


class TestBudgetEnforcer:
    def test_initial_state(self):
        b = BudgetEnforcer(100)
        assert b.remaining == 100
        assert b.consumed == 0
        assert not b.is_exhausted

    def test_record_and_track(self):
        b = BudgetEnforcer(100)
        b.record_rollouts(30)
        assert b.consumed == 30
        assert b.remaining == 70

    def test_exact_budget(self):
        b = BudgetEnforcer(50)
        b.record_rollouts(50)
        assert b.is_exhausted
        assert b.remaining == 0

    def test_over_budget(self):
        b = BudgetEnforcer(50)
        b.record_rollouts(60)
        assert b.is_exhausted
        assert b.remaining == 0

    def test_check_raises(self):
        b = BudgetEnforcer(10)
        b.record_rollouts(10)
        with pytest.raises(BudgetExhaustedError):
            b.check()

    def test_check_ok(self):
        b = BudgetEnforcer(10)
        b.record_rollouts(5)
        b.check()  # Should not raise

    def test_thread_safety(self):
        b = BudgetEnforcer(10000)
        errors = []

        def increment():
            try:
                for _ in range(250):
                    b.record_rollouts(1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert b.consumed == 1000

    def test_negative_max_raises(self):
        with pytest.raises(ValueError):
            BudgetEnforcer(-1)

    def test_zero_budget(self):
        b = BudgetEnforcer(0)
        assert b.is_exhausted
        with pytest.raises(BudgetExhaustedError):
            b.check()


# -- RunSpec Tests ------------------------------------------------------------


class TestRunSpec:
    def test_auto_uuid(self):
        s = RunSpec(optimizer="iso", benchmark="hotpotqa", seed=42, budget_rollouts=100)
        assert len(s.run_id) == 36  # UUID format

    def test_custom_run_id(self):
        s = RunSpec(
            run_id="custom", optimizer="iso", benchmark="hotpotqa",
            seed=42, budget_rollouts=100,
        )
        assert s.run_id == "custom"


# -- Orchestrator Tests -------------------------------------------------------


class TestBuildMatrix:
    def test_full_config(self, tmp_path: Path):
        config = _load_full_pinned(tmp_path)
        orch = Orchestrator(config, runs_dir=tmp_path / "runs")
        # full.yaml has optimizers: {} so matrix is empty (no optimizer entries)
        matrix = orch.build_matrix()
        assert isinstance(matrix, list)

    def test_pilot_smoke_test(self):
        config = load_config(PILOT_YAML)
        orch = Orchestrator(config, runs_dir=Path("/tmp/test_runs_build_matrix"))
        matrix = orch.build_matrix()
        assert len(matrix) == 1
        assert matrix[0].benchmark == "ifbench"
        assert matrix[0].budget_rollouts == 100


class TestExecute:
    def test_successful_run(self, tmp_path: Path):
        config = load_config(PILOT_YAML)
        orch = Orchestrator(config, runs_dir=tmp_path)

        def fake_run(spec, run_dir, budget):
            budget.record_rollouts(50)
            return {"final_score": 0.75}

        matrix = [RunSpec(optimizer="test", benchmark="ifbench", seed=0, budget_rollouts=100)]
        results = orch.execute(matrix, run_fn=fake_run)

        assert len(results) == 1
        assert results[0]["status"] == "completed"
        assert results[0]["rollouts_consumed"] == 50

    def test_failure_handling(self, tmp_path: Path):
        config = load_config(PILOT_YAML)
        orch = Orchestrator(config, runs_dir=tmp_path)

        call_count = 0

        def failing_run(spec, run_dir, budget):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("test failure")

        matrix = [
            RunSpec(optimizer="test", benchmark="b", seed=i, budget_rollouts=100)
            for i in range(5)
        ]
        results = orch.execute(matrix, run_fn=failing_run)

        # Should halt after 3 consecutive failures
        assert call_count == 3
        assert len(results) == 3
        assert all(r["status"] == "failed" for r in results)

    def test_no_run_fn(self, tmp_path: Path):
        config = load_config(PILOT_YAML)
        orch = Orchestrator(config, runs_dir=tmp_path)

        matrix = [RunSpec(optimizer="test", benchmark="b", seed=0, budget_rollouts=100)]
        results = orch.execute(matrix)

        assert results[0]["status"] == "skipped"

    def test_budget_exhausted(self, tmp_path: Path):
        config = load_config(PILOT_YAML)
        orch = Orchestrator(config, runs_dir=tmp_path)

        def greedy_run(spec, run_dir, budget):
            budget.record_rollouts(spec.budget_rollouts + 1)
            budget.check()

        matrix = [RunSpec(optimizer="test", benchmark="b", seed=0, budget_rollouts=100)]
        results = orch.execute(matrix, run_fn=greedy_run)

        assert results[0]["status"] == "budget_exhausted"

    def test_consecutive_failure_reset(self, tmp_path: Path):
        config = load_config(PILOT_YAML)
        orch = Orchestrator(config, runs_dir=tmp_path)

        call_count = 0

        def intermittent_run(spec, run_dir, budget):
            nonlocal call_count
            call_count += 1
            if spec.seed in (1, 2):  # Fail on seeds 1 and 2
                raise RuntimeError("fail")
            return {"score": 1.0}

        matrix = [
            RunSpec(optimizer="t", benchmark="b", seed=s, budget_rollouts=100)
            for s in [0, 1, 2, 3, 4]
        ]
        results = orch.execute(matrix, run_fn=intermittent_run)

        # All 5 should run (failures are not consecutive enough to halt)
        assert call_count == 5
        assert results[0]["status"] == "completed"
        assert results[1]["status"] == "failed"
        assert results[2]["status"] == "failed"
        assert results[3]["status"] == "completed"
        assert results[4]["status"] == "completed"

    def test_run_dir_created_with_config(self, tmp_path: Path):
        config = load_config(PILOT_YAML)
        orch = Orchestrator(config, runs_dir=tmp_path)

        matrix = [
            RunSpec(
                run_id="test-run-123", optimizer="t", benchmark="b",
                seed=0, budget_rollouts=100,
            )
        ]
        orch.execute(matrix)

        config_path = tmp_path / "test-run-123" / "config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data["benchmark"] == "b"
        assert data["seed"] == 0


class TestDryRun:
    def test_prints_matrix(self, tmp_path: Path, capsys):
        config = load_config(PILOT_YAML)
        orch = Orchestrator(config, runs_dir=tmp_path)

        matrix = [
            RunSpec(optimizer="iso", benchmark="hotpotqa", seed=0, budget_rollouts=6500),
            RunSpec(optimizer="iso", benchmark="ifbench", seed=0, budget_rollouts=3500),
        ]
        orch.dry_run(matrix)

        captured = capsys.readouterr()
        assert "2 runs planned" in captured.out
        assert "hotpotqa" in captured.out
        assert "ifbench" in captured.out
        assert "Total rollout budget: 10000" in captured.out
