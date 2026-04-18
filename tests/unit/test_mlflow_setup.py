"""Tests for MLflow integration: setup, experiments, runs, metrics."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow

from iso_harness.experiment.mlflow_setup import (
    end_run,
    get_or_create_experiment,
    log_round_metrics,
    log_run_summary,
    register_artifacts,
    setup_mlflow,
    start_run,
)


class TestSetupMlflow:
    def test_sets_tracking_uri(self, tmp_path: Path):
        uri = f"file://{tmp_path}/mlflow"
        result = setup_mlflow(uri)
        assert result == uri
        assert mlflow.get_tracking_uri() == uri


class TestExperiments:
    def test_create_experiment(self, tmp_path: Path):
        setup_mlflow(f"file://{tmp_path}/mlflow")
        exp_id = get_or_create_experiment("test-experiment")
        assert exp_id is not None

    def test_get_existing_experiment(self, tmp_path: Path):
        setup_mlflow(f"file://{tmp_path}/mlflow")
        id1 = get_or_create_experiment("test-experiment-2")
        id2 = get_or_create_experiment("test-experiment-2")
        assert id1 == id2


class TestRuns:
    def test_start_and_end_run(self, tmp_path: Path):
        setup_mlflow(f"file://{tmp_path}/mlflow")
        exp_id = get_or_create_experiment("test-runs")
        active_run = start_run(exp_id, "test-run-1", params={"optimizer": "iso"})
        assert active_run is not None
        run_id = mlflow.active_run().info.run_id
        assert run_id is not None
        end_run()
        assert mlflow.active_run() is None

    def test_params_logged(self, tmp_path: Path):
        setup_mlflow(f"file://{tmp_path}/mlflow")
        exp_id = get_or_create_experiment("test-params")
        start_run(exp_id, "test-run-params", params={"benchmark": "hotpotqa", "seed": "42"})
        run = mlflow.active_run()
        # Params are logged -- verify via the run data
        run_data = mlflow.get_run(run.info.run_id)
        assert run_data.data.params["benchmark"] == "hotpotqa"
        end_run()

    def test_long_param_truncated(self, tmp_path: Path):
        setup_mlflow(f"file://{tmp_path}/mlflow")
        exp_id = get_or_create_experiment("test-truncate")
        long_val = "x" * 1000
        start_run(exp_id, "test-truncate", params={"long": long_val})
        run = mlflow.active_run()
        run_data = mlflow.get_run(run.info.run_id)
        assert len(run_data.data.params["long"]) == 500
        end_run()


class TestMetrics:
    def test_log_round_metrics(self, tmp_path: Path):
        setup_mlflow(f"file://{tmp_path}/mlflow")
        exp_id = get_or_create_experiment("test-metrics")
        start_run(exp_id, "test-round-metrics")
        log_round_metrics(0, {"mean_score": 0.5, "rollouts": 100})
        log_round_metrics(1, {"mean_score": 0.65, "rollouts": 200})
        run = mlflow.active_run()
        run_data = mlflow.get_run(run.info.run_id)
        assert "mean_score" in run_data.data.metrics
        end_run()

    def test_log_run_summary(self, tmp_path: Path):
        setup_mlflow(f"file://{tmp_path}/mlflow")
        exp_id = get_or_create_experiment("test-summary")
        start_run(exp_id, "test-summary")
        log_run_summary({
            "final_score_val": 0.75,
            "final_score_test": 0.72,
            "duration_seconds": 3600,
            "rollouts_consumed_total": 6500,
            "tokens_consumed_total": 1200000,
            "cost_estimate_usd": 1.99,
            "non_numeric_field": "should be ignored",
        })
        run = mlflow.active_run()
        run_data = mlflow.get_run(run.info.run_id)
        assert run_data.data.metrics["final_score_test"] == 0.72
        assert "non_numeric_field" not in run_data.data.metrics
        end_run()


class TestArtifacts:
    def test_register_artifacts(self, tmp_path: Path):
        setup_mlflow(f"file://{tmp_path}/mlflow")
        exp_id = get_or_create_experiment("test-artifacts")
        start_run(exp_id, "test-artifacts")

        # Create some fake artifact files
        run_dir = tmp_path / "run_artifacts"
        run_dir.mkdir()
        (run_dir / "rollouts.jsonl").write_text('{"score": 0.5}\n')
        (run_dir / "summary.json").write_text('{"optimizer": "iso"}')
        (run_dir / "COMPLETE").write_text('{}')

        register_artifacts(run_dir)

        run = mlflow.active_run()
        run_data = mlflow.get_run(run.info.run_id)
        artifact_uri = run_data.info.artifact_uri
        assert artifact_uri is not None
        end_run()

    def test_nonexistent_dir_no_crash(self, tmp_path: Path):
        setup_mlflow(f"file://{tmp_path}/mlflow")
        exp_id = get_or_create_experiment("test-nodir")
        start_run(exp_id, "test-nodir")
        register_artifacts(tmp_path / "nonexistent")  # Should not crash
        end_run()
