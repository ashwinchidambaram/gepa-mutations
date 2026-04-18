"""MLflow integration for ISO experiment tracking.

Self-hosted SQLite backend — no external SaaS dependency. All tracking
data lives on the same persistent storage as experiment artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import mlflow to avoid import-time failures when mlflow isn't installed
_mlflow = None


def _get_mlflow():
    """Lazy-import mlflow."""
    global _mlflow
    if _mlflow is None:
        import mlflow
        _mlflow = mlflow
    return _mlflow


def setup_mlflow(tracking_uri: str = "file:///tmp/mlflow") -> str:
    """Configure MLflow tracking URI and return it.

    Args:
        tracking_uri: MLflow tracking URI. Defaults to local temp.
            On RunPod: "file:///workspace/mlflow"
            Loaded from config.logging.mlflow_tracking_uri in practice.

    Returns:
        The tracking URI that was set.
    """
    mlflow = _get_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI set to: %s", tracking_uri)
    return tracking_uri


def get_or_create_experiment(experiment_name: str) -> str:
    """Get or create an MLflow experiment by name.

    Returns:
        The experiment ID (string).
    """
    mlflow = _get_mlflow()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    experiment_id = mlflow.create_experiment(experiment_name)
    logger.info("Created MLflow experiment '%s' (id=%s)", experiment_name, experiment_id)
    return experiment_id


def start_run(
    experiment_id: str,
    run_name: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """Start an MLflow run and log initial parameters.

    Args:
        experiment_id: MLflow experiment ID from get_or_create_experiment().
        run_name: Human-readable run name (e.g., "ISO-Tide/hotpotqa/seed=0").
        params: Dict of parameters to log (config values, git sha, etc.).

    Returns:
        mlflow.ActiveRun context manager. Use with `with start_run(...) as run:`.
    """
    mlflow = _get_mlflow()
    active_run = mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
    )

    if params:
        # MLflow params must be strings; truncate long values
        safe_params = {}
        for k, v in params.items():
            sv = str(v)
            if len(sv) > 500:
                sv = sv[:497] + "..."
            safe_params[k] = sv
        mlflow.log_params(safe_params)

    return active_run


def log_round_metrics(round_num: int, metrics: dict[str, float]) -> None:
    """Log scalar metrics for a specific round.

    Args:
        round_num: The optimizer round number (used as MLflow step).
        metrics: Dict of metric name -> value (e.g., {"mean_score": 0.75}).
    """
    mlflow = _get_mlflow()
    mlflow.log_metrics(metrics, step=round_num)


def log_run_summary(summary_dict: dict[str, Any]) -> None:
    """Log final run metrics from a RunSummary dict.

    Extracts scalar fields and logs them as MLflow metrics.
    """
    mlflow = _get_mlflow()

    # Log scalar metrics
    metric_keys = [
        "final_score_val", "final_score_test", "duration_seconds",
        "rollouts_consumed_total", "tokens_consumed_total", "cost_estimate_usd",
    ]
    metrics = {}
    for key in metric_keys:
        if key in summary_dict and isinstance(summary_dict[key], (int, float)):
            metrics[key] = float(summary_dict[key])
    if metrics:
        mlflow.log_metrics(metrics)


def register_artifacts(run_dir: Path) -> None:
    """Register all JSONL and JSON files in a run directory as MLflow artifacts.

    Args:
        run_dir: Path to the run directory (e.g., runs/{run_id}/).
    """
    mlflow = _get_mlflow()
    run_dir = Path(run_dir)
    if not run_dir.exists():
        logger.warning("Run directory does not exist: %s", run_dir)
        return

    for pattern in ("*.jsonl", "*.json", "COMPLETE"):
        for f in run_dir.glob(pattern):
            mlflow.log_artifact(str(f))
            logger.debug("Registered artifact: %s", f.name)


def end_run() -> None:
    """End the current active MLflow run."""
    mlflow = _get_mlflow()
    mlflow.end_run()


def enable_dspy_autolog() -> None:
    """Enable DSPy autolog for GEPA/MIPROv2 baseline runs.

    Only call this for runs that use DSPy's native LM system.
    ISO runs use a custom LM and should NOT call this.
    """
    mlflow = _get_mlflow()
    try:
        mlflow.dspy.autolog(
            log_traces=True,
            log_traces_from_eval=True,
        )
        logger.info("DSPy autolog enabled")
    except Exception as e:
        logger.warning("Failed to enable DSPy autolog: %s", e)
