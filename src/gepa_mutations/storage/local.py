"""Local filesystem result persistence.

Directory structure: runs/<benchmark>/<method>/<seed>/
  - config.json   — full experiment config snapshot
  - result.json   — scores, best prompt, rollout count
  - metrics.json  — diagnostic metrics from callback
  - state.pkl     — GEPA state binary for post-hoc re-analysis
"""

from __future__ import annotations

import json
import pickle  # Required for GEPA state binary serialization (GEPAState uses pickle internally)
from pathlib import Path
from typing import Any


RUNS_DIR = Path("runs")


def _run_dir(benchmark: str, method: str = "gepa", seed: int = 0) -> Path:
    return RUNS_DIR / benchmark / method / str(seed)


def save_result(
    benchmark: str,
    seed: int,
    result_data: dict[str, Any],
    config_data: dict[str, Any],
    metrics_data: dict[str, Any] | None = None,
    state_obj: Any = None,
    method: str = "gepa",
) -> Path:
    """Save experiment results to local filesystem.

    Returns the run directory path.
    """
    run_path = _run_dir(benchmark, method, seed)
    run_path.mkdir(parents=True, exist_ok=True)

    # Config snapshot
    with open(run_path / "config.json", "w") as f:
        json.dump(config_data, f, indent=2, default=str)

    # Result data
    with open(run_path / "result.json", "w") as f:
        json.dump(result_data, f, indent=2, default=str)

    # Metrics
    if metrics_data is not None:
        with open(run_path / "metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2, default=str)

    # GEPA state binary (pickle is required here as GEPAState.save() uses pickle internally)
    if state_obj is not None:
        with open(run_path / "state.pkl", "wb") as f:
            pickle.dump(state_obj, f)

    return run_path


def load_result(benchmark: str, seed: int, method: str = "gepa") -> dict[str, Any]:
    """Load experiment result from local filesystem."""
    run_path = _run_dir(benchmark, method, seed)
    result_file = run_path / "result.json"

    if not result_file.exists():
        raise FileNotFoundError(f"No result found at {result_file}")

    with open(result_file) as f:
        return json.load(f)


def load_metrics(benchmark: str, seed: int, method: str = "gepa") -> dict[str, Any] | None:
    """Load metrics data if available."""
    run_path = _run_dir(benchmark, method, seed)
    metrics_file = run_path / "metrics.json"

    if not metrics_file.exists():
        return None

    with open(metrics_file) as f:
        return json.load(f)


def load_config(benchmark: str, seed: int, method: str = "gepa") -> dict[str, Any]:
    """Load experiment config snapshot."""
    run_path = _run_dir(benchmark, method, seed)
    config_file = run_path / "config.json"

    if not config_file.exists():
        raise FileNotFoundError(f"No config found at {config_file}")

    with open(config_file) as f:
        return json.load(f)


def list_runs(benchmark: str | None = None, method: str = "gepa") -> list[dict[str, Any]]:
    """List all completed runs, optionally filtered by benchmark.

    Returns list of dicts with keys: benchmark, method, seed, path.
    """
    runs = []
    base = RUNS_DIR

    if not base.exists():
        return runs

    benchmarks = [benchmark] if benchmark else [d.name for d in base.iterdir() if d.is_dir()]

    for bm in benchmarks:
        method_dir = base / bm / method
        if not method_dir.exists():
            continue
        for seed_dir in sorted(method_dir.iterdir()):
            if seed_dir.is_dir() and (seed_dir / "result.json").exists():
                runs.append({
                    "benchmark": bm,
                    "method": method,
                    "seed": int(seed_dir.name),
                    "path": str(seed_dir),
                })

    return runs
