"""Local filesystem result persistence.

Directory structure: runs/<benchmark>/<method>/<seed>/
  - config.json   — full experiment config snapshot
  - result.json   — scores, best prompt, rollout count
  - metrics.json  — diagnostic metrics from callback
  - state.pkl     — GEPA state binary for post-hoc re-analysis
  - environment.json — software/hardware environment metadata (once per model_tag)
"""

from __future__ import annotations

import json
import pickle  # Required for GEPA state binary serialization (GEPAState uses pickle internally)
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import socket


RUNS_DIR = Path("runs")


def _run_dir(benchmark: str, method: str = "gepa", seed: int = 0, model_tag: str = "") -> Path:
    if model_tag:
        return RUNS_DIR / model_tag / benchmark / method / str(seed)
    return RUNS_DIR / benchmark / method / str(seed)


def save_environment(model_tag: str) -> Path:
    """Save software/hardware environment metadata once per model tag.

    Creates runs/{model_tag}/environment.json with Python version, timestamp,
    package versions, GPU info, and hostname. If the file already exists, skips
    creation and returns the existing path.

    Args:
        model_tag: Model identifier string (e.g., "qwen3-8b", "gemma3-1b")

    Returns:
        Path to the environment.json file
    """
    env_path = RUNS_DIR / model_tag / "environment.json"

    if env_path.exists():
        return env_path  # already saved

    env_data = {
        "python_version": sys.version,
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
    }

    # Try to get package versions
    for pkg, key in [("torch", "torch_version"), ("vllm", "vllm_version"), ("gepa", "gepa_version")]:
        try:
            mod = __import__(pkg)
            env_data[key] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env_data[key] = "not installed"

    # Try to get GPU info via nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            env_data["gpu_info"] = result.stdout.strip()
    except Exception:
        env_data["gpu_info"] = "unavailable"

    env_path.parent.mkdir(parents=True, exist_ok=True)
    with open(env_path, "w") as f:
        json.dump(env_data, f, indent=2)

    return env_path


def save_result(
    benchmark: str,
    seed: int,
    result_data: dict[str, Any],
    config_data: dict[str, Any],
    metrics_data: dict[str, Any] | None = None,
    state_obj: Any = None,
    method: str = "gepa",
    model_tag: str = "",
    test_outputs: list[dict] | None = None,
) -> Path:
    """Save experiment results to local filesystem.

    Returns the run directory path.
    """
    # Save environment metadata once per model_tag
    if model_tag:
        save_environment(model_tag)

    run_path = _run_dir(benchmark, method, seed, model_tag=model_tag)
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

    # Per-example model outputs for qualitative error analysis
    if test_outputs is not None:
        with open(run_path / "test_outputs.json", "w") as f:
            json.dump(test_outputs, f, indent=2, default=str)

    return run_path


def load_result(benchmark: str, seed: int, method: str = "gepa", model_tag: str = "") -> dict[str, Any]:
    """Load experiment result from local filesystem."""
    run_path = _run_dir(benchmark, method, seed, model_tag=model_tag)
    result_file = run_path / "result.json"

    if not result_file.exists():
        raise FileNotFoundError(f"No result found at {result_file}")

    with open(result_file) as f:
        return json.load(f)


def load_metrics(benchmark: str, seed: int, method: str = "gepa", model_tag: str = "") -> dict[str, Any] | None:
    """Load metrics data if available."""
    run_path = _run_dir(benchmark, method, seed, model_tag=model_tag)
    metrics_file = run_path / "metrics.json"

    if not metrics_file.exists():
        return None

    with open(metrics_file) as f:
        return json.load(f)


def load_config(benchmark: str, seed: int, method: str = "gepa", model_tag: str = "") -> dict[str, Any]:
    """Load experiment config snapshot."""
    run_path = _run_dir(benchmark, method, seed, model_tag=model_tag)
    config_file = run_path / "config.json"

    if not config_file.exists():
        raise FileNotFoundError(f"No config found at {config_file}")

    with open(config_file) as f:
        return json.load(f)


def list_runs(benchmark: str | None = None, method: str = "gepa", model_tag: str = "") -> list[dict[str, Any]]:
    """List all completed runs, optionally filtered by benchmark.

    Returns list of dicts with keys: benchmark, method, seed, path.
    """
    runs = []
    base = RUNS_DIR / model_tag if model_tag else RUNS_DIR

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
