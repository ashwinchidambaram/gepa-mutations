"""Fixtures and markers for ISO validation tests.

Markers:
- @pytest.mark.live_server — requires running vLLM server(s)
- @pytest.mark.slow — takes >30s
- @pytest.mark.local_only — only runs on researcher's local machine
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest


def _server_available(port: int) -> bool:
    """Check if a server is responding on the given port."""
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
        return True
    except Exception:
        return False


@pytest.fixture
def live_vllm_task():
    """Skip test if vLLM task server (port 8000) is not running."""
    if not _server_available(8000):
        pytest.skip("vLLM task server not running on port 8000")


@pytest.fixture
def live_vllm_reflection():
    """Skip test if vLLM reflection server (port 8001) is not running."""
    if not _server_available(8001):
        pytest.skip("vLLM reflection server not running on port 8001")


@pytest.fixture
def live_mlflow():
    """Skip test if MLflow UI is not running."""
    if not _server_available(5000):
        pytest.skip("MLflow UI not running on port 5000")


@pytest.fixture
def sample_config():
    """Load pilot config for testing."""
    from iso_harness.experiment.config import load_config
    return load_config(Path("configs/pilot.yaml"))


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """Create a temporary run directory structure."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    return run_dir
