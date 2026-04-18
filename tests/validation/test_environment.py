"""V1: Environment validation — Python, imports, disk."""

from __future__ import annotations

import sys


class TestEnvironment:
    def test_python_version(self):
        """Python >= 3.12"""
        assert sys.version_info >= (3, 12), f"Need Python 3.12+, got {sys.version}"

    def test_core_imports(self):
        """All critical packages importable."""
        import pydantic
        import pyarrow
        import yaml
        import duckdb
        assert pydantic is not None
        assert pyarrow is not None
        assert yaml is not None
        assert duckdb is not None

    def test_ml_imports(self):
        """ML packages importable (may not have GPU)."""
        import dspy
        import mlflow
        assert dspy is not None
        assert mlflow is not None

    def test_iso_harness_imports(self):
        """iso_harness package importable."""
        from iso_harness.experiment import schemas
        from iso_harness.experiment import config
        from iso_harness.experiment import context
        from iso_harness.experiment import jsonl_writer
        from iso_harness.experiment import checkpoint
        from iso_harness.experiment import orchestrator
        from iso_harness.experiment import monitor
        from iso_harness.experiment import consolidate
        from iso_harness.experiment import reporter
        from iso_harness.experiment import protocols
        assert schemas is not None
        assert config is not None
        assert context is not None
        assert jsonl_writer is not None
        assert checkpoint is not None
        assert orchestrator is not None
        assert monitor is not None
        assert consolidate is not None
        assert reporter is not None
        assert protocols is not None
