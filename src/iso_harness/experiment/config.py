"""YAML-based experiment configuration with Pydantic validation.

Loads pilot.yaml or full.yaml, validates all fields, and composes with
the existing gepa_mutations Settings for env-var based runtime config.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for a single vLLM-served model."""

    model_id: str
    hf_sha: str = "<PIN_AT_IMPLEMENTATION>"
    dtype: str = "bfloat16"
    quantization: str | None = None
    max_model_len: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=2.0, default=1.0)
    max_tokens: int = Field(gt=0, default=8192)


class ModelsConfig(BaseModel):
    """Task and reflection model pair."""

    task: ModelConfig
    reflection: ModelConfig


class SmokeTestConfig(BaseModel):
    """Configuration for the pre-pilot smoke test."""

    optimizer: str = "gepa"
    benchmark: str = "ifbench"
    subset_size: int = Field(gt=0, default=50)
    budget_rollouts: int = Field(gt=0, default=100)


class BenchmarkBudget(BaseModel):
    """Per-benchmark rollout budget."""

    budget_rollouts: int = Field(gt=0)


class RunMatrixConfig(BaseModel):
    """Full experiment run matrix."""

    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    benchmarks: dict[str, BenchmarkBudget] = Field(default_factory=dict)
    optimizers: dict[str, dict] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    """MLflow and artifact logging settings."""

    mlflow_tracking_uri: str = "file:///workspace/mlflow"
    artifacts_dir: str = "/workspace/iso-experiment/runs"


class MonitoringConfig(BaseModel):
    """System monitoring intervals and thresholds."""

    gpu_poll_interval_sec: int = Field(gt=0, default=30)
    kv_cache_poll_interval_sec: int = Field(gt=0, default=60)
    disk_check_interval_min: int = Field(gt=0, default=10)
    disk_min_free_gb: int = Field(ge=0, default=20)


class ISOExperimentConfig(BaseModel):
    """Top-level experiment configuration loaded from YAML."""

    phase: Literal["pilot", "full"]
    runpod_template_id: str = "<PIN_AT_IMPLEMENTATION>"
    models: ModelsConfig
    smoke_test: SmokeTestConfig | None = None
    run_matrix: RunMatrixConfig | None = None
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @model_validator(mode="after")
    def _check_hf_sha_pinned(self) -> "ISOExperimentConfig":
        """Warn in pilot, error in full if HF SHAs aren't pinned."""
        for name, model in [("task", self.models.task), ("reflection", self.models.reflection)]:
            if model.hf_sha == "<PIN_AT_IMPLEMENTATION>":
                if self.phase == "full":
                    raise ValueError(
                        f"models.{name}.hf_sha must be pinned for full phase runs"
                    )
                else:
                    logger.warning(
                        "models.%s.hf_sha not pinned — acceptable for pilot, "
                        "must be pinned before full phase",
                        name,
                    )
        return self


def load_config(path: str | Path) -> ISOExperimentConfig:
    """Load and validate experiment config from a YAML file.

    Exits with code 2 on validation failure (invalid configuration).
    """
    path = Path(path)
    if not path.exists():
        print(f"ERROR: Config file not found: {path}", file=sys.stderr)
        sys.exit(2)

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        print(f"ERROR: Config file must be a YAML mapping: {path}", file=sys.stderr)
        sys.exit(2)

    try:
        return ISOExperimentConfig.model_validate(raw)
    except Exception as e:
        print(f"ERROR: Config validation failed for {path}:\n{e}", file=sys.stderr)
        sys.exit(2)
