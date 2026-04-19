"""Pydantic models for pilot phase YAML configs.

Two tracks are supported:
- Track 1 (``PilotPhaseAConfig``): ISO variants + baselines.
- Track 2 (``PilotPhaseBConfig``): Meta-optimizers.

``load_pilot_config`` auto-detects the track from the ``track`` field.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Union

import yaml
from pydantic import BaseModel, ValidationError

__all__ = [
    "PilotModelConfig",
    "PilotModelsConfig",
    "PilotBenchmarkConfig",
    "PilotSmokeTestConfig",
    "MetaOptimizerConfig",
    "PilotPhaseAConfig",
    "PilotPhaseBConfig",
    "load_pilot_config",
]


class PilotModelConfig(BaseModel):
    name: str
    port: int = 8000
    max_tokens: int = 8192


class PilotModelsConfig(BaseModel):
    task: PilotModelConfig
    reflection: PilotModelConfig
    meta: PilotModelConfig | None = None


class PilotBenchmarkConfig(BaseModel):
    budget: int
    seeds: list[int] = [0, 1, 2]
    subset_size: int | None = None


class PilotSmokeTestConfig(BaseModel):
    budget: int = 100
    subset_size: int = 20
    overrides: dict[str, Any] = {}


class MetaOptimizerConfig(BaseModel):
    n_episodes: int = 50
    surrogate_size: int = 20
    playbook_update_interval: int | None = None


class PilotPhaseAConfig(BaseModel):
    """Track 1: ISO variants + baselines."""

    phase: Literal["pilot"]
    track: Literal[1]
    models: PilotModelsConfig
    variants: list[str]
    baselines: list[str] = []
    benchmarks: dict[str, PilotBenchmarkConfig]
    smoke_test: PilotSmokeTestConfig | None = None


class PilotPhaseBConfig(BaseModel):
    """Track 2: Meta-optimizers."""

    phase: Literal["pilot"]
    track: Literal[2]
    models: PilotModelsConfig
    inner_variant: str
    meta_optimizers: dict[str, MetaOptimizerConfig]
    benchmark: str


def load_pilot_config(path: str | Path) -> Union[PilotPhaseAConfig, PilotPhaseBConfig]:
    """Load a pilot config YAML, auto-detecting track from the 'track' field.

    Args:
        path: Path to a pilot_phase_a.yaml or pilot_phase_b.yaml config file.

    Returns:
        A ``PilotPhaseAConfig`` (track 1) or ``PilotPhaseBConfig`` (track 2).

    Raises:
        SystemExit(2): On file-not-found, invalid YAML, missing fields, or
            unknown track value.
    """
    p = Path(path)
    if not p.exists():
        print(f"Error: config file not found: {p}", file=sys.stderr)
        sys.exit(2)

    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        print(f"Error: failed to parse YAML: {exc}", file=sys.stderr)
        sys.exit(2)

    if not isinstance(raw, dict):
        print("Error: config must be a YAML mapping", file=sys.stderr)
        sys.exit(2)

    track = raw.get("track")
    if track == 1:
        model_cls = PilotPhaseAConfig
    elif track == 2:
        model_cls = PilotPhaseBConfig
    else:
        print(
            f"Error: unknown track value {track!r}; expected 1 or 2",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        return model_cls.model_validate(raw)
    except ValidationError as exc:
        print(f"Error: invalid config:\n{exc}", file=sys.stderr)
        sys.exit(2)
