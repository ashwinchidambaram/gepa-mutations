"""Minimal stub of gepa.core.adapter for cluster deployment.

Only provides EvaluationBatch — the dataclass used by evaluators.py.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

RolloutOutput = TypeVar("RolloutOutput")
Trajectory = TypeVar("Trajectory")


@dataclass
class EvaluationBatch(Generic[Trajectory, RolloutOutput]):
    outputs: list[RolloutOutput]
    scores: list[float]
    trajectories: list[Trajectory] | None = None
    objective_scores: list[dict[str, float]] | None = None
