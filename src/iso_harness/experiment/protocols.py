"""Interface contracts for ISO experiment infrastructure.

Defines Protocol classes that ISO optimizer variants and feedback functions
must satisfy. These match the contracts in the infrastructure spec Section 11.2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict, runtime_checkable


class FeedbackResult(TypedDict):
    """Return type for feedback/metric functions."""

    score: float
    feedback: str
    metadata: dict


@runtime_checkable
class Optimizer(Protocol):
    """Optimizer protocol matching dspy.Teleprompter contract.

    Any optimizer (ISO variants, GEPA, MIPROv2) must implement this interface
    to be used by the orchestrator.
    """

    def compile(self, student: Any, trainset: list, valset: list) -> Any: ...


@runtime_checkable
class FeedbackFunction(Protocol):
    """Feedback/metric function protocol.

    Evaluates a prediction against a gold label, returning a score,
    natural-language feedback, and benchmark-specific metadata.
    """

    def __call__(
        self, gold: Any, pred: Any, trace: Any, pred_name: str
    ) -> FeedbackResult: ...


@runtime_checkable
class Checkpointable(Protocol):
    """Protocol for optimizers that support checkpoint/resume.

    Implementors save and restore their full internal state (candidate pool,
    round counter, metrics) to/from a directory on disk.
    """

    def save_state(self, path: Path) -> None: ...

    def load_state(self, path: Path) -> None: ...
