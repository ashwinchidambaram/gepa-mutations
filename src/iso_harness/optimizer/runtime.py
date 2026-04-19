"""ISO optimizer runtime objects: ISORuntime, RolloutCounter, TraceStore.

``ISORuntime`` is the single context object threaded through all optimizer
calls within one run.  ``RolloutCounter`` wraps ``BudgetEnforcer`` with the
interface expected by the ISO spec.  ``TraceStore`` is an in-memory key-value
store for ``ModuleTrace`` objects indexed by ``(candidate_id, example_id)``.

A ``ContextVar`` accessor pair (``set_current_runtime`` / ``get_current_runtime``)
allows async/threaded code to retrieve the active runtime without explicit
parameter passing.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from iso_harness.optimizer.candidate import ModuleTrace

__all__ = [
    "ISORuntime",
    "RolloutCounter",
    "TraceStore",
    "set_current_runtime",
    "get_current_runtime",
    "runtime_context",
]


# ---------------------------------------------------------------------------
# RolloutCounter
# ---------------------------------------------------------------------------


class RolloutCounter:
    """Thread-safe rollout counter that wraps ``BudgetEnforcer``.

    In standalone mode (``enforcer=None``) it maintains its own internal
    counter, which is useful for testing without the full experiment harness.

    Args:
        enforcer: A ``BudgetEnforcer`` instance, or ``None`` for standalone
            mode.
    """

    def __init__(self, enforcer=None) -> None:
        if enforcer is None:
            self._standalone = True
            self._count = 0
        else:
            self._standalone = False
            self._enforcer = enforcer

    def increment(self, n: int = 1) -> None:
        """Record *n* rollouts consumed."""
        if self._standalone:
            self._count += n
        else:
            self._enforcer.record_rollouts(n)

    def value(self) -> int:
        """Total rollouts consumed so far."""
        if self._standalone:
            return self._count
        return self._enforcer.consumed

    def remaining(self, budget: int) -> int:
        """Rollouts remaining before *budget* is exhausted."""
        return max(0, budget - self.value())


# ---------------------------------------------------------------------------
# TraceStore
# ---------------------------------------------------------------------------


class TraceStore:
    """In-memory trace store keyed by ``(candidate_id, example_id)``.

    Stores ``ModuleTrace`` objects (or any object) produced during evaluation.
    The store is cleared between rounds via ``clear_round()``.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], Any] = {}

    def put(self, candidate_id: str, example_id: str, trace: Any) -> None:
        """Insert or overwrite the trace for a (candidate, example) pair."""
        self._store[(candidate_id, example_id)] = trace

    def get(self, candidate_id: str, example_id: str) -> Any | None:
        """Retrieve the trace, or ``None`` if not found."""
        return self._store.get((candidate_id, example_id))

    def get_worst_for_candidate(
        self,
        candidate_id: str,
        per_example_scores: dict[str, float],
        n: int,
    ) -> list[Any]:
        """Return up to *n* traces for the lowest-scoring examples.

        Args:
            candidate_id: Candidate whose traces to retrieve.
            per_example_scores: Mapping of ``example_id -> score``.
            n: Maximum number of traces to return.

        Returns:
            List of trace objects sorted by ascending score (worst first).
            Missing traces are skipped rather than included as ``None``.
        """
        sorted_examples = sorted(per_example_scores.items(), key=lambda x: x[1])
        result: list[Any] = []
        for ex_id, _ in sorted_examples:
            trace = self.get(candidate_id, ex_id)
            if trace is not None:
                result.append(trace)
            if len(result) >= n:
                break
        return result

    def clear_round(self) -> None:
        """Discard all stored traces (call at the end of each round)."""
        self._store.clear()

    def size(self) -> int:
        """Number of entries currently in the store."""
        return len(self._store)


# ---------------------------------------------------------------------------
# ISORuntime
# ---------------------------------------------------------------------------


@dataclass
class ISORuntime:
    """All per-run mutable state needed by the ISO optimizer.

    This object is created once per experiment run and passed (or accessed via
    ``get_current_runtime()``) by every component in the optimizer loop.

    Attributes:
        reflection_lm: Pre-configured reflection LM (DSPy-compatible).
        task_lm: Pre-configured task LM (DSPy-compatible).
        metric: Feedback function with signature
            ``(gold, pred, trace, pred_name) -> FeedbackResult``.
        run_id: Experiment run identifier (UUID4 string).
        seed: Random seed for this run.
        rng: Seeded ``random.Random`` instance for reproducible sampling.
        trace_store: In-memory store for ``ModuleTrace`` objects.
        rollout_counter: Wraps the budget enforcer.
        round_num: Current optimisation round (incremented by the engine).
    """

    reflection_lm: Any
    task_lm: Any
    metric: Callable
    run_id: str
    seed: int
    rng: random.Random
    trace_store: TraceStore
    rollout_counter: RolloutCounter
    round_num: int = 0
    rollout_writer: Any = None  # Optional JSONLWriter for per-rollout JSONL logging


# ---------------------------------------------------------------------------
# ContextVar accessor
# ---------------------------------------------------------------------------

_current_runtime: ContextVar[ISORuntime] = ContextVar("_current_runtime")


def set_current_runtime(runtime: ISORuntime) -> None:
    """Set the active ``ISORuntime`` in the current context."""
    _current_runtime.set(runtime)


def get_current_runtime() -> ISORuntime:
    """Return the active ``ISORuntime`` for the current context.

    Raises:
        LookupError: If no runtime has been set in this context.
    """
    return _current_runtime.get()


@contextmanager
def runtime_context(runtime: ISORuntime):
    """Context manager that sets the active runtime and resets it on exit.

    Usage::

        with runtime_context(rt):
            # get_current_runtime() returns rt here
            ...
        # previous runtime (or unset state) is restored

    Args:
        runtime: The ``ISORuntime`` to activate.
    """
    token = _current_runtime.set(runtime)
    try:
        yield runtime
    finally:
        _current_runtime.reset(token)
