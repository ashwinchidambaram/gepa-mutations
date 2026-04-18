"""Context variable propagation for ISO experiment state.

Uses Python contextvars to flow run_id, round_num, candidate_id, and phase
through the call stack without explicit parameter threading. The LoggingLM
wrapper reads these to tag each JSONL log entry.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

current_run_id: ContextVar[str] = ContextVar("current_run_id")
current_round_num: ContextVar[int] = ContextVar("current_round_num")
current_candidate_id: ContextVar[str] = ContextVar("current_candidate_id")
current_phase: ContextVar[str] = ContextVar("current_phase")


def set_context(
    *,
    run_id: str | None = None,
    round_num: int | None = None,
    candidate_id: str | None = None,
    phase: str | None = None,
) -> None:
    """Set one or more context variables. Only sets vars for non-None arguments."""
    if run_id is not None:
        current_run_id.set(run_id)
    if round_num is not None:
        current_round_num.set(round_num)
    if candidate_id is not None:
        current_candidate_id.set(candidate_id)
    if phase is not None:
        current_phase.set(phase)


def get_context() -> dict[str, Any]:
    """Return a snapshot of all current context variable values.

    Missing values are omitted from the dict (not set to None).
    """
    ctx: dict[str, Any] = {}
    for name, var in [
        ("run_id", current_run_id),
        ("round_num", current_round_num),
        ("candidate_id", current_candidate_id),
        ("phase", current_phase),
    ]:
        try:
            ctx[name] = var.get()
        except LookupError:
            pass
    return ctx


@contextmanager
def run_context(
    *,
    run_id: str | None = None,
    round_num: int | None = None,
    candidate_id: str | None = None,
    phase: str | None = None,
):
    """Context manager that sets context vars and resets them on exit.

    Usage:
        with run_context(run_id="abc-123", phase="pilot"):
            # all code here sees run_id="abc-123", phase="pilot"
            ...
        # context vars reset to their previous values
    """
    tokens: list[tuple[ContextVar, Token]] = []

    if run_id is not None:
        tokens.append((current_run_id, current_run_id.set(run_id)))
    if round_num is not None:
        tokens.append((current_round_num, current_round_num.set(round_num)))
    if candidate_id is not None:
        tokens.append((current_candidate_id, current_candidate_id.set(candidate_id)))
    if phase is not None:
        tokens.append((current_phase, current_phase.set(phase)))

    try:
        yield
    finally:
        for var, token in tokens:
            var.reset(token)
