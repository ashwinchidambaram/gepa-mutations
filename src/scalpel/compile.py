"""Thin entry point for the SCALPEL optimizer.

This is the function the orchestrator (``iso_harness.experiment.run_fn``)
will call.  It wraps :class:`scalpel.optimizer.SCALPEL` with sensible
defaults and matches the DSPy-style ``compile(student, trainset, valset)``
signature.

Public surface:

* :func:`compile` — factory that constructs a SCALPEL and runs it.
"""

from __future__ import annotations

from typing import Any, Callable

from scalpel.benchmarks.adapter import BenchmarkAdapter
from scalpel.edits.grammar import StructuredPrompt
from scalpel.optimizer import SCALPEL

__all__ = ["compile"]


def compile(  # noqa: A001  -- DSPy-style API name
    student: dict[str, str | StructuredPrompt] | BenchmarkAdapter,
    trainset: list[Any],
    valset: list[Any],
    metric: Callable[[Any, Any], float] | None = None,
    feedback: Callable[..., str] | None = None,
    *,
    task_lm: Callable[..., Any],
    reflect_lm: Callable[..., Any],
    **kwargs: Any,
) -> dict[str, StructuredPrompt]:
    """Run SCALPEL on ``student`` and return the best system's prompts.

    If ``student`` is a :class:`BenchmarkAdapter`, ``metric`` and ``feedback``
    are pulled from the adapter when not supplied, and the seed prompt
    defaults to ``student.metadata["seed_prompt"]`` (or
    ``"You are a helpful assistant."``).  Otherwise ``student`` must be a
    ``dict[str, str | StructuredPrompt]`` mapping module name to seed prompt.
    """
    if isinstance(student, BenchmarkAdapter):
        if metric is None:
            metric = student.metric
        if feedback is None:
            feedback = student.feedback
        seed_prompt = student.metadata.get("seed_prompt", "You are a helpful assistant.")
        student_dict: dict[str, str | StructuredPrompt] = {"default": seed_prompt}
    else:
        student_dict = student

    optimizer = SCALPEL(task_lm=task_lm, reflect_lm=reflect_lm, **kwargs)
    return optimizer.compile(
        student_dict,
        trainset=trainset,
        valset=valset,
        metric=metric,
        feedback=feedback,
    )
