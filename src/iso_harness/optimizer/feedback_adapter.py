"""Adapts existing benchmark evaluators to ISO's FeedbackFunction protocol.

Existing evaluators return (float, str). ISO expects {score, feedback, metadata}.
"""

from __future__ import annotations

from typing import Any, Callable


def adapt_evaluator_to_feedback_fn(
    evaluator: Callable,
) -> Callable:
    """Bridge (example, prediction) -> (float, str) evaluators to ISO's metric contract.

    ISO expects: (gold, pred, trace=None, pred_name=None) -> {"score": float, "feedback": str, "metadata": dict}
    """
    def feedback_fn(gold: Any, pred: Any, trace: Any = None, pred_name: str | None = None) -> dict:
        score, feedback = evaluator(gold, pred)
        return {
            "score": float(score),
            "feedback": str(feedback),
            "metadata": {},
        }
    return feedback_fn
