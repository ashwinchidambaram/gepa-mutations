"""Evaluation helpers for standalone optimization methods.

Standalone methods (PTS, SMNO, SPDO, ACPCO) don't use GEPAEngine and need
to evaluate prompts directly via the adapter. These helpers provide a
consistent interface and integrate with MetricsCollector for rollout counting.
"""

from __future__ import annotations

from typing import Any

from gepa_mutations.metrics.collector import MetricsCollector


def evaluate_prompt(
    adapter: Any,
    dataset: list,
    candidate: dict[str, str],
    collector: MetricsCollector | None = None,
    indices: list[int] | None = None,
) -> tuple[float, list[float]]:
    """Evaluate a prompt on a dataset (or subset).

    Args:
        adapter: GEPAAdapter instance from get_adapter().
        dataset: Full dataset (train or val list).
        candidate: Prompt dict, e.g. {"system_prompt": "..."}.
        collector: Optional MetricsCollector for rollout counting.
        indices: If provided, evaluate only these indices from dataset.

    Returns:
        (mean_score, per_example_scores)
    """
    batch = [dataset[i] for i in indices] if indices is not None else dataset
    if not batch:
        return 0.0, []

    eval_out = adapter.evaluate(batch, candidate, capture_traces=False)
    scores = eval_out.scores

    if collector is not None:
        collector.record_rollouts(n=len(batch))

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return mean_score, list(scores)


def evaluate_prompt_with_traces(
    adapter: Any,
    dataset: list,
    candidate: dict[str, str],
    collector: MetricsCollector | None = None,
    indices: list[int] | None = None,
) -> tuple[float, list[float], Any]:
    """Like evaluate_prompt but also returns the full EvaluationBatch (for traces).

    Returns:
        (mean_score, per_example_scores, eval_batch)
    """
    batch = [dataset[i] for i in indices] if indices is not None else dataset
    if not batch:
        return 0.0, [], None

    eval_out = adapter.evaluate(batch, candidate, capture_traces=True)
    scores = eval_out.scores

    if collector is not None:
        collector.record_rollouts(n=len(batch))

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return mean_score, list(scores), eval_out
