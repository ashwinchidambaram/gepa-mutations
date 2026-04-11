"""Difficulty estimation for ESO (Ecological Succession Optimization).

Evaluates the seed prompt multiple times per example (with temperature > 0 for
variance) and computes mean scores per example to get continuous difficulty
estimates. This avoids the empty-tier problem with binary 0/1 scoring.
"""

from __future__ import annotations

from collections import defaultdict


def estimate_difficulty(
    adapter: object,
    trainset: list,
    seed_candidate: dict[str, str],
    n_evals: int = 3,
) -> dict[int, float]:
    """Evaluate seed prompt n_evals times per example, return mean scores per index.

    Args:
        adapter: GEPAAdapter instance (from get_adapter()).
        trainset: Training examples to evaluate.
        seed_candidate: Seed prompt dict, e.g. {"system_prompt": "..."}.
        n_evals: Number of evaluation passes per example (temperature > 0 gives
                 score variance, enabling continuous difficulty estimation).

    Returns:
        Dict mapping example index -> mean score across n_evals passes.
        Score of 1.0 = easy (always correct), 0.0 = hard (always wrong),
        0.5 = medium (sometimes correct).
    """
    all_scores: dict[int, list[float]] = defaultdict(list)
    for _ in range(n_evals):
        eval_out = adapter.evaluate(trainset, seed_candidate, capture_traces=False)
        for i, s in enumerate(eval_out.scores):
            all_scores[i].append(s)
    return {i: sum(scores) / len(scores) for i, scores in all_scores.items()}


def partition_by_difficulty(
    difficulty_scores: dict[int, float],
    easy_pct: float = 0.20,
    medium_pct: float = 0.30,
) -> tuple[list[int], list[int], list[int]]:
    """Partition example indices into easy/medium/hard tiers by mean score.

    Examples are sorted by decreasing mean score (higher = easier). The top
    easy_pct fraction go to "easy", the next medium_pct to "medium", and the
    remainder to "hard".

    Args:
        difficulty_scores: Dict from estimate_difficulty().
        easy_pct: Fraction of examples assigned to the easy tier.
        medium_pct: Fraction of examples assigned to the medium tier.

    Returns:
        (easy_ids, medium_ids, hard_ids) — lists of example indices.
    """
    sorted_indices = sorted(
        difficulty_scores.keys(),
        key=lambda i: difficulty_scores[i],
        reverse=True,  # highest score (easiest) first
    )
    n = len(sorted_indices)
    n_easy = max(1, int(n * easy_pct))
    n_medium = max(1, int(n * medium_pct))

    easy = sorted_indices[:n_easy]
    medium = sorted_indices[n_easy : n_easy + n_medium]
    hard = sorted_indices[n_easy + n_medium :]

    return easy, medium, hard
