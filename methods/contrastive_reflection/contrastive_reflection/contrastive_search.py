"""Contrastive candidate search for the contrastive reflection mutation.

Maintains a per-example score index on training data (NOT val data) and
finds candidates that outperformed the current candidate on specific examples.

CRITICAL: Do NOT use ``state.prog_candidate_val_subscores`` -- those are val IDs,
not train IDs. This module builds its own index from training evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContrastiveTrainIndex:
    """Tracks per-example scores on training data for contrastive search.

    Built inside ``propose()`` after ``adapter.evaluate()`` returns, using
    ``dict(zip(subsample_ids, eval_curr.scores))``.

    The index accumulates scores across iterations, mapping
    ``(candidate_idx, example_id) -> score``.
    """

    # {candidate_idx: {example_id: score}}
    scores: dict[int, dict[Any, float]] = field(default_factory=dict)
    # {candidate_idx: candidate_text_dict}
    candidates: dict[int, dict[str, str]] = field(default_factory=dict)

    def update(
        self,
        candidate_idx: int,
        candidate: dict[str, str],
        example_ids: list,
        scores: list[float],
    ) -> None:
        """Record scores for a candidate on specific examples."""
        if candidate_idx not in self.scores:
            self.scores[candidate_idx] = {}
            self.candidates[candidate_idx] = dict(candidate)
        for eid, score in zip(example_ids, scores):
            self.scores[candidate_idx][eid] = score


def find_contrastive_candidates(
    index: ContrastiveTrainIndex,
    current_candidate_idx: int,
    current_scores: dict[Any, float],  # {example_id: score}
    num_pairs: int = 3,
    min_score_gap: float = 0.1,
) -> list[dict[str, Any]]:
    """Find candidates that performed better than the current one on specific examples.

    Searches the accumulated training score index for candidates whose score on
    a given example exceeds the current candidate's score by at least
    ``min_score_gap``. Returns the top ``num_pairs`` pairs sorted by score gap.

    Returns:
        List of dicts, each containing:
        - ``candidate_idx``: int -- index of the better candidate
        - ``candidate``: dict[str, str] -- the candidate text mapping
        - ``example_id``: the example where it excelled
        - ``current_score``: score of current candidate on this example
        - ``contrastive_score``: score of the better candidate
        - ``score_gap``: difference (contrastive - current)
    """
    contrastive_pairs: list[dict[str, Any]] = []

    for example_id, current_score in current_scores.items():
        for cand_idx, cand_scores in index.scores.items():
            if cand_idx == current_candidate_idx:
                continue
            if example_id in cand_scores:
                gap = cand_scores[example_id] - current_score
                if gap >= min_score_gap:
                    contrastive_pairs.append({
                        "candidate_idx": cand_idx,
                        "candidate": index.candidates[cand_idx],
                        "example_id": example_id,
                        "current_score": current_score,
                        "contrastive_score": cand_scores[example_id],
                        "score_gap": gap,
                    })

    # Sort by score gap descending, take top N
    contrastive_pairs.sort(key=lambda x: x["score_gap"], reverse=True)
    return contrastive_pairs[:num_pairs]
