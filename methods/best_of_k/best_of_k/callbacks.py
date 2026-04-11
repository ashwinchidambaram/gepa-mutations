"""BestOfKMetricsCallback: tracks per-K metrics during optimization.

Records best-of-K specific diagnostics (K value, unique candidates per
iteration, winning k index, score distributions) whenever a candidate
produced by BestOfKProposer is accepted by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BestOfKIterationMetrics:
    """Metrics for a single best-of-K iteration."""

    iteration: int
    k_value: int
    unique_candidates: int
    winning_k_index: int
    all_k_scores: list[float]
    best_score: float
    accepted: bool


@dataclass
class BestOfKRunMetrics:
    """Aggregated best-of-K metrics across a full run."""

    iterations: list[BestOfKIterationMetrics] = field(default_factory=list)

    @property
    def total_iterations_with_k(self) -> int:
        """Number of iterations where K > 1 was used."""
        return sum(1 for it in self.iterations if it.k_value > 1)

    @property
    def average_unique_ratio(self) -> float:
        """Average ratio of unique candidates to K across iterations."""
        k_iters = [it for it in self.iterations if it.k_value > 1]
        if not k_iters:
            return 1.0
        return sum(it.unique_candidates / it.k_value for it in k_iters) / len(k_iters)

    @property
    def k0_win_rate(self) -> float:
        """Fraction of iterations where k=0 (first candidate) was the winner."""
        k_iters = [it for it in self.iterations if it.k_value > 1]
        if not k_iters:
            return 0.0
        return sum(1 for it in k_iters if it.winning_k_index == 0) / len(k_iters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_iterations_with_k": self.total_iterations_with_k,
            "average_unique_ratio": self.average_unique_ratio,
            "k0_win_rate": self.k0_win_rate,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "k_value": it.k_value,
                    "unique_candidates": it.unique_candidates,
                    "winning_k_index": it.winning_k_index,
                    "all_k_scores": it.all_k_scores,
                    "best_score": it.best_score,
                    "accepted": it.accepted,
                }
                for it in self.iterations
            ],
        }


class BestOfKMetricsCallback:
    """GEPA callback that captures best-of-K specific diagnostics.

    Since CandidateAcceptedEvent does not carry proposal metadata, this
    callback uses a shared reference to the proposer's latest metadata
    dict, which the BestOfKProposer populates before returning its
    CandidateProposal.  Call ``set_proposer()`` after construction to
    wire the reference.

    Alternatively, the callback can be populated directly via
    ``record_iteration()`` from the proposer.
    """

    def __init__(self) -> None:
        self._metrics = BestOfKRunMetrics()
        self._current_iteration: int = 0
        self._pending_metadata: dict[str, Any] | None = None

    @property
    def metrics(self) -> BestOfKRunMetrics:
        """Access the accumulated best-of-K metrics."""
        return self._metrics

    def record_iteration(
        self,
        iteration: int,
        k_value: int,
        unique_candidates: int,
        winning_k_index: int,
        all_k_scores: list[float],
        best_score: float,
        accepted: bool = True,
    ) -> None:
        """Directly record metrics for a best-of-K iteration.

        Called by BestOfKProposer after building its CandidateProposal,
        bypassing the engine's callback system which does not forward
        proposal metadata.
        """
        self._metrics.iterations.append(
            BestOfKIterationMetrics(
                iteration=iteration,
                k_value=k_value,
                unique_candidates=unique_candidates,
                winning_k_index=winning_k_index,
                all_k_scores=all_k_scores,
                best_score=best_score,
                accepted=accepted,
            )
        )

    def on_iteration_start(self, event: dict[str, Any]) -> None:
        self._current_iteration = event.get("iteration", 0)

    def on_candidate_accepted(self, event: dict[str, Any]) -> None:
        """Record best-of-K metrics when a K-proposal is accepted.

        Guards against duplicate recording since ``record_iteration()`` is
        called directly by the proposer before the engine fires this event.
        """
        # Skip if this iteration was already recorded via direct record_iteration() call
        if self._metrics.iterations and self._metrics.iterations[-1].iteration == self._current_iteration:
            return

        if self._pending_metadata:
            metadata = self._pending_metadata
            self._pending_metadata = None
        else:
            metadata = event.get("metadata", {})

        if not metadata:
            return

        k_value = metadata.get("mutation_candidates")
        if k_value is None:
            return

        unique_candidates = metadata.get("unique_candidates", k_value)
        winning_k_index = metadata.get("winning_k_index", 0)
        all_k_scores = metadata.get("all_k_scores", [])

        best_score = max(all_k_scores) if all_k_scores else event.get("new_score", 0.0)

        self._metrics.iterations.append(
            BestOfKIterationMetrics(
                iteration=self._current_iteration,
                k_value=k_value,
                unique_candidates=unique_candidates,
                winning_k_index=winning_k_index,
                all_k_scores=all_k_scores,
                best_score=best_score,
                accepted=True,
            )
        )
