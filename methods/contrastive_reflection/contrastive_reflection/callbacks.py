"""ContrastiveMetricsCallback: tracks contrastive reflection diagnostics.

Records per-iteration: how many contrastive pairs were found, how many
were injected, and the score gaps — enabling post-hoc analysis of
whether the contrastive mechanism was active and useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContrastiveIterationMetrics:
    """Metrics for a single contrastive reflection iteration."""

    iteration: int
    num_pairs_found: int
    num_pairs_used: int
    score_gaps: list[float]
    accepted: bool


@dataclass
class ContrastiveRunMetrics:
    """Aggregated contrastive reflection metrics across a full run."""

    iterations: list[ContrastiveIterationMetrics] = field(default_factory=list)

    @property
    def total_iterations(self) -> int:
        return len(self.iterations)

    @property
    def active_iterations(self) -> int:
        """Iterations where contrastive pairs were actually injected."""
        return sum(1 for it in self.iterations if it.num_pairs_used > 0)

    @property
    def active_ratio(self) -> float:
        """Fraction of iterations where contrastive injection was active."""
        if not self.iterations:
            return 0.0
        return self.active_iterations / len(self.iterations)

    @property
    def mean_pairs_found(self) -> float:
        if not self.iterations:
            return 0.0
        return sum(it.num_pairs_found for it in self.iterations) / len(self.iterations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_iterations": self.total_iterations,
            "active_iterations": self.active_iterations,
            "active_ratio": self.active_ratio,
            "mean_pairs_found": self.mean_pairs_found,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "num_pairs_found": it.num_pairs_found,
                    "num_pairs_used": it.num_pairs_used,
                    "score_gaps": it.score_gaps,
                    "accepted": it.accepted,
                }
                for it in self.iterations
            ],
        }


class ContrastiveMetricsCallback:
    """Captures contrastive reflection diagnostics per iteration.

    Called directly by ContrastiveReflectionProposer via record_iteration().
    """

    def __init__(self) -> None:
        self._metrics = ContrastiveRunMetrics()

    @property
    def metrics(self) -> ContrastiveRunMetrics:
        return self._metrics

    def record_iteration(
        self,
        iteration: int,
        num_pairs_found: int,
        num_pairs_used: int,
        score_gaps: list[float],
        accepted: bool = True,
    ) -> None:
        """Record contrastive metrics for one iteration."""
        self._metrics.iterations.append(
            ContrastiveIterationMetrics(
                iteration=iteration,
                num_pairs_found=num_pairs_found,
                num_pairs_used=num_pairs_used,
                score_gaps=score_gaps,
                accepted=accepted,
            )
        )
