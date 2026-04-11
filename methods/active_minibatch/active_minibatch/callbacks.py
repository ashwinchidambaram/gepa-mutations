"""ActiveMinibatchCallback: feeds evaluation scores to ActiveMinibatchSampler.

Tracks which examples were sampled each iteration and feeds scores back to the
sampler so it can maintain its per-example disagreement index. Also records
method-specific diagnostics for post-hoc analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from active_minibatch.sampler import ActiveMinibatchSampler


@dataclass
class ActiveMinibatchIterationMetrics:
    """Diagnostics for a single active minibatch iteration."""

    iteration: int
    minibatch_ids: list[Any]
    scores: list[float]
    mean_disagreement: float
    n_active: int
    n_random: int


@dataclass
class ActiveMinibatchRunMetrics:
    """Aggregated active minibatch metrics across a full run."""

    iterations: list[ActiveMinibatchIterationMetrics] = field(default_factory=list)

    @property
    def total_active_selections(self) -> int:
        return sum(it.n_active for it in self.iterations)

    @property
    def total_random_selections(self) -> int:
        return sum(it.n_random for it in self.iterations)

    @property
    def active_vs_random_ratio(self) -> float:
        total_random = self.total_random_selections
        if total_random == 0:
            return float("inf")
        return self.total_active_selections / total_random

    @property
    def mean_disagreement_scores(self) -> list[float]:
        return [it.mean_disagreement for it in self.iterations]

    def to_dict(self, sampler: "ActiveMinibatchSampler | None" = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "total_active_selections": self.total_active_selections,
            "total_random_selections": self.total_random_selections,
            "active_vs_random_ratio": self.active_vs_random_ratio,
            "disagreement_scores": self.mean_disagreement_scores,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "n_active": it.n_active,
                    "n_random": it.n_random,
                    "mean_disagreement": it.mean_disagreement,
                    "num_examples": len(it.minibatch_ids),
                }
                for it in self.iterations
            ],
        }
        if sampler is not None:
            result["cache_hit_rate"] = sampler.cache_hit_rate
        return result


class ActiveMinibatchCallback:
    """Callback that feeds evaluation scores to the ActiveMinibatchSampler.

    Tracks the last sampled minibatch IDs from on_minibatch_sampled and
    feeds them — with scores — back to the sampler on on_evaluation_end.
    Only processes training evaluations (candidate_idx is not None, since
    proposal evaluations use the same minibatch but we want the current
    candidate's scores for disagreement tracking).

    Also records per-iteration diagnostics (disagreement, active vs random
    fractions) for post-hoc analysis.
    """

    def __init__(self, sampler: "ActiveMinibatchSampler") -> None:
        self._sampler = sampler
        self._metrics = ActiveMinibatchRunMetrics()
        self._last_minibatch_ids: list[Any] = []
        self._current_iteration: int = 0
        # Track whether the current evaluation is a training eval (with traces)
        self._current_is_training: bool = False

    @property
    def metrics(self) -> ActiveMinibatchRunMetrics:
        return self._metrics

    def on_iteration_start(self, event: dict[str, Any]) -> None:
        self._current_iteration = event.get("iteration", 0)

    def on_minibatch_sampled(self, event: dict[str, Any]) -> None:
        """Store the sampled minibatch IDs for use in on_evaluation_end."""
        self._last_minibatch_ids = list(event.get("minibatch_ids", []))

    def on_evaluation_start(self, event: dict[str, Any]) -> None:
        """Track whether this is a training evaluation (capture_traces=True)."""
        self._current_is_training = bool(event.get("capture_traces", False))

    def on_evaluation_end(self, event: dict[str, Any]) -> None:
        """Feed scores to the sampler and record disagreement metrics.

        Only processes training evaluations (capture_traces=True at start)
        so that proposal evaluations don't double-count scores.
        """
        if not self._current_is_training:
            return

        scores = event.get("scores", [])
        minibatch_ids = self._last_minibatch_ids

        if not minibatch_ids or not scores:
            return

        if len(scores) != len(minibatch_ids):
            return

        # Feed scores to sampler
        self._sampler.update_scores(minibatch_ids, scores)

        # Compute mean disagreement across the current minibatch
        disagreements = [self._sampler._compute_disagreement(eid) for eid in minibatch_ids]
        finite_disagreements = [d for d in disagreements if d != float("inf")]
        mean_disagreement = (
            sum(finite_disagreements) / len(finite_disagreements)
            if finite_disagreements
            else 0.0
        )

        # Infer active vs random split from sampler settings
        n_active = int(len(minibatch_ids) * (1 - self._sampler.fallback_ratio))
        n_random = len(minibatch_ids) - n_active

        self._metrics.iterations.append(
            ActiveMinibatchIterationMetrics(
                iteration=self._current_iteration,
                minibatch_ids=list(minibatch_ids),
                scores=list(scores),
                mean_disagreement=mean_disagreement,
                n_active=n_active,
                n_random=n_random,
            )
        )

    def record_iteration(
        self,
        iteration: int,
        minibatch_ids: list[Any],
        scores: list[float],
        mean_disagreement: float,
        n_active: int,
        n_random: int,
    ) -> None:
        """Directly record metrics for an active minibatch iteration.

        Alternative to using on_evaluation_end — can be called directly
        from the runner or proposer if needed.
        """
        self._metrics.iterations.append(
            ActiveMinibatchIterationMetrics(
                iteration=iteration,
                minibatch_ids=minibatch_ids,
                scores=scores,
                mean_disagreement=mean_disagreement,
                n_active=n_active,
                n_random=n_random,
            )
        )
