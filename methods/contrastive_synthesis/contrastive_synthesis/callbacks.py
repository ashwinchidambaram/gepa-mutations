"""ContrastiveSynthesisCallback: tracks per-iteration synthesis diagnostics.

Records synthesized principles and contrastive pair counts for post-hoc
analysis of whether the synthesis step is active and producing useful output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SynthesisIterationRecord:
    """Record for a single synthesis iteration."""

    iteration: int
    principle: str
    n_pairs: int


@dataclass
class SynthesisRunMetrics:
    """Aggregated synthesis metrics across a full run."""

    iterations: list[SynthesisIterationRecord] = field(default_factory=list)

    @property
    def total_iterations(self) -> int:
        return len(self.iterations)

    @property
    def active_iterations(self) -> int:
        """Iterations where a non-empty principle was synthesized."""
        return sum(1 for r in self.iterations if r.principle)

    @property
    def mean_pairs_per_iter(self) -> float:
        if not self.iterations:
            return 0.0
        return sum(r.n_pairs for r in self.iterations) / len(self.iterations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_iterations": self.total_iterations,
            "active_iterations": self.active_iterations,
            "mean_pairs_per_iter": self.mean_pairs_per_iter,
            "synthesis_principles": [
                {"iteration": r.iteration, "principle": r.principle}
                for r in self.iterations
            ],
            "num_contrastive_pairs_per_iter": [r.n_pairs for r in self.iterations],
        }


class ContrastiveSynthesisCallback:
    """Callback that tracks synthesized principles and pair counts.

    Called directly by ContrastiveSynthesisProposer via record_synthesis()
    after each synthesis step.
    """

    def __init__(self) -> None:
        self._metrics = SynthesisRunMetrics()

    @property
    def metrics(self) -> SynthesisRunMetrics:
        return self._metrics

    def record_synthesis(
        self,
        iteration: int,
        principle: str,
        n_pairs: int,
    ) -> None:
        """Record synthesis results for one iteration.

        Args:
            iteration: Current optimization iteration number.
            principle: The synthesized principle string (may be empty if
                no pairs were found or synthesis failed).
            n_pairs: Number of contrastive pairs used for synthesis.
        """
        self._metrics.iterations.append(
            SynthesisIterationRecord(
                iteration=iteration,
                principle=principle,
                n_pairs=n_pairs,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return self._metrics.to_dict()
