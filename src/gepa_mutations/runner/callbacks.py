"""Metrics callback for capturing diagnostic data during GEPA optimization runs.

Hooks into GEPA's callback system to capture score-vs-rollout curves, acceptance rates,
Pareto front evolution, merge statistics, and per-example score matrices.

These diagnostics are essential for Phase 2 mutation design.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class IterationMetrics:
    """Metrics captured for a single GEPA iteration."""

    iteration: int
    wall_clock_seconds: float
    proposal_accepted: bool
    candidate_idx: int | None = None
    candidate_score: float | None = None
    new_candidate_idx: int | None = None
    new_score: float | None = None
    rejection_reason: str | None = None
    merge_attempted: bool = False
    merge_accepted: bool = False
    merge_rejected_reason: str | None = None
    pareto_front_size: int | None = None
    pareto_displaced: list[int] | None = None
    metric_calls_used: int = 0
    metric_calls_delta: int = 0


@dataclass
class ConvergencePoint:
    """A single point on the dense score-vs-rollout convergence curve."""

    rollout_idx: int
    iteration: int
    running_best_val_score: float


@dataclass
class RunMetrics:
    """Aggregated metrics for a complete GEPA optimization run."""

    benchmark: str
    seed: int
    iterations: list[IterationMetrics] = field(default_factory=list)
    total_metric_calls: int = 0
    total_iterations: int = 0
    best_candidate_idx: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    valset_scores: list[dict[str, Any]] = field(default_factory=list)
    convergence_curve: list[ConvergencePoint] = field(default_factory=list)

    @property
    def total_wall_clock(self) -> float:
        return self.end_time - self.start_time

    @property
    def acceptance_rate(self) -> float:
        if not self.iterations:
            return 0.0
        accepted = sum(1 for it in self.iterations if it.proposal_accepted)
        return accepted / len(self.iterations)

    @property
    def merge_acceptance_rate(self) -> float:
        merges = [it for it in self.iterations if it.merge_attempted]
        if not merges:
            return 0.0
        return sum(1 for m in merges if m.merge_accepted) / len(merges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "seed": self.seed,
            "total_metric_calls": self.total_metric_calls,
            "total_iterations": self.total_iterations,
            "best_candidate_idx": self.best_candidate_idx,
            "total_wall_clock_seconds": self.total_wall_clock,
            "acceptance_rate": self.acceptance_rate,
            "merge_acceptance_rate": self.merge_acceptance_rate,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "wall_clock_seconds": it.wall_clock_seconds,
                    "proposal_accepted": it.proposal_accepted,
                    "candidate_idx": it.candidate_idx,
                    "candidate_score": it.candidate_score,
                    "new_candidate_idx": it.new_candidate_idx,
                    "new_score": it.new_score,
                    "rejection_reason": it.rejection_reason,
                    "merge_attempted": it.merge_attempted,
                    "merge_accepted": it.merge_accepted,
                    "merge_rejected_reason": it.merge_rejected_reason,
                    "pareto_front_size": it.pareto_front_size,
                    "pareto_displaced": it.pareto_displaced,
                    "metric_calls_used": it.metric_calls_used,
                    "metric_calls_delta": it.metric_calls_delta,
                }
                for it in self.iterations
            ],
            "valset_scores": self.valset_scores,
            "convergence_curve": [
                {
                    "rollout_idx": pt.rollout_idx,
                    "iteration": pt.iteration,
                    "running_best_val_score": pt.running_best_val_score,
                }
                for pt in self.convergence_curve
            ],
        }


class MetricsCallback:
    """GEPA callback that captures diagnostic metrics during optimization.

    Implements the GEPACallback protocol by defining the relevant on_* methods.
    """

    def __init__(self, benchmark: str, seed: int):
        self.metrics = RunMetrics(benchmark=benchmark, seed=seed)
        self._iteration_start_time: float = 0.0
        self._current_iteration: IterationMetrics | None = None
        self._running_best_val_score: float = 0.0
        # Starts at 0; does not include seed evaluation cost (budget hook
        # is registered after seed eval in GEPA's engine).
        self._current_rollout_idx: int = 0

    def on_optimization_start(self, event: dict[str, Any]) -> None:
        self.metrics.start_time = time.time()

    def on_optimization_end(self, event: dict[str, Any]) -> None:
        self.metrics.end_time = time.time()
        self.metrics.total_iterations = event.get("total_iterations", 0)
        self.metrics.total_metric_calls = event.get("total_metric_calls", 0)
        self.metrics.best_candidate_idx = event.get("best_candidate_idx", 0)

    def on_iteration_start(self, event: dict[str, Any]) -> None:
        self._iteration_start_time = time.time()
        self._current_iteration = IterationMetrics(
            iteration=event.get("iteration", 0),
            wall_clock_seconds=0.0,
            proposal_accepted=False,
        )

    def on_iteration_end(self, event: dict[str, Any]) -> None:
        if self._current_iteration is not None:
            self._current_iteration.wall_clock_seconds = time.time() - self._iteration_start_time
            self._current_iteration.proposal_accepted = event.get("proposal_accepted", False)
            self.metrics.iterations.append(self._current_iteration)

            # Emit dense convergence point every iteration
            self.metrics.convergence_curve.append(ConvergencePoint(
                rollout_idx=self._current_rollout_idx,
                iteration=self._current_iteration.iteration,
                running_best_val_score=self._running_best_val_score,
            ))

            self._current_iteration = None

    def on_candidate_selected(self, event: dict[str, Any]) -> None:
        if self._current_iteration is not None:
            self._current_iteration.candidate_idx = event.get("candidate_idx")
            self._current_iteration.candidate_score = event.get("score")

    def on_candidate_accepted(self, event: dict[str, Any]) -> None:
        if self._current_iteration is not None:
            self._current_iteration.new_candidate_idx = event.get("new_candidate_idx")
            self._current_iteration.new_score = event.get("new_score")

    def on_candidate_rejected(self, event: dict[str, Any]) -> None:
        if self._current_iteration is not None:
            self._current_iteration.rejection_reason = event.get("reason")

    def on_merge_attempted(self, event: dict[str, Any]) -> None:
        if self._current_iteration is not None:
            self._current_iteration.merge_attempted = True

    def on_merge_accepted(self, event: dict[str, Any]) -> None:
        if self._current_iteration is not None:
            self._current_iteration.merge_accepted = True

    def on_merge_rejected(self, event: dict[str, Any]) -> None:
        if self._current_iteration is not None:
            self._current_iteration.merge_rejected_reason = event.get("reason")

    def on_pareto_front_updated(self, event: dict[str, Any]) -> None:
        if self._current_iteration is not None:
            new_front = event.get("new_front", [])
            self._current_iteration.pareto_front_size = len(new_front)
            self._current_iteration.pareto_displaced = event.get("displaced_candidates", [])

    def on_budget_updated(self, event: dict[str, Any]) -> None:
        self._current_rollout_idx = event.get("metric_calls_used", 0)
        if self._current_iteration is not None:
            self._current_iteration.metric_calls_used = self._current_rollout_idx
            self._current_iteration.metric_calls_delta = event.get("metric_calls_delta", 0)

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        avg_score = event.get("average_score", 0.0)
        if avg_score > self._running_best_val_score:
            self._running_best_val_score = avg_score
        self.metrics.valset_scores.append({
            "iteration": event.get("iteration"),
            "candidate_idx": event.get("candidate_idx"),
            "average_score": avg_score,
            "is_best_program": event.get("is_best_program"),
        })

    def save(self, output_path: str | Path) -> None:
        """Save metrics to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
