"""MetricsCollector: unified cost/quality/efficiency metrics for all methods.

Every runner (standalone or GEPA-based) creates a MetricsCollector at startup
and calls its record_* methods throughout optimization. At the end, call
finalize() to compute derived metrics and get a dict for metrics.json.
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsCollector:
    """Collects cost, quality, and efficiency metrics across all methods."""

    # Cost metrics (primary)
    rollout_count: int = 0
    reflection_call_count: int = 0

    # Token breakdown
    task_input_tokens: int = 0
    task_output_tokens: int = 0
    reflection_input_tokens: int = 0
    reflection_output_tokens: int = 0

    # Timing
    start_time: float = field(default_factory=time.time)

    # Trajectories (for iterative methods)
    val_score_trajectory: list[tuple[int, float]] = field(default_factory=list)
    best_val_trajectory: list[tuple[int, float]] = field(default_factory=list)
    rollout_trajectory: list[tuple[int, int]] = field(default_factory=list)

    # Method-specific metrics (each method populates its own keys)
    method_specific: dict[str, Any] = field(default_factory=dict)

    # Error tracking (Task 17)
    task_error_count: int = 0
    reflection_error_count: int = 0

    # Internal tracking
    _best_val: float = 0.0
    _convergence_iteration: int | None = None
    _convergence_patience: int = 10  # iterations without improvement before marking convergence
    _no_improve_count: int = 0

    def record_rollouts(self, n: int = 1, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record n task evaluation rollouts with optional token counts."""
        self.rollout_count += n
        self.task_input_tokens += input_tokens
        self.task_output_tokens += output_tokens

    def record_reflection_call(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record a single reflection/mutation LLM call."""
        self.reflection_call_count += 1
        self.reflection_input_tokens += input_tokens
        self.reflection_output_tokens += output_tokens

    def record_task_error(self) -> None:
        """Increment task LM error counter."""
        self.task_error_count += 1

    def record_reflection_error(self) -> None:
        """Increment reflection LM error counter."""
        self.reflection_error_count += 1

    def record_val_score(self, iteration: int, score: float) -> None:
        """Record a validation score checkpoint (for iterative methods)."""
        self.val_score_trajectory.append((iteration, score))
        self.rollout_trajectory.append((iteration, self.rollout_count))

        if score > self._best_val:
            self._best_val = score
            self._no_improve_count = 0
            self._convergence_iteration = iteration
        else:
            self._no_improve_count += 1

        self.best_val_trajectory.append((iteration, self._best_val))

    @property
    def total_tokens(self) -> int:
        return (
            self.task_input_tokens
            + self.task_output_tokens
            + self.reflection_input_tokens
            + self.reflection_output_tokens
        )

    @property
    def wall_clock_seconds(self) -> float:
        return time.time() - self.start_time

    def finalize(
        self,
        test_score: float,
        best_prompt: str | dict[str, str],
        test_example_scores: list[float] | None = None,
        test_example_ids: list[str] | None = None,
        model: str = "",
        model_tag: str = "",
        benchmark: str = "",
        seed: int = -1,
        method: str = "",
        seed_prompt: str = "",
    ) -> dict[str, Any]:
        """Compute derived metrics and return full dict for metrics.json.

        Call this once at the end of optimization, after test evaluation.
        """
        wall_clock = self.wall_clock_seconds
        prompt_text = best_prompt if isinstance(best_prompt, str) else best_prompt.get("system_prompt", "")
        seed_text = seed_prompt if isinstance(seed_prompt, str) else ""

        # Derived efficiency metrics
        score_per_1k = (
            test_score / (self.rollout_count / 1000)
            if self.rollout_count > 0
            else 0.0
        )

        # Prompt edit distance metrics (Task 15)
        prompt_char_delta = len(prompt_text) - len(seed_text)
        prompt_levenshtein_ratio = difflib.SequenceMatcher(None, seed_text, prompt_text).ratio()
        prompt_length_tokens_approx = len(prompt_text) // 4
        prompt_word_count = len(prompt_text.split())

        return {
            # Cost metrics (4 primary)
            "rollout_count": self.rollout_count,
            "reflection_call_count": self.reflection_call_count,
            "wall_clock_seconds": round(wall_clock, 2),
            "total_tokens": self.total_tokens,
            # Token breakdown
            "task_input_tokens": self.task_input_tokens,
            "task_output_tokens": self.task_output_tokens,
            "reflection_input_tokens": self.reflection_input_tokens,
            "reflection_output_tokens": self.reflection_output_tokens,
            # Quality
            "test_score": test_score,
            "model": model,
            "model_tag": model_tag,
            "benchmark": benchmark,
            "seed": seed,
            "method": method,
            "val_score": self._best_val,
            "prompt_length_chars": len(prompt_text),
            "prompt_length_tokens": prompt_length_tokens_approx,  # rough estimate
            "test_example_scores": test_example_scores or [],
            "test_example_ids": test_example_ids or [],
            # Efficiency (derived)
            "score_per_1k_rollouts": round(score_per_1k, 4),
            "convergence_iteration": self._convergence_iteration,
            # Prompt edit distance (Task 15)
            "prompt_char_delta": prompt_char_delta,
            "prompt_levenshtein_ratio": round(prompt_levenshtein_ratio, 4),
            "prompt_word_count": prompt_word_count,
            # Error tracking (Task 17)
            "task_error_count": self.task_error_count,
            "reflection_error_count": self.reflection_error_count,
            # Trajectories
            "val_score_trajectory": self.val_score_trajectory,
            "rollout_trajectory": self.rollout_trajectory,
            "best_val_trajectory": self.best_val_trajectory,
            # Method-specific
            **self.method_specific,
        }


REQUIRED_METRIC_FIELDS = [
    "rollout_count",
    "reflection_call_count",
    "task_input_tokens",
    "task_output_tokens",
    "reflection_input_tokens",
    "reflection_output_tokens",
    "total_tokens",
    "wall_clock_seconds",
    "model",
    "model_tag",
    "benchmark",
    "seed",
    "method",
]


def validate_metrics(data: dict) -> list[str]:
    """Return list of missing required fields in a metrics.json dict."""
    return [f for f in REQUIRED_METRIC_FIELDS if not data.get(f) and data.get(f) != 0]
