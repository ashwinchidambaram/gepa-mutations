"""TrackedLM: wrapper that records token usage from LM calls into MetricsCollector.

Usage:
    collector = MetricsCollector()
    reflection_lm = build_reflection_lm(settings)
    tracked_reflection = TrackedLM(reflection_lm, collector, role="reflection")
    # Use tracked_reflection in place of reflection_lm
"""

from __future__ import annotations

from typing import Any

from gepa_mutations.metrics.collector import MetricsCollector


class TrackedLM:
    """Wraps an LM instance, intercepting token usage after each call.

    Reads ``lm._last_usage`` (set by the patched ``LM.__call__``) and
    accumulates token counts in the provided MetricsCollector.

    Args:
        lm: The underlying LM instance (from experiment.py).
        collector: MetricsCollector to accumulate token counts into.
        role: Either "task" or "reflection" — determines which token
              counters are incremented.
    """

    def __init__(self, lm: Any, collector: MetricsCollector, role: str = "task"):
        self._lm = lm
        self._collector = collector
        self._role = role

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        result = self._lm(prompt)

        # Extract token usage stored by the patched LM.__call__
        usage = getattr(self._lm, "_last_usage", None)
        if usage is not None:
            inp = getattr(usage, "prompt_tokens", 0) or 0
            out = getattr(usage, "completion_tokens", 0) or 0

            if self._role == "task":
                self._collector.task_input_tokens += inp
                self._collector.task_output_tokens += out
            else:
                self._collector.reflection_input_tokens += inp
                self._collector.reflection_output_tokens += out

        if self._role == "reflection":
            self._collector.reflection_call_count += 1

        return result

    def __repr__(self) -> str:
        return f"TrackedLM(role={self._role!r}, lm={self._lm!r})"
