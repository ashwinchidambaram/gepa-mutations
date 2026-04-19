"""LoggingLM: wraps LM with per-call JSONL logging and context variable tagging.

Sits on top of TrackedLM (or bare LM) in the wrapping chain:
  LoggingLM(TrackedLM(LM()))  -- full chain with token counting + logging
  LoggingLM(LM())             -- logging only, no MetricsCollector

Every call appends a RolloutRecord or ReflectionRecord to a JSONL file,
tagged with the current context variables (run_id, round_num, candidate_id).
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from iso_harness.experiment.context import get_context
from iso_harness.experiment.jsonl_writer import JSONLWriter
from iso_harness.experiment.schemas import ReflectionRecord, RolloutRecord


class LoggingLM:
    """LM wrapper that streams every call to append-only JSONL.

    Args:
        lm: The underlying LM-protocol object (LM, TrackedLM, or anything callable).
        writer: JSONLWriter instance to append records to.
        role: "task" for rollout logging, "reflection" for reflection logging.
        example_id_fn: Optional callable that returns the current example_id.
            If not provided, example_id defaults to "unknown".
    """

    def __init__(
        self,
        lm: Any,
        writer: JSONLWriter,
        role: Literal["task", "reflection"] = "task",
        example_id_fn: Any | None = None,
    ) -> None:
        self._lm = lm
        self._writer = writer
        self._role = role
        self._example_id_fn = example_id_fn

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str | list[str]:
        # Serialize prompt for logging
        if isinstance(prompt, str):
            prompt_text = prompt
        else:
            prompt_text = str(prompt)

        start = time.perf_counter()
        result = self._lm(prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Normalize result for logging: dspy.LM returns list[str], others return str
        if isinstance(result, list):
            result_text = result[0] if result else ""
        else:
            result_text = str(result)

        # Read token usage from the innermost LM
        lm = self._lm
        # Walk the wrapping chain to find _last_usage
        usage = getattr(lm, "_last_usage", None)
        if usage is None and hasattr(lm, "_lm"):
            usage = getattr(lm._lm, "_last_usage", None)

        tokens_in = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0 if usage else 0

        # Read context vars
        ctx = get_context()
        run_id = ctx.get("run_id", "")
        round_num = ctx.get("round_num", 0)
        candidate_id = ctx.get("candidate_id", "")

        # Get example_id
        example_id = "unknown"
        if self._example_id_fn is not None:
            try:
                example_id = str(self._example_id_fn())
            except Exception:
                pass

        now = datetime.now(timezone.utc)

        if self._role == "task":
            record = RolloutRecord(
                rollout_id=str(uuid.uuid4()),
                run_id=run_id,
                round_num=round_num,
                candidate_id=candidate_id,
                module_name=None,
                example_id=example_id,
                prompt=prompt_text,
                response=result_text,
                score=0.0,  # Score is filled in later by the evaluator
                feedback="",
                metadata={},
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                latency_ms=elapsed_ms,
                timestamp=now,
            )
        else:
            record = ReflectionRecord(
                reflection_id=str(uuid.uuid4()),
                run_id=run_id,
                round_num=round_num,
                triggered_by="mutation",  # Default; overridden via context if needed
                target_candidate_id=candidate_id,
                target_module=None,
                input_traces=[],
                input_prompt=prompt_text,
                output=result_text,
                parsed_candidate_before="",
                parsed_candidate_after="",
                diff="",
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                latency_ms=elapsed_ms,
                timestamp=now,
            )

        self._writer.append(record)
        return result

    @property
    def model(self) -> str:
        """Proxy the model name from the inner LM."""
        return getattr(
            self._lm,
            "model",
            getattr(self._lm, "_lm", object()).__class__.__name__,
        )

    def __repr__(self) -> str:
        return f"LoggingLM(role={self._role!r}, lm={self._lm!r})"
