"""SCALPEL LiteLLM client for the raycluster vLLM endpoint.

Phase 1 of SCALPEL. Provides a thin synchronous wrapper around
``litellm.completion`` that:

  * pins thinking-mode OFF (per SCALPEL Q4),
  * targets the single shared raycluster endpoint (per SCALPEL Q8),
  * exposes ``_last_usage`` for ``iso_harness.experiment.logging_lm.LoggingLM``,
  * supports ``guided_json`` reflection calls with xgrammar / guidance fallback,
  * caps in-flight requests with a ``threading.Semaphore`` to play nicely with
    the shared endpoint.

Hardware profiles other than ``raycluster`` raise ``NotImplementedError`` —
RunPod support is a future concern.

The schema in :data:`EDIT_LIST_SCHEMA` mirrors §5.4 of ``docs/scalpel/SCALPEL.md``
and is the schema Phase 3 will hand to ``LiteLLMClient.reflect``.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import namedtuple
from typing import Any, Literal

import litellm

from iso_harness.experiment.logging_lm import LoggingLM
from scripts.raycluster.config import (
    INFERENCE_BASE_URL,
    MAX_TOKENS_QA,
    MAX_TOKENS_REFLECT,
    MODEL_NAME,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)

__all__ = [
    "EDIT_LIST_SCHEMA",
    "LiteLLMClient",
    "Usage",
    "wrap_with_logging",
]

logger = logging.getLogger(__name__)

Usage = namedtuple("Usage", ["prompt_tokens", "completion_tokens"])

# JSON schema verbatim from docs/scalpel/SCALPEL.md §5.4.  Phase 3's reflection
# builder will pass this into LiteLLMClient.reflect via guided_json.
EDIT_LIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["edits", "lessons"],
    "properties": {
        "edits": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "required": ["operation", "target_span", "content"],
                "properties": {
                    "operation": {"enum": ["REPLACE", "APPEND", "DELETE", "INSERT"]},
                    "target_span": {"enum": ["S1", "S2", "S3", "S4", "S5", "S6"]},
                    "target_line": {"type": ["integer", "null"], "minimum": 1},
                    "content": {"type": "string"},
                },
            },
        },
        "lessons": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 150},
        },
    },
}

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


class LiteLLMClient:
    """Synchronous LiteLLM wrapper for the raycluster vLLM endpoint.

    The default ``__call__`` performs a *task* invocation; ``reflect`` performs
    a reflection invocation, optionally with a ``guided_json`` schema.  Both
    paths force ``enable_thinking=False`` per SCALPEL Q4.

    The instance exposes ``_last_usage`` (a :class:`Usage` namedtuple) which
    :class:`iso_harness.experiment.logging_lm.LoggingLM` reads after every call.
    """

    def __init__(
        self,
        *,
        model: str = MODEL_NAME,
        api_base: str = INFERENCE_BASE_URL,
        max_tokens_task: int = MAX_TOKENS_QA,
        max_tokens_reflect: int = MAX_TOKENS_REFLECT,
        temperature_task: float = TEMPERATURE,
        temperature_reflect: float = 0.7,
        top_p: float = TOP_P,
        top_k: int = TOP_K,
        request_timeout: float = 120.0,
        max_concurrency: int = 64,
        num_retries: int = 3,
        hardware_profile: str | None = None,
    ) -> None:
        profile = hardware_profile or os.environ.get(
            "SCALPEL_HARDWARE_PROFILE", "raycluster"
        )
        if profile != "raycluster":
            raise NotImplementedError(
                f"Hardware profile {profile!r} is not yet supported"
            )
        self.hardware_profile = profile

        self.model = model
        self.api_base = api_base
        self.max_tokens_task = max_tokens_task
        self.max_tokens_reflect = max_tokens_reflect
        self.temperature_task = temperature_task
        self.temperature_reflect = temperature_reflect
        self.top_p = top_p
        self.top_k = top_k
        self.request_timeout = request_timeout
        self.num_retries = num_retries

        self._semaphore = threading.Semaphore(max_concurrency)
        self._last_usage: Usage = Usage(0, 0)

    # ------------------------------------------------------------------ #
    # Public call protocol expected by LoggingLM
    # ------------------------------------------------------------------ #

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        """Default invocation — a *task* call."""
        return self._call(prompt, role="task")

    def task(self, prompt: str | list[dict[str, Any]]) -> str:
        """Explicit task call.  Identical to :meth:`__call__`."""
        return self._call(prompt, role="task")

    def reflect(
        self,
        prompt: str | list[dict[str, Any]],
        guided_json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Reflection call.

        If ``guided_json_schema`` is provided, the schema is sent through vLLM's
        ``guided_json`` extension with the ``xgrammar`` backend.  If the server
        rejects xgrammar (recognised by the substring ``"xgrammar"`` in the
        error message), one retry is issued with the ``guidance`` backend.
        After both attempts fail, the original exception is re-raised.
        """
        return self._call(prompt, role="reflect", guided_json_schema=guided_json_schema)

    # ------------------------------------------------------------------ #
    # Core implementation
    # ------------------------------------------------------------------ #

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        role: Literal["task", "reflect"],
        guided_json_schema: dict[str, Any] | None,
        guided_decoding_backend: str,
    ) -> dict[str, Any]:
        max_tokens = (
            self.max_tokens_task if role == "task" else self.max_tokens_reflect
        )
        temperature = (
            self.temperature_task if role == "task" else self.temperature_reflect
        )

        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": self.top_k,
        }
        if role == "reflect" and guided_json_schema is not None:
            extra_body["guided_json"] = guided_json_schema
            extra_body["guided_decoding_backend"] = guided_decoding_backend

        return {
            "model": f"hosted_vllm/{self.model}",
            "api_base": self.api_base,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": self.top_p,
            "extra_body": extra_body,
            "timeout": self.request_timeout,
            "num_retries": self.num_retries,
        }

    def _call(
        self,
        prompt: str | list[dict[str, Any]],
        role: Literal["task", "reflect"],
        guided_json_schema: dict[str, Any] | None = None,
    ) -> str:
        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = list(prompt)

        with self._semaphore:
            kwargs = self._build_kwargs(
                messages=messages,
                role=role,
                guided_json_schema=guided_json_schema,
                guided_decoding_backend="xgrammar",
            )
            try:
                resp = litellm.completion(**kwargs)
            except Exception as exc:
                # Retry with the guidance backend only when xgrammar was the
                # failure point and we were issuing a guided call.
                if (
                    role == "reflect"
                    and guided_json_schema is not None
                    and "xgrammar" in str(exc).lower()
                ):
                    logger.warning(
                        "xgrammar guided decoding failed (%s); retrying with "
                        "guided_decoding_backend='guidance'",
                        exc,
                    )
                    fallback_kwargs = self._build_kwargs(
                        messages=messages,
                        role=role,
                        guided_json_schema=guided_json_schema,
                        guided_decoding_backend="guidance",
                    )
                    try:
                        resp = litellm.completion(**fallback_kwargs)
                    except Exception:
                        logger.warning(
                            "guidance backend also failed; re-raising original "
                            "xgrammar exception"
                        )
                        raise exc
                else:
                    raise

        # LiteLLM's `completion()` is typed as Union[ModelResponse, CustomStreamWrapper];
        # in non-streaming mode it's always ModelResponse with `.choices`. Access
        # defensively to satisfy the type checker (the streaming union member has no
        # `.choices`, but we never request streaming).
        choices = getattr(resp, "choices", None) or []  # pyright: ignore[reportAttributeAccessIssue]
        msg = choices[0].message if choices else None
        text = (
            getattr(msg, "content", None)
            or getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        )
        text = _THINK_BLOCK.sub("", text).strip()

        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = (
            int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        )
        self._last_usage = Usage(prompt_tokens, completion_tokens)

        return text


def wrap_with_logging(
    client: LiteLLMClient,
    writer: Any,
    role: Literal["task", "reflection"],
) -> LoggingLM:
    """Convenience helper: wrap a :class:`LiteLLMClient` in :class:`LoggingLM`.

    ``LoggingLM`` calls ``client(prompt)`` and reads ``client._last_usage`` —
    the two pieces of the call-protocol :class:`LiteLLMClient` provides.
    """
    return LoggingLM(lm=client, writer=writer, role=role)
