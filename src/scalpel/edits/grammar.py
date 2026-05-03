"""SCALPEL EDIT grammar — pydantic v2 models for the addressable-span EDIT list.

Phase 3 of SCALPEL.  Defines the data model for the six addressable spans
(S1..S6, see ``docs/scalpel/SCALPEL.md`` §3.A) and the diff-edit operations
the reflector emits over them.

Public surface:

* :data:`SPAN_IDS` — the six span identifiers in canonical order.
* :data:`SpanId` — ``Literal`` alias used by every model below.
* :data:`SPAN_NAMES` — mapping from id to canonical name (e.g. ``"S3"`` ->
  ``"strategy_bullets"``).
* :class:`Span` — frozen pydantic model: id, name, content.
* :class:`StructuredPrompt` — frozen pydantic model: ordered list of spans,
  cached materialization, token count.
* :class:`Edit` — one of REPLACE / APPEND / DELETE / INSERT against a span.
* :class:`EditList` — a reflector's full proposal: ``edits`` + ``lessons``,
  both capped at four entries (matches §5.4 of the SCALPEL doc).
* :data:`EDIT_LIST_SCHEMA` — re-exported from :mod:`scalpel.llm.client` so
  that downstream callers have a single import surface for the grammar +
  schema.  The canonical definition lives in ``scalpel.llm.client`` (kept
  there to avoid a circular import); see that module's docstring.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from scalpel.llm.client import EDIT_LIST_SCHEMA as _CANONICAL_EDIT_LIST_SCHEMA

__all__ = [
    "EDIT_LIST_SCHEMA",
    "Edit",
    "EditList",
    "SPAN_IDS",
    "SPAN_NAMES",
    "Span",
    "SpanId",
    "StructuredPrompt",
]


SPAN_IDS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6")

SpanId = Literal["S1", "S2", "S3", "S4", "S5", "S6"]

SPAN_NAMES: dict[str, str] = {
    "S1": "task_description",
    "S2": "input_schema",
    "S3": "strategy_bullets",
    "S4": "format_rules",
    "S5": "failure_modes_to_avoid",
    "S6": "output_template",
}


# Re-export the canonical schema from scalpel.llm.client so that callers can
# do ``from scalpel.edits.grammar import EDIT_LIST_SCHEMA``.  The schema is
# defined exactly once — in ``scalpel/llm/client.py`` — to keep a single
# source of truth and avoid the circular import that would arise if the
# client tried to import this module.
EDIT_LIST_SCHEMA = _CANONICAL_EDIT_LIST_SCHEMA


class Span(BaseModel):
    """One addressable span (S1..S6) of a structured prompt."""

    model_config = ConfigDict(frozen=True)

    id: SpanId
    name: str
    content: str


class StructuredPrompt(BaseModel):
    """A prompt parsed into the six canonical addressable spans.

    ``raw_text`` is the cached materialization (use
    :func:`scalpel.edits.span_index.materialize` to compute it); ``token_count``
    is filled by :mod:`scalpel.edits.apply` via a simple word-count
    heuristic — a real tokenizer can be plugged in later if Phase 7's
    surrogate needs it.
    """

    model_config = ConfigDict(frozen=True)

    spans: list[Span]
    raw_text: str
    token_count: int = 0


class Edit(BaseModel):
    """One diff edit operation against an addressable span."""

    operation: Literal["REPLACE", "APPEND", "DELETE", "INSERT"]
    target_span: SpanId
    target_line: Optional[int] = None
    content: str = ""


class EditList(BaseModel):
    """A reflector proposal: a list of edits plus a list of lessons.

    Both lists are capped at 4 entries to match the SCALPEL §5.4 schema.
    """

    edits: list[Edit] = Field(default_factory=list, max_length=4)
    lessons: list[str] = Field(default_factory=list, max_length=4)
