"""SCALPEL edit application + length cap.

Phase 3 of SCALPEL.  Applies a list of :class:`scalpel.edits.grammar.Edit`
operations to a :class:`scalpel.edits.grammar.StructuredPrompt`,
deterministically, and enforces the length cap from §3.A:

  ``new_token_count - parent_token_count <= alpha * parent_token_count``

The default ``alpha`` is 0.15 per the SCALPEL doc.  Violations raise
:class:`LengthCapExceeded`.

Public surface:

* :class:`LengthCapExceeded` — raised when an edit list overshoots the cap.
* :func:`apply` — apply a list of edits and return the new prompt.
"""

from __future__ import annotations

from typing import cast

from scalpel.edits.grammar import SPAN_IDS, Edit, Span, SpanId, StructuredPrompt
from scalpel.edits.span_index import materialize

__all__ = ["LengthCapExceeded", "apply"]


class LengthCapExceeded(Exception):
    """Raised when an edit list adds more than ``alpha * |parent|`` tokens."""


def _approx_token_count(s: str) -> int:
    """Whitespace-based token approximation.

    A real tokenizer (e.g. Qwen3) can replace this if Phase 7's surrogate
    needs tighter accounting; until then, word count is sufficient for
    enforcing the §3.A length cap.
    """
    return len(s.split())


def _spans_by_id(prompt: StructuredPrompt) -> dict[str, Span]:
    """Index ``prompt.spans`` by id, filling missing ids with empty spans."""
    from scalpel.edits.grammar import SPAN_NAMES  # local import to keep top minimal

    out: dict[str, Span] = {s.id: s for s in prompt.spans}
    for sid in SPAN_IDS:
        if sid not in out:
            out[sid] = Span(id=cast(SpanId, sid), name=SPAN_NAMES[sid], content="")
    return out


def _apply_one(span: Span, edit: Edit) -> Span:
    """Apply a single edit to a span, returning a new (frozen) Span.

    Operations follow §5.3 of the SCALPEL doc:

    * REPLACE w/o line: replace whole content.
    * REPLACE w/ line: replace 1-indexed line N.
    * APPEND: append ``content`` as a trailing line.
    * DELETE w/o line: empty the span.
    * DELETE w/ line: drop 1-indexed line N.
    * INSERT: insert ``content`` BEFORE 1-indexed line N
      (``target_line is None`` -> ``ValueError``).

    Out-of-range line indices for line-targeted REPLACE / DELETE / INSERT
    raise ``IndexError`` so the optimizer's race code can re-sample
    rather than silently corrupt the prompt.
    """
    lines = span.content.split("\n")
    op = edit.operation

    if op == "REPLACE":
        if edit.target_line is None:
            new_content = edit.content
        else:
            idx = edit.target_line - 1
            if not (0 <= idx < len(lines)):
                raise IndexError(
                    f"REPLACE target_line={edit.target_line} out of range for "
                    f"span {span.id} (has {len(lines)} lines)"
                )
            lines[idx] = edit.content
            new_content = "\n".join(lines)
    elif op == "APPEND":
        if span.content:
            new_content = span.content + "\n" + edit.content
        else:
            new_content = edit.content
    elif op == "DELETE":
        if edit.target_line is None:
            new_content = ""
        else:
            idx = edit.target_line - 1
            if not (0 <= idx < len(lines)):
                raise IndexError(
                    f"DELETE target_line={edit.target_line} out of range for "
                    f"span {span.id} (has {len(lines)} lines)"
                )
            del lines[idx]
            new_content = "\n".join(lines)
    elif op == "INSERT":
        if edit.target_line is None:
            raise ValueError("INSERT requires target_line (cannot be None)")
        idx = edit.target_line - 1
        # INSERT at index 0..len(lines) is valid (len(lines) -> append-style).
        if not (0 <= idx <= len(lines)):
            raise IndexError(
                f"INSERT target_line={edit.target_line} out of range for "
                f"span {span.id} (has {len(lines)} lines)"
            )
        # Special case: a span whose content is the empty string still
        # produces lines == [""] from split("\n"); inserting before line 1
        # in that case should yield exactly the inserted content (not
        # "<content>\n").
        if lines == [""]:
            new_content = edit.content
        else:
            lines.insert(idx, edit.content)
            new_content = "\n".join(lines)
    else:  # pragma: no cover — pydantic Literal forbids other values
        raise ValueError(f"Unknown operation: {op!r}")

    return Span(id=span.id, name=span.name, content=new_content)


def apply(
    prompt: StructuredPrompt,
    edits: list[Edit],
    alpha: float = 0.15,
) -> StructuredPrompt:
    """Apply ``edits`` to ``prompt`` and return a new :class:`StructuredPrompt`.

    Edits are applied **in order**; later edits see the effects of
    earlier ones.  After all edits, the resulting prompt is materialized
    and its token count compared against ``parent`` to enforce the
    length cap (default ``alpha=0.15``).

    Determinism: identical ``(prompt, edits, alpha)`` produces a
    byte-identical output.  No randomness, no time, no set ordering.

    Args:
        prompt: parent :class:`StructuredPrompt`.
        edits: ordered list of :class:`Edit` operations.
        alpha: length-cap fraction; default 0.15 per SCALPEL §3.A.

    Returns:
        A fresh :class:`StructuredPrompt` whose ``raw_text`` is the
        materialization of the post-edit spans and whose ``token_count``
        is the whitespace-approximation token count of that text.

    Raises:
        LengthCapExceeded: when the post-edit token count exceeds the
            allowance ``alpha * parent_token_count``.
        ValueError: when an INSERT edit is missing ``target_line``.
        IndexError: when a line-targeted edit has an out-of-range
            ``target_line``.
    """
    by_id = _spans_by_id(prompt)
    for e in edits:
        sid = e.target_span
        by_id[sid] = _apply_one(by_id[sid], e)

    new_spans = [by_id[sid] for sid in SPAN_IDS]
    new_prompt = StructuredPrompt(spans=new_spans, raw_text="", token_count=0)
    materialized = materialize(new_prompt)
    new_count = _approx_token_count(materialized)

    parent_count = _approx_token_count(materialize(prompt))
    delta = new_count - parent_count
    cap = alpha * parent_count
    if delta > cap:
        raise LengthCapExceeded(
            f"edit list adds {delta} tokens (cap: {cap:.0f}, "
            f"parent={parent_count}, new={new_count})"
        )

    return StructuredPrompt(
        spans=new_spans,
        raw_text=materialized,
        token_count=new_count,
    )
