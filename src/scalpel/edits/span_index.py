"""SCALPEL span parsing and materialization.

Phase 3 of SCALPEL.  Implements the two-path parser described in
``docs/scalpel/SCALPEL.md`` §5.3:

1. **Tagged form** — XML-like ``<S\\d name="...">...</S\\d>`` blocks. The
   round-trip invariant ``materialize(parse(s)).strip() == s.strip()``
   holds byte-exactly for any well-formed tagged input.
2. **Heuristic form** — header-anchored scan over canonical phrases
   (``Task:``, ``Strategy:``, etc.). Round-trip is best-effort: a
   ``logging.warning`` fires if the diff exceeds 5%.

Public surface: :func:`parse` and :func:`materialize`.
"""

from __future__ import annotations

import logging
import re
from typing import cast

from scalpel.edits.grammar import SPAN_IDS, SPAN_NAMES, Span, SpanId, StructuredPrompt

__all__ = ["materialize", "parse"]


logger = logging.getLogger(__name__)


# Tagged-form regex.  Captures (id_digit, name, content).  ``re.DOTALL`` lets
# ``.*?`` span newlines, which is required because span content is
# multi-line in practice.
TAG_RE = re.compile(r'<S(\d) name="([^"]+)">(.*?)</S\1>', re.DOTALL)


# Heuristic headers per span, listed in priority order.  Matched
# case-insensitively and only at the start of a line (after optional
# whitespace).  Locked by tests; do not reorder without updating fixtures.
HEURISTIC_HEADERS: dict[str, list[str]] = {
    "S1": ["task:", "you are", "role:"],
    "S2": ["input:", "inputs:", "given:"],
    "S3": ["strategy:", "approach:", "steps:", "reasoning:"],
    "S4": ["output format:", "format:", "return:"],
    "S5": ["avoid:", "do not:", "common mistakes:"],
    "S6": ["output template:", "template:", "example output:"],
}


def _empty_spans() -> dict[str, Span]:
    """Build a dict of six empty spans keyed by id, in S1..S6 order."""
    return {sid: Span(id=cast(SpanId, sid), name=SPAN_NAMES[sid], content="") for sid in SPAN_IDS}


def _detect_header(line: str) -> str | None:
    """Return the span id whose header matches ``line``, or ``None``.

    Match is case-insensitive and anchored at the start of the line
    (leading whitespace is allowed and ignored).
    """
    stripped = line.lstrip().lower()
    if not stripped:
        return None
    for sid, headers in HEURISTIC_HEADERS.items():
        for h in headers:
            if stripped.startswith(h):
                return sid
    return None


def _parse_tagged(raw: str) -> StructuredPrompt:
    """Parse a tagged input via :data:`TAG_RE`.

    Spans not present in the input are filled in as empty placeholders.
    Output spans are always emitted in canonical S1..S6 order so that
    downstream materialization is deterministic.
    """
    spans = _empty_spans()
    for match in TAG_RE.finditer(raw):
        digit, name, content = match.group(1), match.group(2), match.group(3)
        sid = f"S{digit}"
        if sid not in SPAN_IDS:
            # Defensive: the regex restricts to one digit, but catch any S0/S7.
            continue
        spans[sid] = Span(id=cast(SpanId, sid), name=name, content=content)

    materialized = "\n".join(
        f'<{s.id} name="{s.name}">{s.content}</{s.id}>' for s in spans.values()
    )
    return StructuredPrompt(spans=list(spans.values()), raw_text=materialized, token_count=0)


def _parse_heuristic(raw: str) -> StructuredPrompt:
    """Parse a header-anchored input.

    The first detected header begins its span; preceding lines accumulate
    into S1 (``task_description``).  Subsequent headers close the previous
    span and open a new one.  If the same span id is hit twice, the
    second occurrence is appended to the first (joined with a newline).
    """
    spans = _empty_spans()
    # Start by collecting any preamble into S1.
    current_sid = "S1"
    buffers: dict[str, list[str]] = {sid: [] for sid in SPAN_IDS}

    for line in raw.split("\n"):
        sid = _detect_header(line)
        if sid is not None:
            current_sid = sid
            stripped = line.lstrip()
            # Colon-form headers ("Task: do X", "Strategy:") drop the prefix
            # and keep any trailing content as the first body line.  Bare
            # forms like "You are a helpful assistant." keep the whole line.
            colon_idx = stripped.find(":")
            looks_colon_form = (
                colon_idx != -1
                and colon_idx <= 30  # cheap "is this a header colon" guard
                and stripped[:colon_idx].lower() + ":" in HEURISTIC_HEADERS[sid]
            )
            if looks_colon_form:
                rest = stripped[colon_idx + 1 :].lstrip()
                if rest:
                    buffers[current_sid].append(rest)
            else:
                buffers[current_sid].append(line)
            continue
        buffers[current_sid].append(line)

    for sid in SPAN_IDS:
        content = "\n".join(buffers[sid]).strip("\n")
        spans[sid] = Span(id=cast(SpanId, sid), name=SPAN_NAMES[sid], content=content)

    parsed = StructuredPrompt(
        spans=list(spans.values()),
        raw_text="",
        token_count=0,
    )
    materialized = materialize(parsed)
    parsed = parsed.model_copy(update={"raw_text": materialized})

    # Round-trip diff check (length-fraction proxy as documented in §5.3).
    a, b = materialized.strip(), raw.strip()
    diff_frac = abs(len(a) - len(b)) / max(len(b), 1)
    if diff_frac > 0.05:
        prefix = raw.strip()[:60].replace("\n", " ")
        logger.warning(
            "heuristic round-trip diff > 5%% on input prefix '%s' (diff=%.3f)",
            prefix,
            diff_frac,
        )
    return parsed


def parse(raw: str) -> StructuredPrompt:
    """Parse ``raw`` into a :class:`StructuredPrompt`.

    Selects the tagged path if any ``<S\\d ...>`` opening tag is present,
    otherwise falls back to the heuristic header scan.
    """
    if re.search(r"<S\d ", raw):
        return _parse_tagged(raw)
    return _parse_heuristic(raw)


def materialize(prompt: StructuredPrompt) -> str:
    """Serialize a :class:`StructuredPrompt` back to a tagged string.

    Always emits all six spans in canonical S1..S6 order, separated by a
    single newline.  Empty spans are still emitted (as empty tags) so
    that the round-trip is byte-exact.
    """
    by_id = {s.id: s for s in prompt.spans}
    lines: list[str] = []
    for sid in SPAN_IDS:
        s = by_id.get(sid)
        if s is None:
            s = Span(id=cast(SpanId, sid), name=SPAN_NAMES[sid], content="")
        lines.append(f'<{s.id} name="{s.name}">{s.content}</{s.id}>')
    return "\n".join(lines)
