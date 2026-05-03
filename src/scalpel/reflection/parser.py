"""SCALPEL reflection-response parser.

Phase 4 of SCALPEL.  Parses the JSON edit-list emitted by the reflection
LM into a validated :class:`scalpel.edits.grammar.EditList`, applying a
graduated fallback strategy (strict JSON -> embedded JSON -> per-edit
partial validation -> hard failure).

Public surface: :func:`parse_reflection_response`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from scalpel.edits.grammar import SPAN_IDS, Edit, EditList

__all__ = [
    "DEFAULT_MAX_EDITS",
    "DEFAULT_MAX_LESSONS",
    "parse_reflection_response",
]


DEFAULT_MAX_EDITS: int = 4
DEFAULT_MAX_LESSONS: int = 4

_EMBEDDED_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _validate_edits_per_item(
    raw_edits: list[Any],
    errors: list[str],
) -> list[Edit]:
    """Validate edits one at a time, recording per-item errors.

    Drops any edit whose ``target_span`` is not in :data:`SPAN_IDS` or
    that fails pydantic validation; appends a human-readable error string
    to ``errors`` for each drop.
    """
    valid: list[Edit] = []
    for idx, item in enumerate(raw_edits):
        if not isinstance(item, dict):
            errors.append(f"edit #{idx}: not an object ({type(item).__name__})")
            continue
        target_span = item.get("target_span")
        if target_span not in SPAN_IDS:
            errors.append(
                f"edit #{idx}: invalid target_span {target_span!r} "
                f"(must be one of {SPAN_IDS})"
            )
            continue
        try:
            valid.append(Edit.model_validate(item))
        except ValidationError as exc:
            errors.append(f"edit #{idx}: {exc.errors()[0].get('msg', str(exc))}")
    return valid


def _validate_lessons_per_item(
    raw_lessons: list[Any],
    errors: list[str],
) -> list[str]:
    """Filter lessons to non-empty strings, recording drops."""
    valid: list[str] = []
    for idx, item in enumerate(raw_lessons):
        if not isinstance(item, str):
            errors.append(f"lesson #{idx}: not a string ({type(item).__name__})")
            continue
        valid.append(item)
    return valid


def _truncate_with_warning(
    items: list[Any],
    cap: int,
    errors: list[str],
    label: str,
) -> list[Any]:
    """If ``items`` exceeds ``cap``, drop the tail and warn into ``errors``."""
    if len(items) > cap:
        errors.append(
            f"truncated {label}: kept {cap} of {len(items)} (excess dropped)"
        )
        return items[:cap]
    return items


def _build_from_payload(
    payload: dict[str, Any],
    errors: list[str],
) -> EditList:
    """Build an :class:`EditList` from a parsed JSON object.

    Tries strict whole-object validation first; falls back to per-item
    validation on failure.  Always enforces the ``max_edits`` /
    ``max_lessons`` caps, with truncation logged into ``errors``.
    """
    raw_edits = payload.get("edits", []) or []
    raw_lessons = payload.get("lessons", []) or []

    if not isinstance(raw_edits, list):
        errors.append(f"'edits' must be a list, got {type(raw_edits).__name__}")
        raw_edits = []
    if not isinstance(raw_lessons, list):
        errors.append(f"'lessons' must be a list, got {type(raw_lessons).__name__}")
        raw_lessons = []

    raw_edits = _truncate_with_warning(
        raw_edits, DEFAULT_MAX_EDITS, errors, "edits"
    )
    raw_lessons = _truncate_with_warning(
        raw_lessons, DEFAULT_MAX_LESSONS, errors, "lessons"
    )

    # Strict path — every edit valid AND every span id in SPAN_IDS.
    spans_all_ok = all(
        isinstance(e, dict) and e.get("target_span") in SPAN_IDS for e in raw_edits
    )
    if spans_all_ok:
        try:
            return EditList.model_validate(
                {"edits": raw_edits, "lessons": raw_lessons}
            )
        except ValidationError:
            pass  # fall through to per-item

    valid_edits = _validate_edits_per_item(raw_edits, errors)
    valid_lessons = _validate_lessons_per_item(raw_lessons, errors)
    try:
        return EditList.model_validate(
            {"edits": valid_edits, "lessons": valid_lessons}
        )
    except ValidationError as exc:
        errors.append(f"final EditList validation failed: {exc.errors()[0].get('msg', str(exc))}")
        return EditList()


def parse_reflection_response(raw: str) -> tuple[EditList, list[str]]:
    """Parse a reflection LM response into ``(edit_list, errors)``.

    Strategy (per Phase 4 spec):

    1. **Strict JSON** — ``json.loads(raw)`` succeeds and validates as an
       :class:`EditList`.  Returns ``(edits, [])``.
    2. **Embedded JSON** — extract the first ``{...}`` substring with a
       small regex and try again.  On success, ``errors`` records the
       wrapper.
    3. **Per-item** — JSON is parseable but at least one edit fails
       validation.  Skip individual bad edits, keep the rest, record
       drops in ``errors``.
    4. **Hard failure** — no parseable JSON.  Returns
       ``(EditList(), ["no parseable JSON"])``.

    Args:
        raw: the raw text emitted by the reflection LM.

    Returns:
        ``(edit_list, errors)``: a validated :class:`EditList` (possibly
        empty) and a list of human-readable error / warning strings
        (empty when parsing was clean).
    """
    errors: list[str] = []

    # 1. Strict JSON path.
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return _build_from_payload(payload, errors), errors
        errors.append(f"top-level JSON is not an object ({type(payload).__name__})")
    except json.JSONDecodeError:
        pass

    # 2. Embedded JSON path.
    match = _EMBEDDED_JSON_RE.search(raw)
    if match is not None:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            errors.append("raw response had non-JSON wrapper")
            return _build_from_payload(payload, errors), errors

    # 3 & 4. Hard failure.
    errors.append("no parseable JSON")
    return EditList(), errors
