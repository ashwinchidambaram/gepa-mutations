"""SCALPEL reflection prompt builder.

Phase 4 of SCALPEL.  Assembles the system + user prompts handed to the
reflection LM, per ``docs/scalpel/SCALPEL.md`` §5.4 and the addendum
(Q1: one cluster per iteration; Q3: lessons are context-only and never
auto-promoted into the deployed prompt; Q4: thinking off, no CoT
scaffolding in the reflection prompt).

Public surface:

* :func:`build_reflection_prompt` — return the ``(system, user)`` pair.
* :func:`render_lesson_book` — Phase 8-stub renderer for the Lesson Book.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from scalpel.edits.grammar import StructuredPrompt
from scalpel.edits.span_index import materialize

__all__ = [
    "DEFAULT_MAX_EDITS",
    "DEFAULT_MAX_LESSON_LEN_TOKENS",
    "build_reflection_prompt",
    "render_lesson_book",
]


DEFAULT_MAX_EDITS: int = 4
DEFAULT_MAX_LESSON_LEN_TOKENS: int = 30


_SYSTEM_PROMPT: str = (
    "You are a prompt-improvement engineer. Given a current prompt with "
    "addressable\nspans S1..S6, a single failure-mode cluster (one "
    "representative trace), and a\nLesson Book, propose a SHORT list of "
    "EDIT operations that fix the failures\ndescribed by the cluster.\n\n"
    "You may incorporate any lesson from the Lesson Book by emitting an "
    "APPEND or\nINSERT edit; you are not required to. Lessons that do not "
    "warrant materialization\nshould still inform your choice of edits but "
    "should not be copy-pasted verbatim\nunless they directly address the "
    "cluster.\n\n"
    'Output STRICT JSON matching the provided schema: {"edits": [...], '
    '"lessons": [...]}.\nDo not include prose outside the JSON. Do not '
    "include reasoning steps."
)


def render_lesson_book(lessons: Sequence[Mapping[str, str]]) -> str:
    """Render a list of lesson dicts into the §5.4 Lesson-Book block.

    Each lesson must have keys ``"text"`` and ``"status"`` (``"active"`` or
    ``"negative"``).  Active lessons are emitted first as ``- <text>`` lines;
    negative lessons follow as ``- AVOID: <text>`` lines.  An empty input
    returns an empty string — :func:`build_reflection_prompt` substitutes a
    ``(none yet)`` marker in that case.

    This is a stub for Phase 8's full Lesson Book.
    """
    if not lessons:
        return ""

    active_lines: list[str] = []
    avoid_lines: list[str] = []
    for lesson in lessons:
        text = lesson.get("text", "")
        status = lesson.get("status", "active")
        if status == "negative":
            avoid_lines.append(f"- AVOID: {text}")
        else:
            active_lines.append(f"- {text}")

    return "\n".join(active_lines + avoid_lines)


def build_reflection_prompt(
    *,
    parent_prompt: StructuredPrompt,
    target_module: str,
    cluster_id: int,
    cluster_summary: str,
    representative_trace: str,
    lesson_book_text: str,
    alpha_token_budget: int,
    max_edits: int = DEFAULT_MAX_EDITS,
    max_lesson_tokens: int = DEFAULT_MAX_LESSON_LEN_TOKENS,
) -> tuple[str, str]:
    """Build the ``(system, user)`` reflection prompts for one cluster.

    Per SCALPEL addendum Q1, this builder addresses exactly one failure-mode
    cluster per iteration.  Per Q3, lessons are presented as context only;
    the reflection LM MAY incorporate them via APPEND / INSERT edits but is
    not required to — the deployed prompt is never auto-mutated by lessons.
    Per Q4, the reflection prompt is CoT-free (thinking is disabled at the
    inference layer).

    Args:
        parent_prompt: the candidate :class:`StructuredPrompt` to mutate.
        target_module: name of the candidate's pipeline module (e.g.
            ``"query_writer"``) — included in the user prompt so the LM
            knows which module the prompt drives.
        cluster_id: integer cluster id from Phase 5.
        cluster_summary: <=25-token natural-language cluster description.
        representative_trace: full text of one representative failing trace.
            No truncation is applied — the caller is responsible for
            keeping the budget reasonable.
        lesson_book_text: pre-rendered Lesson-Book block (use
            :func:`render_lesson_book`).  Empty / whitespace-only renders
            as ``(none yet)``.
        alpha_token_budget: ``int(alpha * parent_token_count)`` — passed
            through to the constraints section so the LM stays under the
            §3.A length cap.
        max_edits: cap on the ``edits`` array; mirrored in the constraints
            section and re-enforced by :func:`scalpel.reflection.parser.
            parse_reflection_response`.
        max_lesson_tokens: cap on each ``lessons`` entry's length, in
            tokens.  Communicated to the LM in the constraints section.

    Returns:
        ``(system_prompt, user_prompt)``: two strings ready for
        :meth:`scalpel.llm.client.LiteLLMClient.reflect`.
    """
    book_block = (
        lesson_book_text
        if lesson_book_text and lesson_book_text.strip()
        else "(none yet)"
    )
    rendered_parent = materialize(parent_prompt)

    user_sections = [
        f"=== Lesson Book (active, then AVOID) ===\n{book_block}",
        (
            f"=== Current Prompt (target_module={target_module}) ===\n"
            f"{rendered_parent}"
        ),
        (
            "=== Failure Cluster ===\n"
            f"Cluster ID: {cluster_id}\n"
            f"Cluster summary: {cluster_summary}\n"
            "Representative trace:\n"
            f"{representative_trace}"
        ),
        (
            "=== Constraints ===\n"
            f"- Output at most {max_edits} edits.\n"
            f"- Total new content must add ≤ {alpha_token_budget} tokens.\n"
            "- Edits must address the failure mode described above.\n"
            f"- Lessons must be ≤ {max_lesson_tokens} tokens each.\n"
            "- Do NOT emit a full prompt rewrite; SCALPEL's diff representation "
            "depends on\n  surgical, span-targeted edits."
        ),
    ]

    user_prompt = "\n\n".join(user_sections)
    return _SYSTEM_PROMPT, user_prompt
