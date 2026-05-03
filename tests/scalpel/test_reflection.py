"""Tests for SCALPEL Phase 4 — reflection prompt builder + parser."""

from __future__ import annotations

import json

import pytest

from scalpel.edits.apply import apply
from scalpel.edits.grammar import Edit, EditList, Span, StructuredPrompt
from scalpel.edits.span_index import materialize
from scalpel.reflection.parser import parse_reflection_response
from scalpel.reflection.prompt_builder import (
    build_reflection_prompt,
    render_lesson_book,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _sample_prompt() -> StructuredPrompt:
    spans = [
        Span(id="S1", name="task_description", content="Answer the question."),
        Span(id="S2", name="input_schema", content="question: str"),
        Span(
            id="S3",
            name="strategy_bullets",
            content="- Decompose the question into sub-queries.",
        ),
        Span(id="S4", name="format_rules", content="Return only the answer."),
        Span(id="S5", name="failure_modes_to_avoid", content="Do not hallucinate."),
        Span(id="S6", name="output_template", content="Answer: <answer>"),
    ]
    sp = StructuredPrompt(spans=spans, raw_text="", token_count=0)
    return sp.model_copy(update={"raw_text": materialize(sp)})


# --------------------------------------------------------------------------- #
# prompt_builder
# --------------------------------------------------------------------------- #


def test_build_prompt_returns_system_and_user_strings() -> None:
    sp = _sample_prompt()
    sys_p, user_p = build_reflection_prompt(
        parent_prompt=sp,
        target_module="query_writer",
        cluster_id=3,
        cluster_summary="missing decomposition for multi-hop questions",
        representative_trace="Q: foo? A: bar (wrong)",
        lesson_book_text="- prefer decomposition",
        alpha_token_budget=10,
    )
    assert isinstance(sys_p, str) and isinstance(user_p, str)
    assert "prompt-improvement engineer" in sys_p
    assert "may incorporate any lesson" in sys_p

    # Sections present, in order.
    pos_book = user_p.find("=== Lesson Book")
    pos_prompt = user_p.find("=== Current Prompt")
    pos_cluster = user_p.find("=== Failure Cluster")
    pos_constraints = user_p.find("=== Constraints")
    assert -1 < pos_book < pos_prompt < pos_cluster < pos_constraints


def test_user_prompt_renders_parent_via_materialize() -> None:
    sp = _sample_prompt()
    _, user_p = build_reflection_prompt(
        parent_prompt=sp,
        target_module="query_writer",
        cluster_id=0,
        cluster_summary="x",
        representative_trace="t",
        lesson_book_text="",
        alpha_token_budget=5,
    )
    # S3 content from the fixture must appear in the rendered prompt section.
    assert "Decompose the question into sub-queries" in user_p
    # Make sure it lives in the Current Prompt section, not somewhere else.
    cp_section = user_p.split("=== Current Prompt")[1].split("=== Failure Cluster")[0]
    assert "Decompose the question into sub-queries" in cp_section


def test_lesson_book_section_handles_empty() -> None:
    sp = _sample_prompt()
    _, user_p = build_reflection_prompt(
        parent_prompt=sp,
        target_module="m",
        cluster_id=0,
        cluster_summary="x",
        representative_trace="t",
        lesson_book_text="",
        alpha_token_budget=5,
    )
    book_section = user_p.split("=== Lesson Book")[1].split("=== Current Prompt")[0]
    assert "(none yet)" in book_section


def test_lesson_book_section_handles_whitespace_only() -> None:
    sp = _sample_prompt()
    _, user_p = build_reflection_prompt(
        parent_prompt=sp,
        target_module="m",
        cluster_id=0,
        cluster_summary="x",
        representative_trace="t",
        lesson_book_text="   \n\t",
        alpha_token_budget=5,
    )
    book_section = user_p.split("=== Lesson Book")[1].split("=== Current Prompt")[0]
    assert "(none yet)" in book_section


def test_constraints_section_includes_alpha_budget() -> None:
    sp = _sample_prompt()
    _, user_p = build_reflection_prompt(
        parent_prompt=sp,
        target_module="m",
        cluster_id=0,
        cluster_summary="x",
        representative_trace="t",
        lesson_book_text="",
        alpha_token_budget=12,
    )
    constraints = user_p.split("=== Constraints")[1]
    assert "≤ 12 tokens" in constraints


def test_target_module_in_user_prompt() -> None:
    sp = _sample_prompt()
    _, user_p = build_reflection_prompt(
        parent_prompt=sp,
        target_module="query_writer",
        cluster_id=0,
        cluster_summary="x",
        representative_trace="t",
        lesson_book_text="",
        alpha_token_budget=5,
    )
    assert "query_writer" in user_p


def test_render_lesson_book_active_then_avoid() -> None:
    rendered = render_lesson_book(
        [
            {"text": "a", "status": "active"},
            {"text": "b", "status": "negative"},
            {"text": "c", "status": "active"},
        ]
    )
    pos_a = rendered.find("- a")
    pos_c = rendered.find("- c")
    pos_avoid_b = rendered.find("AVOID: b")
    assert pos_a != -1 and pos_c != -1 and pos_avoid_b != -1
    assert pos_a < pos_avoid_b
    assert pos_c < pos_avoid_b


def test_render_lesson_book_empty() -> None:
    assert render_lesson_book([]) == ""


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def _good_payload() -> dict:
    return {
        "edits": [
            {
                "operation": "APPEND",
                "target_span": "S3",
                "content": "- Verify each sub-answer before composing the final answer.",
            },
            {
                "operation": "REPLACE",
                "target_span": "S5",
                "target_line": 1,
                "content": "Avoid hallucinating supporting evidence.",
            },
        ],
        "lessons": ["sub-query decomposition helps multi-hop"],
    }


def test_parse_strict_json_path() -> None:
    raw = json.dumps(_good_payload())
    edit_list, errors = parse_reflection_response(raw)
    assert isinstance(edit_list, EditList)
    assert len(edit_list.edits) == 2
    assert edit_list.edits[0].operation == "APPEND"
    assert edit_list.edits[0].target_span == "S3"
    assert edit_list.lessons == ["sub-query decomposition helps multi-hop"]
    assert errors == []


def test_parse_embedded_json_path() -> None:
    inner = json.dumps(_good_payload())
    raw = f"Here you go: {inner} hope that helps"
    edit_list, errors = parse_reflection_response(raw)
    assert len(edit_list.edits) == 2
    assert any("non-JSON wrapper" in e for e in errors)


def test_parse_invalid_edit_drops_with_warning() -> None:
    payload = {
        "edits": [
            {"operation": "APPEND", "target_span": "S3", "content": "ok edit"},
            {"operation": "APPEND", "target_span": "S99", "content": "bad span"},
        ],
        "lessons": [],
    }
    edit_list, errors = parse_reflection_response(json.dumps(payload))
    assert len(edit_list.edits) == 1
    assert edit_list.edits[0].target_span == "S3"
    assert any("S99" in e for e in errors)


def test_parse_unparseable_returns_empty_with_error() -> None:
    edit_list, errors = parse_reflection_response("not json at all")
    assert edit_list.edits == []
    assert edit_list.lessons == []
    assert len(errors) > 0
    assert any("no parseable JSON" in e for e in errors)


def test_parse_truncates_excess_edits() -> None:
    payload = {
        "edits": [
            {"operation": "APPEND", "target_span": "S3", "content": f"e{i}"}
            for i in range(6)
        ],
        "lessons": [],
    }
    edit_list, errors = parse_reflection_response(json.dumps(payload))
    assert len(edit_list.edits) == 4
    assert any("truncated edits" in e for e in errors)
    # The first 4 are kept.
    assert [e.content for e in edit_list.edits] == ["e0", "e1", "e2", "e3"]


def test_parse_truncates_excess_lessons() -> None:
    payload = {
        "edits": [],
        "lessons": [f"l{i}" for i in range(6)],
    }
    edit_list, errors = parse_reflection_response(json.dumps(payload))
    assert len(edit_list.lessons) == 4
    assert any("truncated lessons" in e for e in errors)
    assert edit_list.lessons == ["l0", "l1", "l2", "l3"]


def test_parse_round_trip_with_real_grammar() -> None:
    original = EditList(
        edits=[
            Edit(operation="APPEND", target_span="S3", content="hello"),
            Edit(operation="DELETE", target_span="S5", target_line=1),
        ],
        lessons=["lesson 1", "lesson 2"],
    )
    raw = original.model_dump_json()
    parsed, errors = parse_reflection_response(raw)
    assert errors == []
    assert parsed == original


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #


def test_prompt_assembled_then_parsed_round_trip() -> None:
    sp = _sample_prompt()
    sys_p, user_p = build_reflection_prompt(
        parent_prompt=sp,
        target_module="query_writer",
        cluster_id=1,
        cluster_summary="missing decomposition",
        representative_trace="trace text",
        lesson_book_text=render_lesson_book(
            [{"text": "decompose multi-hop", "status": "active"}]
        ),
        alpha_token_budget=20,
    )
    assert sys_p and user_p

    # Hand-craft a "response" matching the schema.
    response = json.dumps(
        {
            "edits": [
                {
                    "operation": "APPEND",
                    "target_span": "S3",
                    "content": "- Decompose first.",
                }
            ],
            "lessons": ["decompose first"],
        }
    )
    edit_list, errors = parse_reflection_response(response)
    assert errors == []
    assert len(edit_list.edits) == 1

    # Verify the resulting edits could feed apply() — instantiate, do not require success.
    try:
        apply(sp, edit_list.edits, alpha=0.5)
    except Exception as exc:  # pragma: no cover — defensive
        pytest.fail(f"apply() raised unexpectedly: {exc}")
