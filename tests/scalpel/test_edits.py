"""Tests for SCALPEL Phase 3 — addressable-span EDIT grammar.

Covers the parse -> apply -> materialize round-trip invariant, the §3.A
length cap, the pydantic validation surface of the grammar, and a live
acceptance test that exercises xgrammar guided decoding through the
Phase 1 LiteLLM client (gated by ``SCALPEL_LIVE_TESTS=1``).
"""

from __future__ import annotations

import json
import logging
import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from scalpel.edits.apply import LengthCapExceeded, apply
from scalpel.edits.grammar import (
    SPAN_IDS,
    Edit,
    EditList,
    Span,
    StructuredPrompt,
)
from scalpel.edits.span_index import materialize, parse

# --------------------------------------------------------------------------- #
# Fixture corpus
# --------------------------------------------------------------------------- #


# Five hand-written tagged prompts spanning the variability axes the
# grammar must handle: all-spans, empty spans, multi-line content,
# escape-needing characters, and headers with whitespace.
FIXTURE_PROMPTS: list[str] = [
    # 1. All six spans present, single-line content.
    (
        '<S1 name="task_description">Answer the question.</S1>\n'
        '<S2 name="input_schema">question: str</S2>\n'
        '<S3 name="strategy_bullets">Read carefully.</S3>\n'
        '<S4 name="format_rules">Answer in one sentence.</S4>\n'
        '<S5 name="failure_modes_to_avoid">Do not hallucinate.</S5>\n'
        '<S6 name="output_template">Answer: ...</S6>'
    ),
    # 2. Some spans empty.
    (
        '<S1 name="task_description">You are a helpful assistant.</S1>\n'
        '<S2 name="input_schema"></S2>\n'
        '<S3 name="strategy_bullets">Step 1: think.\nStep 2: answer.</S3>\n'
        '<S4 name="format_rules"></S4>\n'
        '<S5 name="failure_modes_to_avoid"></S5>\n'
        '<S6 name="output_template"></S6>'
    ),
    # 3. Multi-line content in several spans.
    (
        '<S1 name="task_description">Classify the claim.</S1>\n'
        '<S2 name="input_schema">claim: str\nevidence: list[str]</S2>\n'
        '<S3 name="strategy_bullets">- Read claim\n- Compare to evidence\n- Decide</S3>\n'
        '<S4 name="format_rules">Output one of: SUPPORTED, REFUTED.</S4>\n'
        '<S5 name="failure_modes_to_avoid">- No fabrication\n- No partial answers</S5>\n'
        '<S6 name="output_template">Verdict: <label></S6>'
    ),
    # 4. Special characters: braces, parens, slashes.  No literal ``</S\d>``
    # sequences inside content (those would close the enclosing tag).
    (
        '<S1 name="task_description">Parse the JSON {key: value} pair.</S1>\n'
        '<S2 name="input_schema">payload: str (JSON)</S2>\n'
        '<S3 name="strategy_bullets">Use json.loads().\nHandle / and \\ properly.</S3>\n'
        '<S4 name="format_rules">Return Python dict repr.</S4>\n'
        '<S5 name="failure_modes_to_avoid">Reject unsafe code paths.</S5>\n'
        '<S6 name="output_template">{result}</S6>'
    ),
    # 5. Trailing whitespace inside spans (preserved by parser/materializer).
    (
        '<S1 name="task_description">Summarize.</S1>\n'
        '<S2 name="input_schema">doc: str</S2>\n'
        '<S3 name="strategy_bullets">Extract key sentences.  </S3>\n'
        '<S4 name="format_rules">Three bullets.</S4>\n'
        '<S5 name="failure_modes_to_avoid">No quoting.</S5>\n'
        '<S6 name="output_template">- ...\n- ...\n- ...</S6>'
    ),
]


# --------------------------------------------------------------------------- #
# Property tests (hypothesis)
# --------------------------------------------------------------------------- #


@given(prompt_str=st.sampled_from(FIXTURE_PROMPTS))
def test_tagged_round_trip_byte_identical(prompt_str: str) -> None:
    parsed = parse(prompt_str)
    assert materialize(parsed).strip() == prompt_str.strip()


@given(prompt_str=st.sampled_from(FIXTURE_PROMPTS))
def test_apply_empty_edits_is_identity(prompt_str: str) -> None:
    parsed = parse(prompt_str)
    out = apply(parsed, [])
    assert [s.model_dump() for s in out.spans] == [s.model_dump() for s in parsed.spans]


_SPAN_ALPHABET = st.text(alphabet="abcdef ", min_size=1, max_size=8)


@settings(max_examples=25, deadline=None)
@given(
    prompt_str=st.sampled_from(FIXTURE_PROMPTS),
    edits=st.lists(
        st.builds(
            Edit,
            operation=st.sampled_from(["REPLACE", "APPEND"]),
            target_span=st.sampled_from(list(SPAN_IDS)),
            target_line=st.none(),
            content=_SPAN_ALPHABET,
        ),
        min_size=1,
        max_size=4,
    ),
)
def test_random_edits_preserve_all_six_spans(prompt_str: str, edits: list[Edit]) -> None:
    parsed = parse(prompt_str)
    try:
        out = apply(parsed, edits, alpha=10.0)  # generous cap so we focus on structure
    except (LengthCapExceeded, IndexError):
        return  # only assert structure preservation when apply succeeds
    ids = [s.id for s in out.spans]
    assert ids == list(SPAN_IDS)


def test_length_cap_fires_on_huge_append() -> None:
    parsed = parse(FIXTURE_PROMPTS[0])
    huge = " ".join(["bloat"] * 5000)
    big_edit = Edit(operation="APPEND", target_span="S3", content=huge)
    with pytest.raises(LengthCapExceeded):
        apply(parsed, [big_edit])


# --------------------------------------------------------------------------- #
# Concrete behaviour tests
# --------------------------------------------------------------------------- #


def _example_prompt() -> StructuredPrompt:
    return parse(FIXTURE_PROMPTS[2])  # multi-line S3, S5


def test_replace_whole_span() -> None:
    p = _example_prompt()
    edits = [Edit(operation="REPLACE", target_span="S4", content="Single sentence only.")]
    out = apply(p, edits)
    s4 = next(s for s in out.spans if s.id == "S4")
    assert s4.content == "Single sentence only."


def test_replace_specific_line() -> None:
    p = _example_prompt()
    # S3 has three lines (- Read claim / - Compare to evidence / - Decide).
    edits = [
        Edit(operation="REPLACE", target_span="S3", target_line=2, content="- Inspect each piece"),
    ]
    out = apply(p, edits)
    s3 = next(s for s in out.spans if s.id == "S3")
    assert s3.content.split("\n") == ["- Read claim", "- Inspect each piece", "- Decide"]


def test_append_adds_newline() -> None:
    p = _example_prompt()
    edits = [Edit(operation="APPEND", target_span="S3", content="- Verify")]
    out = apply(p, edits)
    s3 = next(s for s in out.spans if s.id == "S3")
    assert s3.content.endswith("\n- Verify")


def test_delete_whole_span() -> None:
    p = _example_prompt()
    edits = [Edit(operation="DELETE", target_span="S5")]
    out = apply(p, edits)
    s5 = next(s for s in out.spans if s.id == "S5")
    assert s5.content == ""


def test_delete_specific_line() -> None:
    p = _example_prompt()
    edits = [Edit(operation="DELETE", target_span="S3", target_line=1)]
    out = apply(p, edits)
    s3 = next(s for s in out.spans if s.id == "S3")
    assert s3.content.split("\n") == ["- Compare to evidence", "- Decide"]


def test_insert_requires_target_line() -> None:
    p = _example_prompt()
    edits = [Edit(operation="INSERT", target_span="S3", content="- New step")]
    with pytest.raises(ValueError):
        apply(p, edits)


def test_insert_at_line_1() -> None:
    p = _example_prompt()
    edits = [
        Edit(operation="INSERT", target_span="S3", target_line=1, content="- Re-read prompt"),
    ]
    out = apply(p, edits)
    s3 = next(s for s in out.spans if s.id == "S3")
    assert s3.content.split("\n") == [
        "- Re-read prompt",
        "- Read claim",
        "- Compare to evidence",
        "- Decide",
    ]


def test_unknown_target_span_caught_by_pydantic() -> None:
    with pytest.raises(ValidationError):
        Edit(operation="REPLACE", target_span="S99", content="x")  # type: ignore[arg-type]


def test_heuristic_parse_warns_on_diff_over_5pct(caplog: pytest.LogCaptureFixture) -> None:
    headered = (
        "Task: classify the input.\n"
        "Strategy: read carefully then decide.\n"
        "Output format: one word.\n"
    )
    with caplog.at_level(logging.WARNING, logger="scalpel.edits.span_index"):
        parsed = parse(headered)
    assert isinstance(parsed, StructuredPrompt)
    # The materialized tagged form is much longer than the bare-headered
    # input, so the diff fraction must exceed 5% and the warning must fire.
    assert any("heuristic round-trip diff" in rec.message for rec in caplog.records)


def test_supported_span_ids_constant() -> None:
    assert SPAN_IDS == ("S1", "S2", "S3", "S4", "S5", "S6")


def test_pydantic_models_are_frozen() -> None:
    s = Span(id="S1", name="task_description", content="hi")
    with pytest.raises(ValidationError):
        s.content = "different"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Live grammar acceptance test (gated by SCALPEL_LIVE_TESTS=1)
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_xgrammar_accepts_edit_list_schema() -> None:
    if os.environ.get("SCALPEL_LIVE_TESTS") != "1":  # pragma: no cover
        pytest.skip("live tests disabled")

    from scalpel.edits.grammar import EDIT_LIST_SCHEMA
    from scalpel.llm.client import LiteLLMClient

    client = LiteLLMClient()
    raw = client.reflect(
        "Given a prompt with span S3 containing a step list, propose a single REPLACE "
        "edit that adds a new bullet about checking units. Output the JSON schema-compliant "
        "edit list now.",
        guided_json_schema=EDIT_LIST_SCHEMA,
    )
    parsed = EditList.model_validate(json.loads(raw))
    assert 0 <= len(parsed.edits) <= 4
    if parsed.edits:
        assert parsed.edits[0].target_span in ("S1", "S2", "S3", "S4", "S5", "S6")
