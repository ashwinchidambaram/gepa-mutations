"""Tests for LoggingLM wrapper.

Tests cover:
- Basic call: wraps inner LM and returns its response
- JSONL output: records contain expected fields and values
- Context vars flow: run_id, round_num, candidate_id propagate to records
- Reflection role: produces ReflectionRecord with correct fields
- Multiple calls: each call appends a unique record
- Token extraction from TrackedLM wrapping: walks the chain to find _last_usage
- Preserves LM interface: model property proxied from inner LM
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from iso_harness.experiment.context import run_context
from iso_harness.experiment.jsonl_writer import JSONLWriter
from iso_harness.experiment.logging_lm import LoggingLM


# ---------------------------------------------------------------------------
# Mock LM — simulates LM interface without real inference
# ---------------------------------------------------------------------------


class MockLM:
    """Fake LM that returns a fixed response and simulates token usage."""

    def __init__(self, response: str = "mock response") -> None:
        self._response = response
        self._last_usage: object | None = None
        self.model = "mock-model"
        self.call_count = 0

    def __call__(self, prompt: str | list) -> str:
        self.call_count += 1

        class Usage:
            prompt_tokens = 10
            completion_tokens = 5

        self._last_usage = Usage()
        return self._response


class MockTrackedLM:
    """Simulates TrackedLM wrapping: stores inner LM as _lm, has no _last_usage."""

    def __init__(self, lm: MockLM) -> None:
        self._lm = lm
        self.model = lm.model

    def __call__(self, prompt: str | list) -> str:
        return self._lm(prompt)


# ======================================================================
# a) Basic call — return value matches inner LM
# ======================================================================


class TestBasicCall:
    """LoggingLM wraps MockLM and returns its response."""

    def test_returns_inner_lm_response(self, tmp_path: Path) -> None:
        mock = MockLM(response="the answer is 42")
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer)

        result = logging_lm("What is the answer?")

        assert result == "the answer is 42"

    def test_inner_lm_is_called(self, tmp_path: Path) -> None:
        mock = MockLM()
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer)

        logging_lm("test prompt")

        assert mock.call_count == 1


# ======================================================================
# b) JSONL output — record has expected fields
# ======================================================================


class TestJSONLOutput:
    """After one call, writer.read_all() returns 1 record with expected fields."""

    def test_single_record_written(self, tmp_path: Path) -> None:
        mock = MockLM(response="hello world")
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer)

        logging_lm("test prompt")

        records = writer.read_all()
        assert len(records) == 1

    def test_record_fields(self, tmp_path: Path) -> None:
        mock = MockLM(response="hello world")
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer)

        logging_lm("test prompt")

        rec = writer.read_all()[0]

        # rollout_id is a valid UUID4
        uuid.UUID(rec["rollout_id"])  # Raises on invalid UUID

        # Token counts from MockLM
        assert rec["tokens_input"] == 10
        assert rec["tokens_output"] == 5

        # Latency is positive
        assert rec["latency_ms"] > 0

        # Response matches inner LM
        assert rec["response"] == "hello world"

        # Prompt is captured
        assert rec["prompt"] == "test prompt"

        # Score defaults to 0.0 (filled in later by evaluator)
        assert rec["score"] == 0.0

    def test_list_prompt_serialized(self, tmp_path: Path) -> None:
        """List-type prompts are serialized to string."""
        mock = MockLM()
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer)

        prompt = [{"role": "user", "content": "Hi"}]
        logging_lm(prompt)

        rec = writer.read_all()[0]
        assert "role" in rec["prompt"]
        assert "user" in rec["prompt"]


# ======================================================================
# c) Context vars flow — run_id, round_num, candidate_id
# ======================================================================


class TestContextVarsFlow:
    """Context variables are captured in the JSONL record."""

    def test_context_vars_in_record(self, tmp_path: Path) -> None:
        mock = MockLM()
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer)

        with run_context(run_id="test-run", round_num=3, candidate_id="cand-1"):
            logging_lm("test prompt")

        rec = writer.read_all()[0]
        assert rec["run_id"] == "test-run"
        assert rec["round_num"] == 3
        assert rec["candidate_id"] == "cand-1"

    def test_default_context_when_unset(self, tmp_path: Path) -> None:
        """Without run_context, fields default to empty/zero."""
        mock = MockLM()
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer)

        logging_lm("test prompt")

        rec = writer.read_all()[0]
        assert rec["run_id"] == ""
        assert rec["round_num"] == 0
        assert rec["candidate_id"] == ""


# ======================================================================
# d) Reflection role — produces ReflectionRecord
# ======================================================================


class TestReflectionRole:
    """LoggingLM with role='reflection' produces ReflectionRecord."""

    def test_reflection_record_fields(self, tmp_path: Path) -> None:
        mock = MockLM(response="improved prompt")
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer, role="reflection")

        with run_context(run_id="refl-run", round_num=2, candidate_id="cand-2"):
            logging_lm("reflect on this")

        rec = writer.read_all()[0]

        # Has reflection-specific fields
        assert "reflection_id" in rec
        uuid.UUID(rec["reflection_id"])  # Valid UUID

        assert rec["triggered_by"] == "mutation"
        assert rec["target_candidate_id"] == "cand-2"
        assert rec["input_prompt"] == "reflect on this"
        assert rec["output"] == "improved prompt"

        # Does NOT have rollout-specific fields
        assert "rollout_id" not in rec
        assert "score" not in rec

    def test_reflection_tokens(self, tmp_path: Path) -> None:
        """Reflection records still capture token usage."""
        mock = MockLM()
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer, role="reflection")

        logging_lm("test")

        rec = writer.read_all()[0]
        assert rec["tokens_input"] == 10
        assert rec["tokens_output"] == 5


# ======================================================================
# e) Multiple calls — each appends a unique record
# ======================================================================


class TestMultipleCalls:
    """Calling LoggingLM N times produces N records with unique IDs."""

    def test_five_calls_five_records(self, tmp_path: Path) -> None:
        mock = MockLM()
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer)

        for _ in range(5):
            logging_lm("prompt")

        records = writer.read_all()
        assert len(records) == 5

        # All rollout_ids are unique
        ids = {rec["rollout_id"] for rec in records}
        assert len(ids) == 5

    def test_inner_lm_called_each_time(self, tmp_path: Path) -> None:
        mock = MockLM()
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer)

        for _ in range(5):
            logging_lm("prompt")

        assert mock.call_count == 5


# ======================================================================
# f) Token extraction from TrackedLM wrapping
# ======================================================================


class TestTrackedLMWrapping:
    """Tokens are extracted by walking the wrapping chain."""

    def test_tokens_via_tracked_lm(self, tmp_path: Path) -> None:
        """MockLM -> MockTrackedLM -> LoggingLM: tokens still extracted."""
        inner = MockLM()
        tracked = MockTrackedLM(inner)
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(tracked, writer)

        logging_lm("test prompt")

        rec = writer.read_all()[0]
        # Tokens come from inner._last_usage, accessed via tracked._lm
        assert rec["tokens_input"] == 10
        assert rec["tokens_output"] == 5

    def test_no_usage_defaults_to_zero(self, tmp_path: Path) -> None:
        """If inner LM has no _last_usage, tokens default to 0."""

        class BareCallable:
            model = "bare"

            def __call__(self, prompt: str) -> str:
                return "bare response"

        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(BareCallable(), writer)

        logging_lm("test")

        rec = writer.read_all()[0]
        assert rec["tokens_input"] == 0
        assert rec["tokens_output"] == 0


# ======================================================================
# g) Preserves LM interface — model property
# ======================================================================


class TestLMInterface:
    """LoggingLM proxies the model property from the inner LM."""

    def test_model_from_inner_lm(self, tmp_path: Path) -> None:
        mock = MockLM()
        mock.model = "qwen3-4b"
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer)

        assert logging_lm.model == "qwen3-4b"

    def test_model_from_tracked_lm(self, tmp_path: Path) -> None:
        inner = MockLM()
        inner.model = "gemma-12b"
        tracked = MockTrackedLM(inner)
        tracked.model = "gemma-12b"
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(tracked, writer)

        assert logging_lm.model == "gemma-12b"

    def test_repr(self, tmp_path: Path) -> None:
        mock = MockLM()
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer)

        r = repr(logging_lm)
        assert "LoggingLM" in r
        assert "task" in r

    def test_example_id_fn(self, tmp_path: Path) -> None:
        """example_id_fn callback is used to tag records."""
        mock = MockLM()
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer, example_id_fn=lambda: "ex-42")

        logging_lm("test")

        rec = writer.read_all()[0]
        assert rec["example_id"] == "ex-42"

    def test_example_id_default(self, tmp_path: Path) -> None:
        """Without example_id_fn, example_id defaults to 'unknown'."""
        mock = MockLM()
        writer = JSONLWriter(tmp_path / "log.jsonl")
        logging_lm = LoggingLM(mock, writer)

        logging_lm("test")

        rec = writer.read_all()[0]
        assert rec["example_id"] == "unknown"
