"""Phase 1 tests for ``scalpel.llm.client.LiteLLMClient``.

Mocked tests run with ``litellm.completion`` patched out via
``unittest.mock.patch``; the live test (marked ``@pytest.mark.live``) is
skipped unless ``SCALPEL_LIVE_TESTS=1`` is set in the environment (handled by
``tests/scalpel/conftest.py``).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from iso_harness.experiment.context import run_context
from iso_harness.experiment.logging_lm import LoggingLM
from iso_harness.experiment.schemas import RolloutRecord
from scalpel.llm.client import (
    EDIT_LIST_SCHEMA,
    LiteLLMClient,
    Usage,
    wrap_with_logging,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _fake_response(
    content: str = "ok",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    """Build a mock object that mirrors the litellm response surface."""
    message = SimpleNamespace(content=content, reasoning_content=None, reasoning=None)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return SimpleNamespace(choices=[choice], usage=usage)


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #


def test_task_call_payload_shape():
    client = LiteLLMClient()
    with patch("scalpel.llm.client.litellm.completion") as mock_completion:
        mock_completion.return_value = _fake_response()
        result = client("hello")

    assert result == "ok"
    mock_completion.assert_called_once()
    kwargs = mock_completion.call_args.kwargs
    assert kwargs["model"] == "hosted_vllm/openai/gpt-oss-120b"
    assert kwargs["temperature"] == 0.6
    assert kwargs["api_base"] == client.api_base
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 20,
    }


def test_reflect_call_with_schema():
    client = LiteLLMClient()
    schema = {"type": "object"}
    with patch("scalpel.llm.client.litellm.completion") as mock_completion:
        mock_completion.return_value = _fake_response(content='{"x":1}')
        result = client.reflect("explain", guided_json_schema=schema)

    assert result == '{"x":1}'
    kwargs = mock_completion.call_args.kwargs
    extra_body = kwargs["extra_body"]
    assert extra_body["guided_json"] == schema
    assert extra_body["guided_decoding_backend"] == "xgrammar"
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": False}
    # Reflect uses its own temperature.
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == client.max_tokens_reflect


def test_xgrammar_fallback_to_guidance():
    client = LiteLLMClient()
    schema = {"type": "object"}

    with patch("scalpel.llm.client.litellm.completion") as mock_completion:
        mock_completion.side_effect = [
            Exception("xgrammar parse error"),
            _fake_response(content='{"y":2}'),
        ]
        result = client.reflect("go", guided_json_schema=schema)

    assert result == '{"y":2}'
    assert mock_completion.call_count == 2
    second_kwargs = mock_completion.call_args_list[1].kwargs
    assert second_kwargs["extra_body"]["guided_decoding_backend"] == "guidance"
    # First attempt should still have used xgrammar.
    first_kwargs = mock_completion.call_args_list[0].kwargs
    assert first_kwargs["extra_body"]["guided_decoding_backend"] == "xgrammar"


def test_xgrammar_and_guidance_both_fail_reraises_original():
    client = LiteLLMClient()
    schema = {"type": "object"}

    with patch("scalpel.llm.client.litellm.completion") as mock_completion:
        mock_completion.side_effect = [
            Exception("xgrammar parse error"),
            Exception("guidance also broken"),
        ]
        with pytest.raises(Exception, match="xgrammar"):
            client.reflect("go", guided_json_schema=schema)
    assert mock_completion.call_count == 2


def test_think_blocks_stripped():
    client = LiteLLMClient()
    with patch("scalpel.llm.client.litellm.completion") as mock_completion:
        mock_completion.return_value = _fake_response(
            content="<think>scratch</think>final answer"
        )
        result = client("q")
    assert result == "final answer"


def test_last_usage_updated():
    client = LiteLLMClient()
    with patch("scalpel.llm.client.litellm.completion") as mock_completion:
        mock_completion.return_value = _fake_response(
            prompt_tokens=10, completion_tokens=5
        )
        client("hi")
    assert client._last_usage == Usage(10, 5)


def test_hardware_profile_runpod_raises():
    with pytest.raises(NotImplementedError):
        LiteLLMClient(hardware_profile="runpod")


def test_wrap_with_logging_returns_logging_lm():
    client = LiteLLMClient()
    writer = MagicMock()
    wrapped = wrap_with_logging(client, writer, "task")
    assert isinstance(wrapped, LoggingLM)


def test_loggingLM_records_rollout():
    client = LiteLLMClient()
    writer = MagicMock()
    wrapped = wrap_with_logging(client, writer, "task")

    with patch("scalpel.llm.client.litellm.completion") as mock_completion:
        mock_completion.return_value = _fake_response(
            content="answer", prompt_tokens=10, completion_tokens=5
        )
        with run_context(run_id="r1", round_num=1, candidate_id="c1"):
            wrapped("question?")

    assert writer.append.call_count == 1
    record = writer.append.call_args.args[0]
    assert isinstance(record, RolloutRecord)
    assert record.tokens_input == 10
    assert record.tokens_output == 5
    assert record.run_id == "r1"
    assert record.round_num == 1
    assert record.candidate_id == "c1"


def test_edit_list_schema_shape():
    """Sanity check the module-level constant matches SCALPEL.md section 5.4."""
    assert EDIT_LIST_SCHEMA["type"] == "object"
    assert EDIT_LIST_SCHEMA["required"] == ["edits", "lessons"]
    assert EDIT_LIST_SCHEMA["properties"]["edits"]["maxItems"] == 4
    assert EDIT_LIST_SCHEMA["properties"]["lessons"]["maxItems"] == 4
    edit_props = EDIT_LIST_SCHEMA["properties"]["edits"]["items"]["properties"]
    assert edit_props["operation"]["enum"] == ["REPLACE", "APPEND", "DELETE", "INSERT"]
    assert edit_props["target_span"]["enum"] == ["S1", "S2", "S3", "S4", "S5", "S6"]


# --------------------------------------------------------------------------- #
# Live test — gated by SCALPEL_LIVE_TESTS=1 (handled in conftest.py)
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_task_and_reflect():
    client = LiteLLMClient()
    resp = client("Say the word 'hello'.")
    assert isinstance(resp, str) and len(resp) > 0

    schema = {
        "type": "object",
        "required": ["greeting"],
        "properties": {"greeting": {"type": "string"}},
    }
    resp = client.reflect("Return JSON {greeting: hello}", guided_json_schema=schema)
    parsed = json.loads(resp)
    assert "greeting" in parsed
