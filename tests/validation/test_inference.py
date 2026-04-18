"""V4: Basic inference validation — both servers respond with well-formed output."""

from __future__ import annotations

import json
import urllib.request

import pytest


def _chat_completion(port: int, prompt: str, max_tokens: int = 64) -> dict:
    """Send a chat completion request to a vLLM server."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


@pytest.mark.live_server
class TestInference:
    def test_task_basic_response(self, live_vllm_task):
        """Task server responds to basic math question."""
        result = _chat_completion(8000, "What is 2+2? Answer with just the number.")
        content = result["choices"][0]["message"]["content"]
        assert "4" in content, f"Expected '4' in response: {content}"

    def test_task_returns_usage(self, live_vllm_task):
        """Task server returns token usage."""
        result = _chat_completion(8000, "Say hello.")
        assert "usage" in result
        assert result["usage"]["prompt_tokens"] > 0
        assert result["usage"]["completion_tokens"] > 0

    def test_reflection_responds(self, live_vllm_reflection):
        """Reflection server produces non-empty response."""
        result = _chat_completion(8001, "Describe a simple optimization strategy in one sentence.")
        content = result["choices"][0]["message"]["content"]
        assert len(content) > 10, f"Response too short: {content}"
