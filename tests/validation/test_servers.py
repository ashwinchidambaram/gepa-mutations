"""V3: vLLM server launch validation — health checks on both ports."""

from __future__ import annotations

import json
import urllib.request

import pytest


@pytest.mark.live_server
class TestServers:
    def test_task_server_health(self, live_vllm_task):
        """Task server responds to /health."""
        resp = urllib.request.urlopen("http://localhost:8000/health", timeout=5)
        assert resp.status == 200

    def test_reflection_server_health(self, live_vllm_reflection):
        """Reflection server responds to /health."""
        resp = urllib.request.urlopen("http://localhost:8001/health", timeout=5)
        assert resp.status == 200

    def test_task_model_listed(self, live_vllm_task):
        """Task server lists expected model."""
        resp = urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
        data = json.loads(resp.read())
        model_ids = [m["id"] for m in data.get("data", [])]
        assert any("Qwen3-8B" in m for m in model_ids), f"Expected Qwen3-8B in {model_ids}"

    def test_reflection_model_listed(self, live_vllm_reflection):
        """Reflection server lists expected model."""
        resp = urllib.request.urlopen("http://localhost:8001/v1/models", timeout=5)
        data = json.loads(resp.read())
        model_ids = [m["id"] for m in data.get("data", [])]
        assert any("Qwen3-32B" in m for m in model_ids), f"Expected Qwen3-32B in {model_ids}"
