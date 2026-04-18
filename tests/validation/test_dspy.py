"""V5: DSPy integration validation — LM connection and structured output."""

from __future__ import annotations

import pytest


@pytest.mark.live_server
class TestDSPyIntegration:
    def test_dspy_lm_task(self, live_vllm_task):
        """DSPy LM can connect to task server."""
        import dspy
        lm = dspy.LM(
            "openai/Qwen/Qwen3-8B",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
            max_tokens=64,
            temperature=0.0,
        )
        result = lm("What is 2+2?")
        assert result is not None
        assert len(str(result)) > 0

    def test_dspy_predict(self, live_vllm_task):
        """DSPy Predict signature produces structured output."""
        import dspy
        lm = dspy.LM(
            "openai/Qwen/Qwen3-8B",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
            max_tokens=64,
            temperature=0.0,
        )
        dspy.configure(lm=lm)
        predict = dspy.Predict("question -> answer")
        result = predict(question="What is the capital of France?")
        assert hasattr(result, "answer")
        assert len(result.answer) > 0
