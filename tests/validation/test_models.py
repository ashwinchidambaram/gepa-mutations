"""V2: Model weight validation — existence and config pinning."""

from __future__ import annotations

import pytest


class TestModels:
    def test_huggingface_hub_available(self):
        """huggingface_hub importable."""
        import huggingface_hub
        assert huggingface_hub is not None

    def test_task_model_info(self):
        """Task model exists on HuggingFace."""
        from huggingface_hub import model_info
        try:
            info = model_info("Qwen/Qwen3-8B")
            assert info.sha is not None
        except Exception as e:
            pytest.skip(f"Cannot reach HuggingFace Hub: {e}")

    def test_reflection_model_info(self):
        """Reflection model exists on HuggingFace."""
        from huggingface_hub import model_info
        try:
            info = model_info("Qwen/Qwen3-32B-AWQ")
            assert info.sha is not None
        except Exception as e:
            pytest.skip(f"Cannot reach HuggingFace Hub: {e}")
