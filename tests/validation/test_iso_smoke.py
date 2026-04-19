"""V16a: End-to-end ISO smoke test with mock LM (no GPU required).

Exercises the full code path: skill discovery → mutation → evaluation →
pruning → reflection → cross-mutation → winner selection.

Runs in CI in under 2 minutes.
"""

from __future__ import annotations

import time
import pytest
import dspy
from unittest.mock import MagicMock
from dspy.clients.base_lm import BaseLM

from iso_harness.optimizer.iso import ISO
from iso_harness.optimizer.runtime import RolloutCounter
from iso_harness.optimizer.helpers import ensure_example_ids
from tests.mocks.mock_lm import MockReflectionLM, MockMetric


# ---------------------------------------------------------------------------
# Reduced config for budget=100
# ---------------------------------------------------------------------------

SMOKE_TEST_OVERRIDES = {
    "n_discovery_examples": 5,
    "target_skills_min": 2,
    "target_skills_max": 3,
    "mutations_per_seed": 0,
    "minibatch_count": 2,
    "minibatch_size": 2,
    "pool_floor": 2,
    "max_rounds": 3,
    "merge_interval": 2,
    "plateau_rounds_threshold": 99,
}


# ---------------------------------------------------------------------------
# DSPy-compatible task LM (must extend BaseLM for dspy.Predict to work)
# ---------------------------------------------------------------------------


class _DSPyMockLM(BaseLM):
    """A BaseLM subclass that returns canned text without calling any server.

    Mirrors the pattern from test_skill_discovery.py so dspy.Predict modules
    can parse the [[ ## field ## ]] markers in the response.
    """

    def __init__(self, default_response: str = "mock answer"):
        super().__init__(model="mock-task-lm", cache=False)
        self.default_response = default_response

    def forward(self, prompt=None, messages=None, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        # DSPy parses [[ ## field ## ]] markers from LM output
        response.choices[0].message.content = (
            f"[[ ## answer ## ]]\n{self.default_response}"
        )
        response.usage = {"prompt_tokens": 10, "completion_tokens": 5}
        response.model = "mock-task-lm"
        return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainset(n: int = 20):
    """Create synthetic training examples with IDs."""
    examples = []
    for i in range(n):
        ex = dspy.Example(
            question=f"What is {i} + {i}?", answer=f"{i + i}"
        ).with_inputs("question")
        ex.id = f"ex_{i}"
        examples.append(ex)
    return examples


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestV16aSmokeTestMock:
    """V16a: Full ISO-Tide wiring test with mock LM."""

    def setup_method(self):
        """Configure DSPy to use the mock task LM before each test."""
        dspy.settings.configure(lm=_DSPyMockLM())

    def test_full_loop_completes(self):
        """ISO-Tide completes with mock LM and smoke config within budget."""
        mock_lm = _DSPyMockLM()
        dspy.settings.configure(lm=mock_lm)

        # Student module
        class SimpleQA(dspy.Module):
            def __init__(self):
                super().__init__()
                self.qa = dspy.Predict("question -> answer")

            def forward(self, question):
                return self.qa(question=question)

        student = SimpleQA()
        trainset = _make_trainset(20)
        valset = _make_trainset(10)
        ensure_example_ids(valset, prefix="val")

        # Mock metric and reflection LM
        metric = MockMetric(base_score=0.4)
        reflection_lm = MockReflectionLM()

        # Create ISO optimizer with smoke overrides
        optimizer = ISO(
            variant="tide",
            metric=metric,
            reflection_lm=reflection_lm,
            task_lm=mock_lm,
            budget=100,
            seed=42,
            **SMOKE_TEST_OVERRIDES,
        )

        start = time.time()
        result = optimizer.compile(student, trainset=trainset, valset=valset)
        elapsed = time.time() - start

        # Assertions
        assert result is not None, "compile() returned None"
        assert hasattr(result, "named_predictors"), "Result is not a DSPy module"
        assert elapsed < 120, f"Smoke test took {elapsed:.1f}s (limit: 120s)"

    def test_all_variants_complete(self):
        """All 5 variants complete with smoke config."""
        mock_lm = _DSPyMockLM()
        dspy.settings.configure(lm=mock_lm)

        class SimpleQA(dspy.Module):
            def __init__(self):
                super().__init__()
                self.qa = dspy.Predict("question -> answer")

            def forward(self, question):
                return self.qa(question=question)

        trainset = _make_trainset(20)
        valset = _make_trainset(10)
        ensure_example_ids(valset, prefix="val")

        for variant in ["sprint", "grove", "tide", "lens", "storm"]:
            student = SimpleQA()
            metric = MockMetric(base_score=0.4)
            reflection_lm = MockReflectionLM()

            optimizer = ISO(
                variant=variant,
                metric=metric,
                reflection_lm=reflection_lm,
                task_lm=mock_lm,
                budget=100,
                seed=42,
                **SMOKE_TEST_OVERRIDES,
            )

            result = optimizer.compile(student, trainset=trainset, valset=valset)
            assert result is not None, f"ISO-{variant} compile() returned None"

    def test_invalid_variant_raises(self):
        """Unknown variant name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ISO variant"):
            ISO(
                variant="invalid",
                metric=MockMetric(),
                reflection_lm=MockReflectionLM(),
                task_lm=None,
                budget=100,
            )

    def test_valset_auto_split(self):
        """When valset=None, trainset is split 80/20."""
        mock_lm = _DSPyMockLM()
        dspy.settings.configure(lm=mock_lm)

        class SimpleQA(dspy.Module):
            def __init__(self):
                super().__init__()
                self.qa = dspy.Predict("question -> answer")

            def forward(self, question):
                return self.qa(question=question)

        trainset = _make_trainset(25)

        optimizer = ISO(
            variant="tide",
            metric=MockMetric(base_score=0.4),
            reflection_lm=MockReflectionLM(),
            task_lm=mock_lm,
            budget=100,
            seed=42,
            **SMOKE_TEST_OVERRIDES,
        )

        # Should not raise — auto-split handles it
        result = optimizer.compile(SimpleQA(), trainset=trainset, valset=None)
        assert result is not None
