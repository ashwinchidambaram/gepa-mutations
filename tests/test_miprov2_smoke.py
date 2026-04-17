"""Smoke tests for MIPROv2 integration (import-level, no LLM needed)."""

import pytest


def test_miprov2_runner_importable():
    """Verify the runner module can be imported without errors."""
    from miprov2.runner import run_miprov2, _QAModule, _examples_to_dspy
    assert callable(run_miprov2)


def test_qa_module_structure():
    """Verify the DSPy module has the expected structure."""
    from miprov2.runner import _QAModule
    import dspy
    module = _QAModule()
    assert hasattr(module, "generate")
    assert len(module.predictors()) == 1


def test_examples_to_dspy():
    """Verify example conversion produces valid DSPy Examples."""
    from miprov2.runner import _examples_to_dspy
    import dspy

    class MockExample:
        input_text = "What is the capital of France?"
        answer = "Paris"

    result = _examples_to_dspy([MockExample()])
    assert len(result) == 1
    assert isinstance(result[0], dspy.Example)
    assert result[0].question == "What is the capital of France?"
    assert result[0].answer == "Paris"


def test_get_scorer_hotpotqa():
    """Verify scorer works for hotpotqa."""
    from gepa_mutations.benchmarks.evaluators import get_scorer
    scorer = get_scorer("hotpotqa")
    assert scorer("Paris", "Paris") == 1.0
    assert scorer("London", "Paris") == 0.0


def test_miprov2_natural_budget_defined():
    """Verify MIPROv2 has natural budget entries."""
    from gepa_mutations.config import PAPER_ROLLOUTS
    assert "miprov2" in PAPER_ROLLOUTS
    assert "hotpotqa" in PAPER_ROLLOUTS["miprov2"]
    assert PAPER_ROLLOUTS["miprov2"]["hotpotqa"] == 1200
