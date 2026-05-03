"""Unit tests for ``scalpel.benchmarks.adapter``.

Strategy A from the Phase 2 spec: monkeypatch ``load_benchmark`` and
``get_adapter`` so no test ever hits Hugging Face.  A single test marked
``local_only`` exercises the real loader and is skipped on CI.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gepa_mutations.benchmarks.loader import BenchmarkData
from scalpel.benchmarks.adapter import (
    SUPPORTED_BENCHMARKS,
    Benchmark,
    BenchmarkAdapter,
    load,
)

# --------------------------------------------------------------------------- #
# Helpers / stubs
# --------------------------------------------------------------------------- #


class _StubScorer:
    """Stub adapter exposing the ``_score`` method that ``BenchmarkAdapter`` uses."""

    def __init__(self, score: float = 0.5, fb: str = "fb") -> None:
        self._score_value = score
        self._fb_value = fb
        # Capture the most recent (gold, pred) for assertions.
        self.last_call: tuple[Any, Any] | None = None

    def _score(self, gold: Any, pred: Any) -> tuple[float, str]:
        self.last_call = (gold, pred)
        return self._score_value, self._fb_value


def _fake_examples(n: int) -> list[Any]:
    """Build ``n`` lightweight example stand-ins (not real ``dspy.Example``s)."""
    return [SimpleNamespace(input=f"q{i}", answer=f"a{i}") for i in range(n)]


def _fake_data(n: int = 3) -> BenchmarkData:
    return BenchmarkData(
        train=_fake_examples(n),
        val=_fake_examples(n),
        test=_fake_examples(n),
        metadata={"name": "stub", "split_sizes": {"train": n, "val": n, "test": n}},
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_supported_benchmark_constant() -> None:
    """The list of benchmarks (and order) is locked by spec."""
    assert SUPPORTED_BENCHMARKS == ("hotpotqa", "hover", "ifbench", "pupa", "aime")


def test_protocol_runtime_check() -> None:
    """``BenchmarkAdapter`` instances satisfy the ``Benchmark`` Protocol at runtime."""
    adapter = BenchmarkAdapter("stub", _fake_data(), _StubScorer())
    assert isinstance(adapter, Benchmark)


def test_unknown_benchmark_raises() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark"):
        load("nonsense")


def test_load_returns_three_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scalpel.benchmarks.adapter.load_benchmark",
        lambda name, seed=0: _fake_data(3),
    )
    monkeypatch.setattr(
        "scalpel.benchmarks.adapter.get_adapter",
        lambda name, task_lm=None, parallel_workers=1: _StubScorer(),
    )

    adapter = load("hotpotqa")
    assert len(adapter.trainset) == 3
    assert len(adapter.valset) == 3
    assert len(adapter.testset) == 3
    assert adapter.name == "hotpotqa"


def test_metric_returns_float() -> None:
    """``metric`` returns the float from the underlying ``_score`` call exactly."""
    stub = _StubScorer(score=0.5, fb="fb")
    adapter = BenchmarkAdapter("stub", _fake_data(), stub)
    gold = SimpleNamespace(answer="a0")

    result = adapter.metric(gold, "any-response")

    assert result == 0.5
    assert isinstance(result, float)


def test_feedback_returns_string() -> None:
    """``feedback`` returns the string from the underlying ``_score`` call exactly."""
    stub = _StubScorer(score=0.5, fb="fb")
    adapter = BenchmarkAdapter("stub", _fake_data(), stub)
    gold = SimpleNamespace(answer="a0")

    result = adapter.feedback(gold, "any-response")

    assert result == "fb"
    assert isinstance(result, str)


def test_metric_handles_dspy_prediction_with_answer() -> None:
    """``pred`` objects exposing ``.answer`` are unwrapped to the answer string.

    Behaviour choice: when ``pred`` has an ``.answer`` attribute, the adapter
    forwards ``str(pred.answer)`` to the underlying ``_score`` (which expects
    a free-form response string per ``QAAdapter._score`` etc.).
    """
    stub = _StubScorer(score=1.0, fb="ok")
    adapter = BenchmarkAdapter("stub", _fake_data(), stub)
    gold = SimpleNamespace(answer="gold-answer")
    pred = SimpleNamespace(answer="model-answer-text")

    score = adapter.metric(gold, pred)

    assert score == 1.0
    assert stub.last_call is not None
    forwarded_gold, forwarded_pred = stub.last_call
    assert forwarded_gold is gold
    assert forwarded_pred == "model-answer-text"


def test_metadata_preserved() -> None:
    """``BenchmarkData.metadata`` is copied onto the adapter."""
    data = _fake_data(3)
    adapter = BenchmarkAdapter("stub", data, _StubScorer())

    assert "name" in adapter.metadata
    assert "split_sizes" in adapter.metadata
    assert adapter.metadata["split_sizes"] == {"train": 3, "val": 3, "test": 3}
    # Ensure it's a copy — mutating the adapter's metadata doesn't affect the source.
    adapter.metadata["mutated"] = True
    assert "mutated" not in data.metadata


@pytest.mark.local_only
def test_real_load_hotpotqa() -> None:
    """Smoke test against the real loader.  Skipped on CI; run locally with HF access."""
    adapter = load("hotpotqa", seed=0)
    assert len(adapter.trainset) == 150
    assert len(adapter.valset) == 300
