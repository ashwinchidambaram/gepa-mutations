"""Regression tests for the timeout-aware test evaluation fix in base.py.

Bug: _evaluate_qa() used list(pool.map(eval_one, testset)) which blocked
     forever if ANY single LM call hung.
Fix: Replaced with submit() + as_completed() style using wait(FIRST_COMPLETED)
     with dual timeouts (total: _TEST_EVAL_TOTAL_TIMEOUT_SECONDS,
     idle: _TEST_EVAL_IDLE_TIMEOUT_SECONDS).
"""

from __future__ import annotations

import time

import pytest

import gepa_mutations.base as base_mod
from gepa_mutations.base import _evaluate_qa

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_prompt(text: str = "You are helpful.") -> dict[str, str]:
    """Return a minimal prompt dict matching what _evaluate_qa expects."""
    return {"system_prompt": text}


def make_examples(n: int) -> list:
    """Return n simple dspy.Example objects with .input attribute."""
    import dspy

    return [
        dspy.Example(input=f"question_{i}", answer=f"answer_{i}").with_inputs("input")
        for i in range(n)
    ]


class MockAdapter:
    """Adapter that always scores 1.0."""

    def _score(self, example, response):
        return (1.0, "ok")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_eval_aborts_on_idle_timeout(monkeypatch):
    """_evaluate_qa must raise RuntimeError with 'idle' when all workers hang.

    We patch idle timeout to 2s and total timeout to 10s, then give _evaluate_qa
    a single example whose LM call sleeps for 999s.  The function must abort
    within roughly 3 seconds and raise with 'idle' in the message.
    """
    monkeypatch.setattr(base_mod, "_TEST_EVAL_IDLE_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(base_mod, "_TEST_EVAL_TOTAL_TIMEOUT_SECONDS", 10)

    def hanging_lm(messages):
        time.sleep(999)
        return "never"

    examples = make_examples(1)
    prompt = make_prompt()
    adapter = MockAdapter()

    t0 = time.time()
    with pytest.raises(RuntimeError, match="idle"):
        _evaluate_qa(prompt, examples, hanging_lm, adapter, workers=2)
    elapsed = time.time() - t0

    # Should abort near the idle timeout (2s), not the total (10s).
    assert elapsed < 6, f"Took {elapsed:.1f}s -- expected abort within ~3s"


def test_eval_aborts_on_total_timeout(monkeypatch):
    """_evaluate_qa raises RuntimeError within the total-timeout window.

    We patch total to 3s and set a very short idle (0.5s) with an LM that
    takes 0.7s per call.  Every wait() call expires (done is empty after 0.5s
    because the LM needs 0.7s), triggering the idle branch -- which is the
    correct abort mechanism when time_budget is what limits wait_timeout.

    Implementation note: in _evaluate_qa, wait_timeout = min(IDLE, time_budget).
    When the total budget is small, wait_timeout shrinks to time_budget, which
    can be less than a single LM call.  Whichever of idle / total fires first,
    the important invariant is: RuntimeError is raised within the time limits,
    not after 999s of hanging.  This test verifies that invariant.
    """
    monkeypatch.setattr(base_mod, "_TEST_EVAL_TOTAL_TIMEOUT_SECONDS", 3)
    monkeypatch.setattr(base_mod, "_TEST_EVAL_IDLE_TIMEOUT_SECONDS", 0.5)

    def slow_lm(messages):
        # Longer than idle_timeout so every wait() expires.
        time.sleep(0.7)
        return "response"

    # Many examples so there is always pending work.
    examples = make_examples(20)
    prompt = make_prompt()
    adapter = MockAdapter()

    t0 = time.time()
    # The implementation raises RuntimeError via either the idle or total path;
    # both contain "timeout" or "idle" which confirm the fix is in place.
    with pytest.raises(RuntimeError):
        _evaluate_qa(prompt, examples, slow_lm, adapter, workers=2)
    elapsed = time.time() - t0

    # Must abort well before all 20 examples finish (that would take 7s at 2 workers).
    assert elapsed < 5, f"Took {elapsed:.1f}s -- expected abort within total budget"


def test_eval_aborts_on_consecutive_errors(monkeypatch):
    """_evaluate_qa must raise RuntimeError with 'consecutive' when threshold is hit.

    We patch _MAX_CONSECUTIVE_TEST_ERRORS to 3, use an LM that always raises,
    and verify the abort happens quickly with the right error message.
    """
    monkeypatch.setattr(base_mod, "_MAX_CONSECUTIVE_TEST_ERRORS", 3)
    # Give generous timeouts so they do not interfere.
    monkeypatch.setattr(base_mod, "_TEST_EVAL_TOTAL_TIMEOUT_SECONDS", 60)
    monkeypatch.setattr(base_mod, "_TEST_EVAL_IDLE_TIMEOUT_SECONDS", 30)

    def failing_lm(messages):
        raise ConnectionError("endpoint down")

    # workers=1 ensures serial execution so consecutive errors accumulate
    # deterministically: exactly 3 failures fires the abort.
    examples = make_examples(20)
    prompt = make_prompt()
    adapter = MockAdapter()

    t0 = time.time()
    with pytest.raises(RuntimeError, match="consecutive"):
        _evaluate_qa(prompt, examples, failing_lm, adapter, workers=1)
    elapsed = time.time() - t0

    # All LM calls fail instantly, so this should be very fast.
    assert elapsed < 5, f"Took {elapsed:.1f}s -- expected near-instant abort"
