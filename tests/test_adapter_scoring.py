"""Regression tests for QAAdapter._score and AIMEAdapter._score fixes.

Bugs tested:
1. QAAdapter had wrong _score (copy-pasted from AIMEAdapter, called _extract_integer
   which QAAdapter doesn't have). Fixed with word-boundary regex matching.
2. AIMEAdapter was missing _score entirely. Fixed by adding integer extraction shim.
"""
from __future__ import annotations

import dspy
import pytest

from gepa_mutations.benchmarks.evaluators import AIMEAdapter, QAAdapter

# ---------------------------------------------------------------------------
# Minimal stubs for constructors
# ---------------------------------------------------------------------------

def _dummy_lm(messages):
    return ""


# ---------------------------------------------------------------------------
# QAAdapter._score — word-boundary answer containment
# ---------------------------------------------------------------------------

class TestQAAdapterScore:
    """QAAdapter._score must use word-boundary matching to avoid false positives."""

    def _adapter(self):
        return QAAdapter(_dummy_lm)

    @pytest.mark.parametrize("answer,response,expected_score", [
        # False-positive guards: short words must not match as substrings
        ("yes", "yesterday I went", 0.0),
        ("no", "another option", 0.0),
        # Normal matches
        ("Paris", "The answer is Paris.", 1.0),
        # Case insensitive
        ("Paris", "the answer is paris", 1.0),
        # Numeric answer
        ("42", "The answer is 42.", 1.0),
        # Partial number must NOT match
        ("42", "The answer is 421.", 0.0),
        # Multi-word answer
        ("New York", "I visited New York last week", 1.0),
    ])
    def test_score_parametrized(self, answer, response, expected_score):
        adapter = self._adapter()
        example = dspy.Example(input="What is the question?", answer=answer)
        score, feedback = adapter._score(example, response)
        assert score == expected_score, (
            f"answer={answer!r}, response={response!r}: "
            f"expected score={expected_score}, got score={score}. feedback={feedback!r}"
        )

    def test_score_returns_tuple(self):
        """_score must return a (float, str) tuple."""
        adapter = self._adapter()
        example = dspy.Example(input="Q?", answer="Paris")
        result = adapter._score(example, "Paris is the capital.")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)

    def test_score_correct_feedback_contains_answer(self):
        """Correct feedback must mention the expected answer."""
        adapter = self._adapter()
        example = dspy.Example(input="Q?", answer="Madrid")
        score, feedback = adapter._score(example, "The capital is Madrid.")
        assert score == 1.0
        assert "Madrid" in feedback

    def test_score_incorrect_feedback_contains_answer(self):
        """Incorrect feedback must hint at the correct answer."""
        adapter = self._adapter()
        example = dspy.Example(input="Q?", answer="Madrid")
        score, feedback = adapter._score(example, "I don't know.")
        assert score == 0.0
        assert "Madrid" in feedback

    def test_yes_inside_yesterday_is_no_match(self):
        """Ensure 'yes' does not match as substring of 'yesterday'."""
        adapter = self._adapter()
        example = dspy.Example(input="Did you go?", answer="yes")
        score, _ = adapter._score(example, "yesterday I went to the store")
        assert score == 0.0

    def test_no_inside_another_is_no_match(self):
        """Ensure 'no' does not match as substring of 'another'."""
        adapter = self._adapter()
        example = dspy.Example(input="Any other options?", answer="no")
        score, _ = adapter._score(example, "another option is available")
        assert score == 0.0

    def test_standalone_yes_matches(self):
        """'yes' must match when it appears as a standalone word."""
        adapter = self._adapter()
        example = dspy.Example(input="Did you go?", answer="yes")
        score, _ = adapter._score(example, "Yes, I went there.")
        assert score == 1.0

    def test_standalone_no_matches(self):
        """'no' must match when it appears as a standalone word."""
        adapter = self._adapter()
        example = dspy.Example(input="Any options?", answer="no")
        score, _ = adapter._score(example, "There is no such option.")
        assert score == 1.0


# ---------------------------------------------------------------------------
# AIMEAdapter._extract_integer
# ---------------------------------------------------------------------------

class TestAIMEAdapterExtractInteger:
    """AIMEAdapter._extract_integer must handle various numeric answer formats."""

    def _adapter(self):
        return AIMEAdapter(_dummy_lm)

    @pytest.mark.parametrize("response,expected", [
        ("Final answer: 42", "42"),
        (r"\boxed{7}", "7"),
        ("no numbers here", None),
        # Last integer wins when multiple candidates exist
        ("I think 3, but actually 5", "5"),
    ])
    def test_extract_integer_parametrized(self, response, expected):
        adapter = self._adapter()
        result = adapter._extract_integer(response)
        assert result == expected, (
            f"response={response!r}: expected={expected!r}, got={result!r}"
        )

    def test_extract_integer_answer_keyword(self):
        """'Answer: N' pattern is recognized."""
        adapter = self._adapter()
        result = adapter._extract_integer("Answer: 100")
        assert result == "100"

    def test_extract_integer_boxed(self):
        r"""'\boxed{N}' pattern is recognized."""
        adapter = self._adapter()
        result = adapter._extract_integer(r"\boxed{42}")
        assert result == "42"

    def test_extract_integer_no_numbers(self):
        """Returns None when no integers are present."""
        adapter = self._adapter()
        result = adapter._extract_integer("The answer cannot be determined.")
        assert result is None

    def test_extract_integer_last_wins(self):
        """When multiple integers appear, the last one is returned as the answer."""
        adapter = self._adapter()
        result = adapter._extract_integer("I first thought 10, then reconsidered: 99")
        # The last standalone integer should be selected
        assert result == "99"

    def test_extract_integer_is_static(self):
        """_extract_integer is a static method and can be called on the class."""
        result = AIMEAdapter._extract_integer("Therefore 15")
        assert result == "15"


# ---------------------------------------------------------------------------
# AIMEAdapter._score
# ---------------------------------------------------------------------------

class TestAIMEAdapterScore:
    """AIMEAdapter._score must correctly compare extracted integer to example.answer."""

    def _adapter(self):
        return AIMEAdapter(_dummy_lm)

    def test_score_correct_answer(self):
        """When response contains the correct answer, score is 1.0."""
        adapter = self._adapter()
        example = dspy.Example(input="Math problem", answer=42, solution="step by step")
        score, feedback = adapter._score(example, "The answer is 42")
        assert score == 1.0

    def test_score_wrong_answer(self):
        """When response contains a wrong integer, score is 0.0."""
        adapter = self._adapter()
        example = dspy.Example(input="Math problem", answer=42, solution="step by step")
        score, feedback = adapter._score(example, "The answer is 7")
        assert score == 0.0

    def test_score_non_parseable(self):
        """When no integer can be extracted from response, score is 0.0."""
        adapter = self._adapter()
        example = dspy.Example(input="Math problem", answer=42)
        score, feedback = adapter._score(example, "I don't know the answer")
        assert score == 0.0

    def test_score_returns_tuple(self):
        """_score must return a (float, str) tuple."""
        adapter = self._adapter()
        example = dspy.Example(input="Math problem", answer=42)
        result = adapter._score(example, "The answer is 42")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)

    def test_score_correct_feedback(self):
        """Correct feedback should indicate the answer is correct."""
        adapter = self._adapter()
        example = dspy.Example(input="Math problem", answer=42)
        score, feedback = adapter._score(example, "Final answer: 42")
        assert score == 1.0
        assert "correct" in feedback.lower()

    def test_score_wrong_feedback_contains_correct_answer(self):
        """Incorrect feedback must mention the expected correct answer."""
        adapter = self._adapter()
        example = dspy.Example(input="Math problem", answer=42)
        score, feedback = adapter._score(example, "The answer is 7")
        assert score == 0.0
        assert "42" in feedback

    def test_score_boxed_correct(self):
        r"""Score 1.0 when the answer is in \boxed{} notation."""
        adapter = self._adapter()
        example = dspy.Example(input="Math problem", answer=7)
        score, _ = adapter._score(example, r"\boxed{7}")
        assert score == 1.0

    def test_score_string_answer_in_example(self):
        """example.answer is cast to int inside _math_metric; string '42' should work."""
        adapter = self._adapter()
        example = dspy.Example(input="Math problem", answer="42")
        score, _ = adapter._score(example, "The answer is 42")
        assert score == 1.0
