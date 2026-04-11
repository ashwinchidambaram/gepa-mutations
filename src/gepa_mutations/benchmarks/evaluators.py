"""Benchmark evaluators and GEPA adapters for the optimize() API.

Each benchmark gets an evaluator function and a GEPAAdapter implementation.
Reference: gepa/examples/aime_math/utils.py and gepa/src/gepa/core/adapter.py
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import dspy
from gepa.core.adapter import EvaluationBatch

logger = logging.getLogger(__name__)

# Abort evaluate() batch after this many consecutive LM errors to avoid
# silently burning rollout budget when the inference endpoint is down.
MAX_CONSECUTIVE_EVAL_ERRORS = 10

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

class TrajectoryRecord(NamedTuple):
    """Per-example trace captured during evaluation."""

    example: dspy.Example
    prompt: str
    output: str
    score: float
    feedback: str


# ---------------------------------------------------------------------------
# AIME evaluator (matches gepa/examples/aime_math/utils.py)
# ---------------------------------------------------------------------------

def _math_metric(example: dspy.Example, prediction: dspy.Prediction) -> tuple[float, str]:
    """Score a math prediction. Returns (score, feedback).

    Matches gepa/examples/aime_math/utils.py:21-39.
    """
    correct_answer = int(example.answer)
    written_solution = getattr(example, "solution", "")
    solution_suffix = (
        f" Here's the full step-by-step solution:\n{written_solution}\n\n"
        "Think about what takeaways you can learn from this solution to improve "
        "your future answers and approach to similar problems"
        if written_solution
        else ""
    )

    try:
        llm_answer = int(prediction.answer)
    except (ValueError, TypeError):
        feedback = (
            f"The final answer must be a valid integer and nothing else. "
            f"You responded with '{prediction.answer}', which couldn't be parsed as a python integer. "
            f"Please ensure your answer is a valid integer without any additional text or formatting. "
            f"The correct answer is '{correct_answer}'.{solution_suffix}"
            f"{' and ensure your final answer is a valid integer.' if written_solution else ''}"
        )
        return 0.0, feedback

    score = float(correct_answer == llm_answer)
    status = "correct" if score == 1.0 else "incorrect"
    feedback = f"Your answer is {status}. The correct answer is '{correct_answer}'.{solution_suffix}"
    return score, feedback


# ---------------------------------------------------------------------------
# AIME GEPAAdapter
# ---------------------------------------------------------------------------

class AIMEAdapter:
    """GEPAAdapter for AIME-2025 math benchmark.

    Uses direct LiteLLM calls (via the task_lm callable) with regex-based
    integer extraction. This avoids DSPy's JSONAdapter which fails when the
    model outputs mathematical set notation like {Z}, {i+1}, etc.
    """

    propose_new_texts = None  # Use GEPA's default reflective mutation proposer

    def __init__(self, task_lm):
        self._lm = task_lm

    def _generate(self, prompt: str, question: str) -> str:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
        return self._lm(messages)

    @staticmethod
    def _extract_integer(response: str) -> str | None:
        """Extract final integer answer from model response.

        Tries patterns in order:
        1. Explicit "Final answer: N" or "Answer: N" at end of response
        2. Last standalone integer (0-999) in the response
        Returns None if no integer found.
        """
        # Pattern 1: explicit answer declaration
        for pattern in [
            r"(?:final\s+answer|answer)\s*[=:]\s*(\d+)",
            r"(?:the\s+answer\s+is|therefore|thus|so)\s+(\d+)",
            r"\\boxed\{(\d+)\}",
        ]:
            matches = list(re.finditer(pattern, response, re.IGNORECASE))
            if matches:
                return matches[-1].group(1)

        # Pattern 2: last standalone integer in response
        integers = re.findall(r"\b(\d{1,3})\b", response)
        if integers:
            return integers[-1]

        return None

    def evaluate(
        self,
        batch: list[dspy.Example],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[TrajectoryRecord, dict[str, str]]:
        prompt = candidate["system_prompt"]
        outputs: list[dict[str, str]] = []
        scores: list[float] = []
        trajectories: list[TrajectoryRecord] | None = [] if capture_traces else None

        consecutive_errors = 0

        for example in batch:
            try:
                response = self._generate(prompt, example.input) or ""
                answer_text = self._extract_integer(response) or ""
                # Create a pseudo-prediction for _math_metric
                prediction = type("Prediction", (), {"answer": answer_text})()
                score, feedback = _math_metric(example, prediction)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                score = 0.0
                feedback = f"Evaluation error: {e}"
                answer_text = ""
                response = ""
                if consecutive_errors <= 3:
                    logger.warning("AIME eval error (%d consecutive): %s", consecutive_errors, e)
                if consecutive_errors >= MAX_CONSECUTIVE_EVAL_ERRORS:
                    raise RuntimeError(
                        f"Aborting AIME evaluation: {consecutive_errors} consecutive LM errors. "
                        f"Last error: {e}"
                    ) from e

            outputs.append({"answer": answer_text})
            scores.append(score)

            if trajectories is not None:
                trajectories.append(TrajectoryRecord(
                    example=example,
                    prompt=prompt,
                    output=response,
                    score=score,
                    feedback=feedback,
                ))

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[TrajectoryRecord, dict[str, str]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        assert eval_batch.trajectories is not None
        result: dict[str, list[dict[str, str]]] = {}

        for comp in components_to_update:
            items: list[dict[str, str]] = []
            for traj in eval_batch.trajectories:
                items.append({
                    "Inputs": traj.example.input,
                    "Generated Outputs": traj.output,
                    "Feedback": traj.feedback,
                })
            result[comp] = items

        return result


# ---------------------------------------------------------------------------
# Generic QA Adapter (for HotpotQA, HoVer, etc.)
# ---------------------------------------------------------------------------

class QAAdapter:
    """GEPAAdapter for QA benchmarks using direct LiteLLM calls.

    Evaluates answer containment (ContainsAnswer pattern from GEPA).
    """

    propose_new_texts = None

    def __init__(self, task_lm_callable):
        """
        Args:
            task_lm_callable: A callable (gepa.lm.LM instance) for generating answers.
        """
        self._lm = task_lm_callable

    def _generate(self, prompt: str, question: str) -> str:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
        return self._lm(messages)

    def evaluate(
        self,
        batch: list[dspy.Example],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[TrajectoryRecord, dict[str, str]]:
        prompt = candidate["system_prompt"]
        outputs: list[dict[str, str]] = []
        scores: list[float] = []
        trajectories: list[TrajectoryRecord] | None = [] if capture_traces else None

        consecutive_errors = 0

        for example in batch:
            try:
                response = self._generate(prompt, example.input) or ""
                score, feedback = self._score(example, response)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                response = ""
                score = 0.0
                feedback = f"Evaluation error: {e}"
                if consecutive_errors <= 3:
                    logger.warning("QA eval error (%d consecutive): %s", consecutive_errors, e)
                if consecutive_errors >= MAX_CONSECUTIVE_EVAL_ERRORS:
                    raise RuntimeError(
                        f"Aborting QA evaluation: {consecutive_errors} consecutive LM errors. "
                        f"Last error: {e}"
                    ) from e

            outputs.append({"response": response})
            scores.append(score)

            if trajectories is not None:
                trajectories.append(TrajectoryRecord(
                    example=example,
                    prompt=prompt,
                    output=response,
                    score=score,
                    feedback=feedback,
                ))

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def _score(self, example: dspy.Example, response: str) -> tuple[float, str]:
        """Default: answer containment check with word-boundary matching.

        Uses word boundaries to avoid false positives from short answers (e.g.
        'yes' matching inside 'yesterday', 'no' inside 'another').
        """
        answer = str(example.answer)
        pattern = r'\b' + re.escape(answer.lower()) + r'\b'
        if re.search(pattern, response.lower()):
            return 1.0, f"Correct. The response contains the expected answer '{answer}'."
        return 0.0, (
            f"Incorrect. The correct answer is '{answer}'. "
            "Ensure that the correct answer is included in the response."
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[TrajectoryRecord, dict[str, str]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        assert eval_batch.trajectories is not None
        result: dict[str, list[dict[str, str]]] = {}

        for comp in components_to_update:
            items: list[dict[str, str]] = []
            for traj in eval_batch.trajectories:
                items.append({
                    "Inputs": traj.example.input,
                    "Generated Outputs": traj.output,
                    "Feedback": traj.feedback,
                })
            result[comp] = items

        return result


# ---------------------------------------------------------------------------
# IFBench Adapter (fraction of constraints satisfied)
# ---------------------------------------------------------------------------

# IFBench constraint checkers — programmatic verification of structural
# requirements from allenai/IF_multi_constraints_upto5 (IFEval-style).
# Constraints are natural language descriptions like:
#   "Your response should contain at least 3 paragraphs."
#   "Include the keyword 'ocean' in your response."
#   "Your entire response should be in English, and in all uppercase."
#   "Your response must contain at least 200 words."

def _check_ifbench_constraint(constraint: str, response: str) -> bool:
    """Programmatically check a single IFBench constraint against a response.

    Handles common IFEval constraint types with regex-based parsing.
    Returns False (conservative) for unrecognized constraint types.
    """
    c = constraint.strip()
    c_lower = c.lower()

    # --- Word count constraints ---
    # "at least N words" / "no more than N words" / "exactly N words"
    m = re.search(r"(?:at\s+least|minimum\s+(?:of\s+)?|no\s+(?:fewer|less)\s+than)\s+(\d+)\s+words", c_lower)
    if m:
        min_words = int(m.group(1))
        word_count = len(response.split())
        return word_count >= min_words

    m = re.search(r"(?:at\s+most|no\s+more\s+than|maximum\s+(?:of\s+)?|not\s+exceed|fewer\s+than)\s+(\d+)\s+words", c_lower)
    if m:
        max_words = int(m.group(1))
        word_count = len(response.split())
        # "fewer than N" means < N; "at most N" / "no more than N" means <= N
        if "fewer than" in c_lower:
            return word_count < max_words
        return word_count <= max_words

    m = re.search(r"(?:exactly|precisely)\s+(\d+)\s+words", c_lower)
    if m:
        target = int(m.group(1))
        return len(response.split()) == target

    # --- Sentence count constraints ---
    m = re.search(r"(?:at\s+least|minimum\s+(?:of\s+)?)\s+(\d+)\s+sentence", c_lower)
    if m:
        min_sentences = int(m.group(1))
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        return len(sentences) >= min_sentences

    m = re.search(r"(?:at\s+most|no\s+more\s+than)\s+(\d+)\s+sentence", c_lower)
    if m:
        max_sentences = int(m.group(1))
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        return len(sentences) <= max_sentences

    # --- Paragraph count constraints ---
    m = re.search(r"(?:at\s+least|minimum\s+(?:of\s+)?)\s+(\d+)\s+paragraph", c_lower)
    if m:
        min_paras = int(m.group(1))
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        return len(paragraphs) >= min_paras

    m = re.search(r"(?:at\s+most|no\s+more\s+than)\s+(\d+)\s+paragraph", c_lower)
    if m:
        max_paras = int(m.group(1))
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        return len(paragraphs) <= max_paras

    m = re.search(r"(?:exactly|precisely)\s+(\d+)\s+paragraph", c_lower)
    if m:
        target = int(m.group(1))
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        return len(paragraphs) == target

    # --- Keyword inclusion ---
    # "include the keyword 'X'" / "must contain the keyword 'X'"
    # Also handles: "include the keywords 'X', 'Y', and 'Z'"
    m = re.search(r"(?:include|contain|use)\s+(?:the\s+)?keywords?\s+(.+)", c_lower)
    if m:
        keywords_str = m.group(1)
        # Extract quoted keywords
        keywords = re.findall(r"['\"]([^'\"]+)['\"]", keywords_str)
        if not keywords:
            # Try unquoted single keyword
            keywords = [keywords_str.strip().rstrip(".")]
        response_lower = response.lower()
        return all(kw.lower() in response_lower for kw in keywords)

    # --- Keyword/phrase forbidden ---
    m = re.search(r"(?:do\s+not|don'?t|avoid|never)\s+(?:use|include|contain|mention)\s+(?:the\s+)?(?:word|keyword|phrase|term)s?\s+(.+)", c_lower)
    if m:
        keywords_str = m.group(1)
        keywords = re.findall(r"['\"]([^'\"]+)['\"]", keywords_str)
        if not keywords:
            keywords = [keywords_str.strip().rstrip(".")]
        response_lower = response.lower()
        return all(kw.lower() not in response_lower for kw in keywords)

    # --- All uppercase ---
    if re.search(r"(?:all|entire|whole)\s+(?:response\s+)?(?:should\s+be\s+)?(?:in\s+)?(?:all\s+)?uppercase", c_lower):
        # Check that all alphabetic characters are uppercase
        alpha_chars = [ch for ch in response if ch.isalpha()]
        return len(alpha_chars) > 0 and all(ch.isupper() for ch in alpha_chars)

    # --- All lowercase ---
    if re.search(r"(?:all|entire|whole)\s+(?:response\s+)?(?:should\s+be\s+)?(?:in\s+)?(?:all\s+)?lowercase", c_lower):
        alpha_chars = [ch for ch in response if ch.isalpha()]
        return len(alpha_chars) > 0 and all(ch.islower() for ch in alpha_chars)

    # --- Title case ---
    if "title case" in c_lower or "capitalize each word" in c_lower:
        words = response.split()
        if not words:
            return False
        return all(w[0].isupper() for w in words if w and w[0].isalpha())

    # --- Bullet points / numbered list ---
    m = re.search(r"(?:at\s+least|minimum\s+(?:of\s+)?)\s+(\d+)\s+(?:bullet\s+point|item)", c_lower)
    if m:
        min_items = int(m.group(1))
        bullet_lines = [line for line in response.split("\n")
                       if re.match(r'\s*[-*\u2022]\s', line) or re.match(r'\s*\d+[.)]\s', line)]
        return len(bullet_lines) >= min_items

    # --- Response format: starts with / ends with ---
    m = re.search(r"(?:start|begin)\s+(?:your\s+)?(?:response\s+)?with\s+['\"]([^'\"]+)['\"]", c_lower)
    if m:
        start_text = m.group(1)
        return response.strip().lower().startswith(start_text.lower())

    m = re.search(r"end\s+(?:your\s+)?(?:response\s+)?with\s+['\"]([^'\"]+)['\"]", c_lower)
    if m:
        end_text = m.group(1)
        return response.strip().lower().endswith(end_text.lower())

    # --- Letter / character count constraints ---
    # "letter" → count only alphabetic chars; "character" → count all chars including spaces.
    m = re.search(r"(?:at\s+least|minimum\s+(?:of\s+)?)\s+(\d+)\s+(letter|character)", c_lower)
    if m:
        min_count = int(m.group(1))
        if m.group(2) == "letter":
            return sum(1 for ch in response if ch.isalpha()) >= min_count
        else:
            return len(response) >= min_count

    # --- Comma-separated items ---
    m = re.search(r"(?:at\s+least|minimum\s+(?:of\s+)?)\s+(\d+)\s+comma[- ]?separated", c_lower)
    if m:
        min_items = int(m.group(1))
        # Count items in the longest comma-separated sequence
        items = [item.strip() for item in response.split(",") if item.strip()]
        return len(items) >= min_items

    # --- Section / heading constraints ---
    m = re.search(r"(?:at\s+least|minimum\s+(?:of\s+)?)\s+(\d+)\s+(?:section|heading)", c_lower)
    if m:
        min_sections = int(m.group(1))
        headings = [line for line in response.split("\n")
                   if re.match(r'\s*#{1,6}\s', line) or (line.strip().isupper() and len(line.strip()) > 2)]
        return len(headings) >= min_sections

    # --- No commas ---
    if re.search(r"(?:do\s+not|don'?t|avoid|no)\s+(?:use\s+)?commas?", c_lower):
        return "," not in response

    # --- Specific language / format pattern (e.g., "in English") ---
    # Skip these — can't easily verify language programmatically

    # --- Placeholder/format patterns ---
    # "wrap your response in double quotes"
    if "wrap" in c_lower and "double quote" in c_lower:
        stripped = response.strip()
        return stripped.startswith('"') and stripped.endswith('"')

    # "put your response in a markdown code block"
    if "markdown code block" in c_lower or "code block" in c_lower:
        return "```" in response

    # --- Fallback: unrecognized constraint → conservative (fail) ---
    # This prevents inflated scores from unrecognized constraint types.
    return False


class IFBenchAdapter(QAAdapter):
    """GEPAAdapter for IFBench: scores fraction of constraints satisfied.

    Uses programmatic constraint checking for common IFEval constraint types
    (word count, paragraph count, keyword inclusion, case requirements, etc.).
    Unrecognized constraint types conservatively score as not satisfied.
    """

    def _score(self, example: dspy.Example, response: str) -> tuple[float, str]:
        constraints = example.constraints if hasattr(example, "constraints") else []
        if not constraints:
            return 1.0, "No constraints to check."

        satisfied = 0
        feedback_parts = []
        for constraint in constraints:
            passed = _check_ifbench_constraint(constraint, response)
            if passed:
                satisfied += 1
                feedback_parts.append(f"  - SATISFIED: {constraint}")
            else:
                feedback_parts.append(f"  - NOT SATISFIED: {constraint}")

        score = satisfied / len(constraints)
        feedback = (
            f"Score: {score:.2f} ({satisfied}/{len(constraints)} constraints satisfied)\n"
            + "\n".join(feedback_parts)
        )
        return score, feedback


# ---------------------------------------------------------------------------
# HoVer Adapter (label match: SUPPORTED / NOT_SUPPORTED)
# ---------------------------------------------------------------------------

def _extract_hover_verdict(response: str) -> str | None:
    """Extract the verdict label from a HoVer model response.

    The model reasons about whether a claim is supported, so "supported"
    appears in nearly every response. We need to extract the actual verdict,
    not just check for substring presence.

    Strategy (ordered by specificity):
    1. Look for explicit verdict patterns (e.g. "Final answer: NOT_SUPPORTED")
    2. Find the LAST occurrence of NOT_SUPPORTED or SUPPORTED (the conclusion)
    3. If only one label appears, use that

    Returns "supported", "not_supported", or None if no verdict found.
    """
    resp = response.strip()

    # Normalize common variations: NOT SUPPORTED -> NOT_SUPPORTED
    # This handles "NOT SUPPORTED", "Not Supported", etc.
    resp_normalized = re.sub(r'\bnot[\s_-]+supported\b', 'NOT_SUPPORTED', resp, flags=re.IGNORECASE)

    # 1. Check for explicit verdict patterns first
    # Patterns like: "Final answer: SUPPORTED", "Verdict: NOT_SUPPORTED",
    # "Therefore, the claim is SUPPORTED", "The answer is NOT_SUPPORTED"
    verdict_patterns = [
        r'(?:final\s+(?:answer|verdict|conclusion|determination)|verdict|conclusion|determination|answer)\s*[:=]\s*(NOT_SUPPORTED|SUPPORTED)',
        r'(?:the\s+claim\s+is|this\s+claim\s+is|claim\s+is)\s+(NOT_SUPPORTED|SUPPORTED)',
        r'(?:therefore|thus|hence|so|in\s+conclusion)\s*,?\s*(?:the\s+claim\s+is\s+)?(NOT_SUPPORTED|SUPPORTED)',
        r'\*\*(NOT_SUPPORTED|SUPPORTED)\*\*',  # **SUPPORTED** markdown bold
    ]
    last_match_pos = -1
    last_match_label = None

    for pattern in verdict_patterns:
        for m in re.finditer(pattern, resp_normalized, re.IGNORECASE):
            if m.start() > last_match_pos:
                last_match_pos = m.start()
                last_match_label = m.group(1).lower().replace(" ", "_").replace("-", "_")

    if last_match_label is not None:
        return last_match_label

    # 2. Find the LAST occurrence of each label in the response.
    # The final mention is most likely the verdict (after reasoning).
    last_not_supported = -1
    last_supported = -1

    for m in re.finditer(r'\bNOT_SUPPORTED\b', resp_normalized, re.IGNORECASE):
        last_not_supported = m.start()

    for m in re.finditer(r'\bSUPPORTED\b', resp_normalized, re.IGNORECASE):
        # Only count standalone SUPPORTED, not part of NOT_SUPPORTED
        pos = m.start()
        # Check this isn't the SUPPORTED part of NOT_SUPPORTED
        prefix_start = max(0, pos - 4)
        prefix = resp_normalized[prefix_start:pos]
        if not re.search(r'NOT[_\s-]$', prefix, re.IGNORECASE):
            last_supported = pos

    # Return whichever appeared last (the verdict after reasoning)
    if last_not_supported > last_supported:
        return "not_supported"
    elif last_supported > last_not_supported:
        return "supported"

    # 3. No label found at all
    return None


class HoVerAdapter(QAAdapter):
    """GEPAAdapter for HoVer: fact verification label match.

    Extracts the verdict label from the model response rather than simple
    substring matching, to avoid false positives from reasoning text.
    """

    LABELS = {"supported", "not_supported"}

    def _score(self, example: dspy.Example, response: str) -> tuple[float, str]:
        expected = str(example.answer).lower().strip()
        predicted = _extract_hover_verdict(response)

        if predicted is None:
            return 0.0, (
                f"Incorrect. Could not extract a verdict from the response. "
                f"The expected label is '{expected}'. "
                f"Ensure your response clearly states SUPPORTED or NOT_SUPPORTED as the final verdict."
            )

        if predicted == expected:
            return 1.0, f"Correct. Extracted verdict '{predicted}' matches expected '{expected}'."

        return 0.0, (
            f"Incorrect. Extracted verdict '{predicted}' does not match expected '{expected}'. "
            f"Ensure your final answer clearly states the correct verdict."
        )


# ---------------------------------------------------------------------------
# PUPA Adapter (PII redaction quality + leakage detection)
# Reference: gepa/tests/test_pareto_frontier_types/test_pareto_frontier_types.py
# ---------------------------------------------------------------------------

class PUPAAdapter(QAAdapter):
    """GEPAAdapter for PUPA: privacy-preserving query redaction.

    Scoring: average of (quality, leakage_score).
    - quality: LLM-as-judge comparing response to gold redaction (0-1)
    - leakage: fraction of PII units NOT leaked in response (0-1)
    """

    def __init__(self, task_lm_callable, *, judge_lm=None):
        super().__init__(task_lm_callable)
        self._judge_lm = judge_lm

    # ------------------------------------------------------------------
    # Quality scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _word_overlap_quality(gold: str, resp: str) -> float:
        """Word-overlap quality score between gold redaction and response."""
        gold_lower = gold.strip().lower()
        resp_lower = resp.strip().lower()
        gold_words = set(gold_lower.split())
        resp_words = set(resp_lower.split())
        if gold_words:
            overlap = len(gold_words & resp_words) / len(gold_words)
        else:
            overlap = 0.0
        return min(1.0, overlap)

    def _judge_quality(self, gold: str, resp: str) -> float:
        """LLM-as-judge quality score comparing response to gold redaction.

        Falls back to word-overlap on any failure.
        Reference: gepa/tests/test_pareto_frontier_types/test_pareto_frontier_types.py
        """
        try:
            if self._judge_lm is None:
                return self._word_overlap_quality(gold, resp)
            messages = [
                {
                    "role": "system",
                    "content": "You are a strict grader evaluating privacy-preserving query rewriting. Return only a number between 0 and 1.",
                },
                {
                    "role": "user",
                    "content": (
                        "Compare the assistant's rewritten query to the gold-standard redaction.\n\n"
                        f"GOLD REDACTION:\n{gold.strip()}\n\n"
                        f"ASSISTANT RESPONSE:\n{resp.strip()}\n\n"
                        "Score (0=completely wrong, 1=perfect match):"
                    ),
                },
            ]
            raw = self._judge_lm(messages)
            try:
                score = float(raw.strip())
            except (ValueError, TypeError):
                # Regex fallback: extract first float-like substring
                m = re.search(r"(\d*\.?\d+)", raw)
                if m:
                    score = float(m.group(1))
                else:
                    return self._word_overlap_quality(gold, resp)
            return max(0.0, min(1.0, score))
        except Exception:
            return self._word_overlap_quality(gold, resp)

    def _score(self, example: dspy.Example, response: str) -> tuple[float, str]:
        # PII leakage check
        pii_units_raw = getattr(example, "pii_units", "")
        if isinstance(pii_units_raw, list):
            pii_list = [str(p).strip() for p in pii_units_raw if str(p).strip()]
        else:
            pii_list = [p.strip() for p in str(pii_units_raw).split("||") if p.strip()]

        if pii_list:
            leaked = sum(1 for pii in pii_list if pii and pii.lower() in response.lower())
            leakage_score = 1.0 - (leaked / len(pii_list))
        else:
            leakage_score = 1.0

        # Quality: LLM-as-judge when available, else word-overlap fallback
        gold = str(example.answer)
        if self._judge_lm is not None:
            quality = self._judge_quality(gold, response)
        else:
            quality = self._word_overlap_quality(gold, response)
        total_score = (quality + leakage_score) / 2

        quality_method = "LLM-as-judge" if self._judge_lm is not None else "word overlap"
        feedback_parts = [
            f"Quality ({quality_method}): {quality:.2f}",
            f"Leakage score: {leakage_score:.2f} ({len(pii_list)} PII units checked)",
            f"Total: {total_score:.2f}",
        ]
        if leakage_score < 1.0:
            feedback_parts.append(
                f"WARNING: PII leakage detected ({int((1-leakage_score)*len(pii_list))}/{len(pii_list)} units leaked)"
            )
        feedback = "\n".join(feedback_parts)

        return total_score, feedback


# ---------------------------------------------------------------------------
# LiveBench-Math Adapter (string comparison for diverse answer formats)
# ---------------------------------------------------------------------------

class LiveBenchAdapter(QAAdapter):
    """GEPAAdapter for LiveBench-Math: string comparison scoring.

    LiveBench answers are diverse: comma-separated sequences, LaTeX expressions,
    zero-padded numbers, letter choices. Uses normalized string comparison.
    """

    def _score(self, example: dspy.Example, response: str) -> tuple[float, str]:
        expected = str(example.answer).strip()
        response_clean = response.strip()

        # Try exact match first (normalized whitespace)
        if self._normalize(expected) == self._normalize(response_clean):
            return 1.0, f"Correct (exact match). Expected: '{expected}'"

        # Check if expected answer appears in the response (word-boundary to avoid
        # false positives when the expected answer is a short string like "1" or "A").
        norm_exp = self._normalize(expected)
        norm_resp = self._normalize(response_clean)
        if re.search(r'\b' + re.escape(norm_exp) + r'\b', norm_resp):
            return 1.0, f"Correct (contained). Expected: '{expected}'"

        # Try matching just the last line or boxed answer
        for line in reversed(response_clean.split("\n")):
            line = line.strip()
            if self._normalize(expected) == self._normalize(line):
                return 1.0, f"Correct (last line match). Expected: '{expected}'"
            # Check for \boxed{answer} pattern
            if "\\boxed{" in line:
                # Brace-depth parsing to handle nested braces like \boxed{\frac{1}{2}}
                raw = line.split("\\boxed{", 1)[-1]
                depth, end = 1, len(raw)
                for ci, ch in enumerate(raw):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = ci
                            break
                boxed = raw[:end]
                if self._normalize(expected) == self._normalize(boxed):
                    return 1.0, f"Correct (boxed match). Expected: '{expected}'"

        return 0.0, (
            f"Incorrect. Expected: '{expected}'. "
            f"Response did not contain the expected answer."
        )

    @staticmethod
    def _normalize(s: str) -> str:
        """Normalize whitespace and common formatting for comparison."""
        s = s.lower().strip()
        s = re.sub(r"\s+", " ", s)
        # Remove trailing periods, commas
        s = s.rstrip(".,;")
        return s


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_adapter(benchmark: str, task_lm=None):
    """Get the appropriate adapter for a benchmark.

    Args:
        benchmark: Benchmark name.
        task_lm: LM callable for QA-style benchmarks (gepa.lm.LM instance).

    Returns:
        A GEPAAdapter instance.
    """
    if benchmark == "aime":
        if task_lm is None:
            raise ValueError("AIMEAdapter requires task_lm — pass a QA-style LM callable")
        return AIMEAdapter(task_lm)
    elif benchmark == "hotpotqa":
        return QAAdapter(task_lm)
    elif benchmark == "ifbench":
        return IFBenchAdapter(task_lm)
    elif benchmark == "hover":
        return HoVerAdapter(task_lm)
    elif benchmark == "pupa":
        return PUPAAdapter(task_lm, judge_lm=task_lm)
    elif benchmark == "livebench":
        return LiveBenchAdapter(task_lm)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
