"""Benchmark evaluators and GEPA adapters for the optimize() API.

Each benchmark gets an evaluator function and a GEPAAdapter implementation.
Reference: gepa/examples/aime_math/utils.py and gepa/src/gepa/core/adapter.py
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import dspy
from gepa.core.adapter import EvaluationBatch

from gepa_mutations.benchmarks.signatures import MathSolverSignature


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


def _run_llm(example: dspy.Example, prompt: str, predictor: dspy.ChainOfThought) -> dspy.Prediction:
    """Run the LLM on a single example. Matches utils.py:15-18."""
    predictor.predict.signature.instructions = prompt
    return predictor(input=example.input)


# ---------------------------------------------------------------------------
# AIME GEPAAdapter
# ---------------------------------------------------------------------------

class AIMEAdapter:
    """GEPAAdapter for AIME-2025 math benchmark.

    Uses dspy.ChainOfThought with MathSolverSignature, matching the paper.
    The task LM must be configured via dspy.configure(lm=...) before use.
    """

    propose_new_texts = None  # Use GEPA's default reflective mutation proposer

    def __init__(self):
        self._predictor = dspy.ChainOfThought(MathSolverSignature)

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

        for example in batch:
            try:
                prediction = _run_llm(example, prompt, self._predictor)
                score, feedback = _math_metric(example, prediction)
                answer_text = prediction.answer
            except Exception as e:
                score = 0.0
                feedback = f"Evaluation error: {e}"
                answer_text = ""

            outputs.append({"answer": answer_text})
            scores.append(score)

            if trajectories is not None:
                trajectories.append(TrajectoryRecord(
                    example=example,
                    prompt=prompt,
                    output=answer_text,
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

        for example in batch:
            try:
                response = self._generate(prompt, example.input)
                score, feedback = self._score(example, response)
            except Exception as e:
                response = ""
                score = 0.0
                feedback = f"Evaluation error: {e}"

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
        """Default: answer containment check."""
        answer = str(example.answer)
        if answer.lower() in response.lower():
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

class IFBenchAdapter(QAAdapter):
    """GEPAAdapter for IFBench: scores fraction of constraints satisfied."""

    def _score(self, example: dspy.Example, response: str) -> tuple[float, str]:
        constraints = example.constraints if hasattr(example, "constraints") else []
        if not constraints:
            return 1.0, "No constraints to check."

        satisfied = 0
        feedback_parts = []
        for constraint in constraints:
            constraint_lower = constraint.lower().strip()
            if constraint_lower in response.lower():
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
# HoVer Adapter (label match: SUPPORTS/REFUTES/NOT ENOUGH INFO)
# ---------------------------------------------------------------------------

class HoVerAdapter(QAAdapter):
    """GEPAAdapter for HoVer: fact verification label match."""

    LABELS = {"supported", "not_supported"}

    def _score(self, example: dspy.Example, response: str) -> tuple[float, str]:
        expected = str(example.answer).lower().strip()
        response_lower = response.lower().strip()

        # Check if the expected label appears in the response
        if expected in response_lower:
            return 1.0, f"Correct. The response contains the expected label '{expected}'."
        return 0.0, (
            f"Incorrect. The expected label is '{expected}'. "
            f"Your response did not contain this label."
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

    def _score(self, example: dspy.Example, response: str) -> tuple[float, str]:
        # PII leakage check
        pii_units_raw = getattr(example, "pii_units", "")
        if isinstance(pii_units_raw, list):
            pii_list = [str(p).strip() for p in pii_units_raw if str(p).strip()]
        else:
            pii_list = [p.strip() for p in str(pii_units_raw).split("||") if p.strip()]

        if pii_list:
            leaked = sum(1 for pii in pii_list if pii and pii in response)
            leakage_score = 1.0 - (leaked / len(pii_list))
        else:
            leakage_score = 1.0

        # Simple quality: check if response resembles the gold redaction
        # (LLM-as-judge is expensive; use string similarity for smoke tests)
        gold = str(example.answer).strip().lower()
        resp = response.strip().lower()

        # Check overlap: what fraction of gold words appear in response
        gold_words = set(gold.split())
        resp_words = set(resp.split())
        if gold_words:
            overlap = len(gold_words & resp_words) / len(gold_words)
        else:
            overlap = 0.0

        quality = min(1.0, overlap)
        total_score = (quality + leakage_score) / 2

        feedback_parts = [
            f"Quality (word overlap): {quality:.2f}",
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

        # Check if expected answer appears in the response
        if self._normalize(expected) in self._normalize(response_clean):
            return 1.0, f"Correct (contained). Expected: '{expected}'"

        # Try matching just the last line or boxed answer
        for line in reversed(response_clean.split("\n")):
            line = line.strip()
            if self._normalize(expected) == self._normalize(line):
                return 1.0, f"Correct (last line match). Expected: '{expected}'"
            # Check for \boxed{answer} pattern
            if "\\boxed{" in line:
                boxed = line.split("\\boxed{")[-1].rstrip("}")
                if self._normalize(expected) == self._normalize(boxed):
                    return 1.0, f"Correct (boxed match). Expected: '{expected}'"

        return 0.0, (
            f"Incorrect. Expected: '{expected}'. "
            f"Response did not contain the expected answer."
        )

    @staticmethod
    def _normalize(s: str) -> str:
        """Normalize whitespace and common formatting for comparison."""
        import re
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
        return AIMEAdapter()
    elif benchmark == "hotpotqa":
        return QAAdapter(task_lm)
    elif benchmark == "ifbench":
        return IFBenchAdapter(task_lm)
    elif benchmark == "hover":
        return HoVerAdapter(task_lm)
    elif benchmark == "pupa":
        return PUPAAdapter(task_lm)
    elif benchmark == "livebench":
        return LiveBenchAdapter(task_lm)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
