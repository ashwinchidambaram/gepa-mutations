"""Mock LMs and metrics for testing ISO optimizer without GPU.

MockTaskLM: returns canned responses for task evaluation.
MockReflectionLM: returns canned JSON responses for reflection/mutation/discovery.
MockMetric: deterministic metric that scores based on response content.
"""

from __future__ import annotations

import json
import random as _random
from typing import Any


class MockTaskLM:
    """Returns canned responses for task evaluation.

    In DSPy, the LM is called by Predict modules, not directly. But for
    testing the ISO optimizer's evaluation loop, we need the student module
    to produce predictions. This mock can be set as dspy.settings.lm.
    """

    def __init__(self, default_response: str = "mock answer", seed: int = 42):
        self.default_response = default_response
        self.call_count = 0
        self._rng = _random.Random(seed)
        self._responses: dict[str, str] = {}

    def add_response(self, trigger: str, response: str):
        """Add a response for when prompt contains trigger."""
        self._responses[trigger] = response

    def __call__(self, prompt=None, **kwargs):
        self.call_count += 1
        prompt_text = str(prompt) if prompt else ""
        for trigger, response in self._responses.items():
            if trigger in prompt_text:
                return response
        return self.default_response

    @property
    def model(self):
        return "mock-task-lm"


class MockReflectionLM:
    """Returns canned JSON responses matching expected schemas.

    Used for skill discovery, mutation, reflection, and cross-mutation calls.
    Trigger-based: register responses for prompt substrings, or fall back to
    skill discovery response.
    """

    def __init__(self, seed: int = 42):
        self.call_count = 0
        self._rng = _random.Random(seed)
        self._responses: dict[str, str] = {}

    def add_response(self, trigger: str, response: str):
        """Register a canned response for when prompt contains trigger."""
        self._responses[trigger] = response

    def __call__(self, prompt=None, **kwargs):
        self.call_count += 1
        prompt_text = str(prompt) if prompt else ""

        for trigger, response in self._responses.items():
            if trigger in prompt_text:
                return response

        # Detect what type of call this is based on prompt content
        if "cluster" in prompt_text.lower() or "skill" in prompt_text.lower() and "failure" in prompt_text.lower():
            return self._skill_discovery_response()
        elif "variation" in prompt_text.lower() or "mutation" in prompt_text.lower() or "improved prompt" in prompt_text.lower():
            return self._mutation_response()
        elif "combining" in prompt_text.lower() or "blend" in prompt_text.lower() or "parent" in prompt_text.lower():
            return self._cross_mutation_response()
        elif "contrastive" in prompt_text.lower() or "improver" in prompt_text.lower():
            return self._contrastive_response()
        elif "complementary" in prompt_text.lower() or "pairs" in prompt_text.lower():
            return self._pair_proposal_response()
        elif "playbook" in prompt_text.lower():
            return self._playbook_response()
        else:
            return self._mutation_response()  # Safe default

    def _skill_discovery_response(self) -> str:
        return json.dumps({
            "clusters": [
                {
                    "label": "reasoning_errors",
                    "description": "Fails to chain multi-step reasoning correctly",
                    "target_module": None,
                    "example_failure_ids": [],
                },
                {
                    "label": "factual_gaps",
                    "description": "Missing factual knowledge needed for the answer",
                    "target_module": None,
                    "example_failure_ids": [],
                },
                {
                    "label": "format_issues",
                    "description": "Output format doesn't match expected structure",
                    "target_module": None,
                    "example_failure_ids": [],
                },
            ]
        })

    def _mutation_response(self) -> str:
        variant = self._rng.randint(1, 100)
        return json.dumps({
            "prompts": {
                "qa": f"You are a helpful assistant. Think step by step. Variant {variant}.",
            }
        })

    def _cross_mutation_response(self) -> str:
        return json.dumps({
            "prompts": {
                "qa": "You are a helpful expert. Combine careful reasoning with clear explanations.",
            }
        })

    def _contrastive_response(self) -> str:
        return json.dumps({
            "what_worked": "Step-by-step reasoning with explicit intermediate steps",
            "what_failed": "Jumping to conclusions without checking facts",
            "recommended_changes": "Add explicit fact-checking before final answer",
        })

    def _pair_proposal_response(self) -> str:
        return json.dumps({
            "pairs": [
                {
                    "parent_a_id": "placeholder_a",
                    "parent_b_id": "placeholder_b",
                    "rationale": "Parent A excels at reasoning, Parent B at factual recall",
                },
            ]
        })

    def _playbook_response(self) -> str:
        return "Updated playbook: prefer higher minibatch counts for noisy benchmarks."

    @property
    def model(self):
        return "mock-reflection-lm"


class MockMetric:
    """Deterministic metric for testing.

    Scores based on whether the prediction contains a keyword.
    Follows ISO's FeedbackFunction protocol:
        (gold, pred, trace=None, pred_name=None) -> {"score": float, "feedback": str, "metadata": dict}
    """

    def __init__(self, keyword: str = "answer", base_score: float = 0.5):
        self.keyword = keyword
        self.base_score = base_score
        self.call_count = 0

    def __call__(
        self, gold: Any, pred: Any, trace: Any = None, pred_name: str | None = None,
    ) -> dict:
        self.call_count += 1

        # Try to extract text from prediction
        pred_text = ""
        if hasattr(pred, 'answer'):
            pred_text = str(pred.answer)
        elif hasattr(pred, 'output'):
            pred_text = str(pred.output)
        elif isinstance(pred, str):
            pred_text = pred
        else:
            pred_text = str(pred)

        # Try to extract expected answer
        gold_text = ""
        if hasattr(gold, 'answer'):
            gold_text = str(gold.answer)
        elif isinstance(gold, str):
            gold_text = gold

        # Score: base_score if keyword found, 0.0 otherwise
        if self.keyword.lower() in pred_text.lower():
            score = min(1.0, self.base_score + 0.3)
        elif gold_text and gold_text.lower() in pred_text.lower():
            score = 1.0
        else:
            score = self.base_score

        feedback = f"Score: {score:.2f}. Prediction: '{pred_text[:100]}'"

        return {
            "score": score,
            "feedback": feedback,
            "metadata": {},
        }
