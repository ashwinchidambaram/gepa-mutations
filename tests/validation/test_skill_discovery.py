"""V11: Skill discovery validation."""

from __future__ import annotations

import random
import pytest
import dspy
from unittest.mock import MagicMock
from dspy.clients.base_lm import BaseLM

from iso_harness.optimizer.candidate import Candidate, SkillCluster, ModuleTrace
from iso_harness.optimizer.runtime import ISORuntime, RolloutCounter, TraceStore
from iso_harness.optimizer.skill_discovery import (
    discover_skills,
    cluster_failures_via_llm,
    instantiate_candidate_from_skill,
    mutate_candidate,
)
from tests.mocks.mock_lm import MockReflectionLM, MockMetric


# ---------------------------------------------------------------------------
# DSPy-compatible task LM (must extend BaseLM for dspy.Predict to work)
# ---------------------------------------------------------------------------


class _DSPyMockLM(BaseLM):
    """A BaseLM subclass that returns canned text without calling any server."""

    def __init__(self, default_response: str = "mock answer"):
        super().__init__(model="mock-model", cache=False)
        self.default_response = default_response

    def forward(self, prompt=None, messages=None, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        # DSPy parses [[ ## field ## ]] markers from LM output
        response.choices[0].message.content = (
            f"[[ ## answer ## ]]\n{self.default_response}"
        )
        response.usage = {"prompt_tokens": 10, "completion_tokens": 5}
        response.model = "mock-model"
        return response


# ---------------------------------------------------------------------------
# Bad reflection LM (plain callable — no JSON output)
# ---------------------------------------------------------------------------


class _BadReflectionLM:
    """Always returns garbage that can't be parsed as JSON."""

    def __call__(self, prompt=None, **kwargs):
        return "this is not valid json at all {{{"

    @property
    def model(self):
        return "bad-reflection-lm"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_student():
    """Create a simple DSPy student module for testing."""

    class SimpleQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = dspy.Predict("question -> answer")

        def forward(self, question):
            return self.qa(question=question)

    return SimpleQA()


def _make_trainset(n: int = 30):
    """Create a synthetic training set with pre-assigned IDs."""
    examples = []
    for i in range(n):
        ex = dspy.Example(
            question=f"Question {i}?", answer=f"Answer {i}"
        ).with_inputs("question")
        ex.id = f"ex_{i}"
        examples.append(ex)
    return examples


def _make_failure_metric():
    """Return a metric that always produces low scores (< 0.5).

    Uses a keyword that will never appear in the mock LM's response so that
    discover_skills always sees failures and calls cluster_failures_via_llm.
    """
    return MockMetric(keyword="xyz_not_in_response", base_score=0.2)


def _make_runtime(reflection_lm=None, metric=None):
    """Create an ISORuntime backed entirely by mocks."""
    if reflection_lm is None:
        reflection_lm = MockReflectionLM()
    if metric is None:
        metric = _make_failure_metric()
    return ISORuntime(
        reflection_lm=reflection_lm,
        task_lm=_DSPyMockLM(),
        metric=metric,
        run_id="test-v11",
        seed=42,
        rng=random.Random(42),
        trace_store=TraceStore(),
        rollout_counter=RolloutCounter(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestV11SkillDiscovery:
    """V11: Skill discovery produces valid clusters."""

    def setup_method(self):
        """Configure DSPy to use the mock task LM before each test."""
        dspy.settings.configure(lm=_DSPyMockLM())

    def test_discover_skills_returns_valid_clusters(self):
        """discover_skills() returns clusters with non-empty labels and descriptions."""
        student = _make_student()
        runtime = _make_runtime()
        trainset = _make_trainset(30)

        clusters = discover_skills(
            student, trainset, n_discovery_examples=10, runtime=runtime
        )

        assert len(clusters) >= 1
        for cluster in clusters:
            assert cluster.label
            assert cluster.description
            assert isinstance(cluster.label, str)
            assert isinstance(cluster.description, str)

    def test_cluster_count_in_range(self):
        """MockReflectionLM default returns exactly 3 clusters."""
        student = _make_student()
        runtime = _make_runtime()
        trainset = _make_trainset(30)

        clusters = discover_skills(
            student, trainset, n_discovery_examples=10, runtime=runtime
        )

        # MockReflectionLM._skill_discovery_response returns 3 clusters
        assert len(clusters) == 3

    def test_cluster_labels_are_valid(self):
        """All clusters have non-empty label and description strings."""
        student = _make_student()
        runtime = _make_runtime()
        trainset = _make_trainset(30)

        clusters = discover_skills(
            student, trainset, n_discovery_examples=10, runtime=runtime
        )

        for cluster in clusters:
            assert isinstance(cluster, SkillCluster)
            assert len(cluster.label) > 0
            assert len(cluster.description) > 0

    def test_instantiate_candidate_produces_valid_candidate(self):
        """instantiate_candidate_from_skill() produces Candidate with correct metadata."""
        student = _make_student()
        runtime = _make_runtime()

        skill = SkillCluster(
            label="reasoning_errors",
            description="Fails to chain reasoning",
            target_module=None,
            example_traces=[
                ModuleTrace(example_id="ex_0", score=0.2, feedback="wrong")
            ],
        )

        candidate = instantiate_candidate_from_skill(skill, student, runtime)

        assert isinstance(candidate, Candidate)
        assert candidate.birth_mechanism == "skill_discovery"
        assert candidate.skill_category == "reasoning_errors"
        # Student has one predictor named "qa"
        assert "qa" in candidate.prompts_by_module

    def test_generated_prompts_are_non_empty(self):
        """instantiate_candidate_from_skill sets a non-empty prompt string."""
        student = _make_student()
        runtime = _make_runtime()

        skill = SkillCluster(
            label="test_skill",
            description="A test skill",
            example_traces=[],
        )

        candidate = instantiate_candidate_from_skill(skill, student, runtime)

        generated_prompt = candidate.prompts_by_module.get("qa", "")
        assert len(generated_prompt) > 0

    def test_fallback_on_malformed_output(self):
        """Fallback to single default cluster if reflector returns garbage."""
        student = _make_student()
        runtime = _make_runtime(reflection_lm=_BadReflectionLM())
        trainset = _make_trainset(10)

        clusters = discover_skills(
            student, trainset, n_discovery_examples=5, runtime=runtime
        )

        assert len(clusters) == 1
        assert clusters[0].label == "default"

    def test_mutate_candidate_produces_new_id(self):
        """mutate_candidate creates a new Candidate with a different ID."""
        runtime = _make_runtime()

        parent = Candidate(
            id="parent-id",
            prompts_by_module={"qa": "Original prompt"},
            skill_category="test",
        )

        child = mutate_candidate(parent, scope="independent", runtime=runtime)

        assert child.id != parent.id
        assert parent.id in child.parent_ids
        assert child.birth_mechanism == "initial_mutation"
        assert child.skill_category == "test"

    def test_mutate_candidate_inherits_skill_category(self):
        """mutate_candidate preserves skill_category from parent."""
        runtime = _make_runtime()

        parent = Candidate(
            id="parent-id",
            prompts_by_module={"qa": "Original prompt"},
            skill_category="factual_gaps",
        )

        child = mutate_candidate(parent, scope="independent", runtime=runtime)

        assert child.skill_category == "factual_gaps"

    def test_insufficient_failures_returns_default(self):
        """When all examples score high, discover_skills returns single default cluster."""
        student = _make_student()

        # High base_score with keyword='answer' which IS in 'mock answer': score = min(1.0, 0.9+0.3)=1.0
        # All scores = 1.0, failure_threshold = max(median(1.0...), 0.5) = 1.0
        # failures = [t for t in traces if t.score < 1.0] => empty => return default
        high_metric = MockMetric(base_score=0.9)
        runtime = _make_runtime(metric=high_metric)
        trainset = _make_trainset(10)

        clusters = discover_skills(
            student, trainset, n_discovery_examples=5, runtime=runtime
        )

        assert len(clusters) == 1
        assert clusters[0].label == "default"

    def test_small_trainset_still_runs(self):
        """discover_skills handles trainset smaller than n_discovery_examples."""
        student = _make_student()
        runtime = _make_runtime()
        trainset = _make_trainset(5)  # fewer than n_discovery_examples=10

        clusters = discover_skills(
            student, trainset, n_discovery_examples=10, runtime=runtime
        )

        # Should run on all 5 examples — either returns clusters or default
        assert len(clusters) >= 1
        for cluster in clusters:
            assert cluster.label

    def test_cluster_failures_via_llm_with_valid_response(self):
        """cluster_failures_via_llm with MockReflectionLM returns 3 clusters."""
        failures = [
            ModuleTrace(example_id=f"ex_{i}", score=0.2, feedback="wrong")
            for i in range(5)
        ]
        runtime = _make_runtime()

        clusters = cluster_failures_via_llm(
            failures=failures,
            target_n_min=3,
            target_n_max=8,
            modules=["qa"],
            runtime=runtime,
        )

        assert 3 <= len(clusters) <= 8
        for cluster in clusters:
            assert cluster.label
            assert cluster.description
