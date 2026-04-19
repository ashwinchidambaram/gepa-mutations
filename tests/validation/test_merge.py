"""V15: Merge operator validation."""
from __future__ import annotations

import random
import pytest

from iso_harness.optimizer.candidate import Candidate
from iso_harness.optimizer.runtime import ISORuntime, RolloutCounter, TraceStore
from iso_harness.optimizer.merge import merge_candidates, compute_per_module_scores, top_pareto_candidates
from tests.mocks.mock_lm import MockReflectionLM, MockMetric


def _make_runtime():
    return ISORuntime(
        reflection_lm=MockReflectionLM(),
        task_lm=None,
        metric=MockMetric(),
        run_id="test-v15",
        seed=42,
        rng=random.Random(42),
        trace_store=TraceStore(),
        rollout_counter=RolloutCounter(),
        round_num=3,
    )


class TestV15Merge:
    """V15: Merge produces valid children on multi-module, no-op on single."""

    def test_single_module_returns_none(self):
        """merge_candidates on single-module system returns None."""
        a = Candidate(id="a", prompts_by_module={"qa": "prompt A"})
        b = Candidate(id="b", prompts_by_module={"qa": "prompt B"})
        scores = {
            "a": {"mean": 0.8, "per_example": {}, "per_example_metadata": {}},
            "b": {"mean": 0.6, "per_example": {}, "per_example_metadata": {}},
        }
        result = merge_candidates([a, b], scores, _make_runtime())
        assert result is None

    def test_multi_module_returns_candidate(self):
        """merge_candidates on multi-module system returns valid Candidate."""
        a = Candidate(id="a", prompts_by_module={"step1": "A step1", "step2": "A step2"})
        b = Candidate(id="b", prompts_by_module={"step1": "B step1", "step2": "B step2"})
        scores = {
            "a": {"mean": 0.8, "per_example": {"ex_0": 0.8}, "per_example_metadata": {}},
            "b": {"mean": 0.6, "per_example": {"ex_0": 0.6}, "per_example_metadata": {}},
        }
        result = merge_candidates([a, b], scores, _make_runtime())
        assert result is not None
        assert isinstance(result, Candidate)
        assert result.birth_mechanism == "merge"
        assert set(result.parent_ids) == {"a", "b"}
        assert "step1" in result.prompts_by_module
        assert "step2" in result.prompts_by_module

    def test_per_module_best_selection(self):
        """Child inherits each module from the higher-scoring parent."""
        a = Candidate(id="a", prompts_by_module={"step1": "A step1", "step2": "A step2"})
        b = Candidate(id="b", prompts_by_module={"step1": "B step1", "step2": "B step2"})
        # Parent A better at step1 (0.9 vs 0.3), Parent B better at step2 (0.4 vs 0.8)
        scores = {
            "a": {
                "mean": 0.65, "per_example": {"ex_0": 0.65},
                "per_example_metadata": {
                    "ex_0": {"per_module_score": {"step1": 0.9, "step2": 0.4}},
                },
            },
            "b": {
                "mean": 0.55, "per_example": {"ex_0": 0.55},
                "per_example_metadata": {
                    "ex_0": {"per_module_score": {"step1": 0.3, "step2": 0.8}},
                },
            },
        }
        result = merge_candidates([a, b], scores, _make_runtime())
        assert result is not None
        assert result.prompts_by_module["step1"] == "A step1"  # A better at step1
        assert result.prompts_by_module["step2"] == "B step2"  # B better at step2

    def test_fallback_whole_system_inheritance(self):
        """Falls back to whole-system when per-module scores unavailable."""
        a = Candidate(id="a", prompts_by_module={"step1": "A1", "step2": "A2"})
        b = Candidate(id="b", prompts_by_module={"step1": "B1", "step2": "B2"})
        scores = {
            "a": {"mean": 0.8, "per_example": {}, "per_example_metadata": {}},
            "b": {"mean": 0.6, "per_example": {}, "per_example_metadata": {}},
        }
        result = merge_candidates([a, b], scores, _make_runtime())
        assert result is not None
        # A has higher mean, so all modules should come from A
        assert result.prompts_by_module["step1"] == "A1"
        assert result.prompts_by_module["step2"] == "A2"

    def test_wrong_parent_count_returns_none(self):
        """merge_candidates with != 2 parents returns None."""
        a = Candidate(id="a", prompts_by_module={"s1": "p1", "s2": "p2"})
        scores = {"a": {"mean": 0.8}}
        assert merge_candidates([a], scores, _make_runtime()) is None
        assert merge_candidates([], scores, _make_runtime()) is None

    def test_birth_mechanism_is_merge(self):
        """Child has birth_mechanism = 'merge'."""
        a = Candidate(id="a", prompts_by_module={"s1": "A1", "s2": "A2"})
        b = Candidate(id="b", prompts_by_module={"s1": "B1", "s2": "B2"})
        scores = {
            "a": {"mean": 0.8, "per_example": {}, "per_example_metadata": {}},
            "b": {"mean": 0.6, "per_example": {}, "per_example_metadata": {}},
        }
        result = merge_candidates([a, b], scores, _make_runtime())
        assert result.birth_mechanism == "merge"


class TestTopParetoCandidates:
    def test_returns_frontier_candidates(self):
        """Pareto frontier includes candidates best on at least one example."""
        pool = [
            Candidate(id="c0"), Candidate(id="c1"), Candidate(id="c2"),
        ]
        scores = {
            "c0": {"mean": 0.8, "per_example": {"ex_0": 1.0, "ex_1": 0.5}},
            "c1": {"mean": 0.7, "per_example": {"ex_0": 0.5, "ex_1": 1.0}},
            "c2": {"mean": 0.3, "per_example": {"ex_0": 0.3, "ex_1": 0.3}},
        }
        result = top_pareto_candidates(pool, scores, n=2)
        ids = {c.id for c in result}
        assert "c0" in ids  # best on ex_0
        assert "c1" in ids  # best on ex_1
        assert "c2" not in ids  # dominated

    def test_returns_at_most_n(self):
        pool = [Candidate(id=f"c{i}") for i in range(5)]
        scores = {
            f"c{i}": {"mean": 0.5, "per_example": {f"ex_{i}": 1.0}}
            for i in range(5)
        }
        result = top_pareto_candidates(pool, scores, n=2)
        assert len(result) <= 2
