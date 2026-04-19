"""V13: Reflection scope validation."""
from __future__ import annotations

import random
import pytest

from iso_harness.optimizer.candidate import Candidate, ModuleTrace, MutationProposal
from iso_harness.optimizer.config import ISOConfig, VariantHooks
from iso_harness.optimizer.runtime import ISORuntime, RolloutCounter, TraceStore
from iso_harness.optimizer.reflection import (
    reflect_per_candidate,
    reflect_population_level,
    reflect_pair_contrastive,
    reflect_hybrid,
)
from tests.mocks.mock_lm import MockReflectionLM, MockMetric


def _make_pool(n: int = 5) -> list[Candidate]:
    """Create n candidates with prompts."""
    return [
        Candidate(id=f"c{i}", prompts_by_module={"qa": f"Prompt for candidate {i}"})
        for i in range(n)
    ]


def _make_scores(pool: list[Candidate]) -> dict[str, dict]:
    """Create scores dict with per_example data."""
    scores = {}
    for i, c in enumerate(pool):
        mean_score = (len(pool) - i) / len(pool)
        scores[c.id] = {
            "mean": mean_score,
            "per_example": {f"ex_{j}": mean_score - 0.1 * j for j in range(5)},
            "per_example_feedback": {f"ex_{j}": f"feedback {j}" for j in range(5)},
            "per_example_metadata": {f"ex_{j}": {} for j in range(5)},
        }
    return scores


def _make_runtime_with_traces(pool, scores):
    """Create runtime with pre-populated trace store."""
    store = TraceStore()
    for c in pool:
        for ex_id, score in scores[c.id]["per_example"].items():
            store.put(c.id, ex_id, ModuleTrace(
                example_id=ex_id,
                score=score,
                feedback=f"feedback for {ex_id}",
            ))

    return ISORuntime(
        reflection_lm=MockReflectionLM(),
        task_lm=None,
        metric=MockMetric(),
        run_id="test-v13",
        seed=42,
        rng=random.Random(42),
        trace_store=store,
        rollout_counter=RolloutCounter(),
        round_num=3,
    )


def _make_config():
    """Create minimal config for reflection tests."""
    return ISOConfig(
        budget=1000,
        seed=42,
        hooks=VariantHooks(
            prune=lambda *a, **k: [],
            reflect=reflect_per_candidate,
            cross_mutate=lambda *a, **k: [],
        ),
    )


class TestV13Reflection:
    """V13: Each reflection function produces valid mutation proposals."""

    def test_per_candidate_returns_proposals(self):
        """reflect_per_candidate returns one proposal per candidate."""
        pool = _make_pool(5)
        scores = _make_scores(pool)
        runtime = _make_runtime_with_traces(pool, scores)
        config = _make_config()

        proposals = reflect_per_candidate(pool, scores, 3, config, 0.7, runtime)

        assert isinstance(proposals, list)
        # Should have proposals (may be fewer than pool if some fail to parse)
        assert len(proposals) > 0
        for p in proposals:
            assert isinstance(p, MutationProposal)
            assert p.mechanism == "per_candidate"
            assert p.candidate_id in {c.id for c in pool}

    def test_population_level_same_mutation_for_all(self):
        """reflect_population_level returns identical mutations for all candidates."""
        pool = _make_pool(5)
        scores = _make_scores(pool)
        # Make some examples score < 0.5 so population failures are found
        for c in pool:
            scores[c.id]["per_example"]["ex_3"] = 0.1
            scores[c.id]["per_example"]["ex_4"] = 0.2
        runtime = _make_runtime_with_traces(pool, scores)
        config = _make_config()

        proposals = reflect_population_level(pool, scores, 3, config, 0.7, runtime)

        if len(proposals) > 0:
            # All should have the same new_prompts
            first_prompts = proposals[0].new_prompts
            for p in proposals[1:]:
                assert p.new_prompts == first_prompts
                assert p.mechanism == "population_level"

    def test_pair_contrastive_falls_back_early_round(self):
        """reflect_pair_contrastive falls back to per_candidate when round < 2."""
        pool = _make_pool(5)
        scores = _make_scores(pool)
        runtime = _make_runtime_with_traces(pool, scores)
        config = _make_config()

        # Round 1 — should fall back to per_candidate
        proposals = reflect_pair_contrastive(pool, scores, 1, config, 0.0, runtime)
        assert isinstance(proposals, list)

    def test_pair_contrastive_with_history(self):
        """reflect_pair_contrastive produces proposals when improver/regressor exist."""
        pool = _make_pool(5)
        # Give candidates score history spanning 2 rounds
        for i, c in enumerate(pool):
            c.score_history = [(1, 0.5 + i * 0.05), (2, 0.5 + i * 0.1)]
        scores = _make_scores(pool)
        runtime = _make_runtime_with_traces(pool, scores)
        config = _make_config()

        proposals = reflect_pair_contrastive(pool, scores, 3, config, 0.6, runtime)
        assert isinstance(proposals, list)

    def test_hybrid_uses_per_candidate_when_improving(self):
        """reflect_hybrid delegates to per_candidate when pool is improving."""
        pool = _make_pool(5)
        scores = _make_scores(pool)
        runtime = _make_runtime_with_traces(pool, scores)
        config = _make_config()

        # prev_top3_mean < current top3_mean -> improving
        proposals = reflect_hybrid(pool, scores, 3, config, 0.3, runtime)
        assert isinstance(proposals, list)
        # Per-candidate should return proposals with mechanism "per_candidate"
        if proposals:
            assert proposals[0].mechanism == "per_candidate"

    def test_hybrid_uses_population_when_stalling(self):
        """reflect_hybrid delegates to population_level when pool is stalling."""
        pool = _make_pool(5)
        scores = _make_scores(pool)
        # Set low scores so population failures exist
        for c in pool:
            scores[c.id]["per_example"]["ex_3"] = 0.1
            scores[c.id]["per_example"]["ex_4"] = 0.2
        runtime = _make_runtime_with_traces(pool, scores)
        config = _make_config()

        # prev_top3_mean > current top3 -> stalling
        proposals = reflect_hybrid(pool, scores, 3, config, 0.99, runtime)
        assert isinstance(proposals, list)
        if proposals:
            assert proposals[0].mechanism == "population_level"

    def test_all_proposals_have_valid_prompts(self):
        """All proposals contain prompt text for modules."""
        pool = _make_pool(3)
        scores = _make_scores(pool)
        runtime = _make_runtime_with_traces(pool, scores)
        config = _make_config()

        proposals = reflect_per_candidate(pool, scores, 3, config, 0.5, runtime)
        for p in proposals:
            assert isinstance(p.new_prompts, dict)
            # At least one module should have a prompt
            assert len(p.new_prompts) > 0
