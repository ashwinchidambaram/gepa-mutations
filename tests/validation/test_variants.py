"""V12: Variant isolation tests."""
from __future__ import annotations

import json
import random
import pytest

from iso_harness.optimizer.candidate import Candidate, ModuleTrace, MutationProposal
from iso_harness.optimizer.config import ISOConfig, VariantHooks
from iso_harness.optimizer.runtime import ISORuntime, RolloutCounter, TraceStore
from iso_harness.optimizer.variants import (
    iso_sprint_config,
    iso_grove_config,
    iso_tide_config,
    iso_lens_config,
    iso_storm_config,
)
from tests.mocks.mock_lm import MockReflectionLM, MockMetric


_BASE = {"budget": 1000, "seed": 42}
_FACTORIES = {
    "sprint": iso_sprint_config,
    "grove": iso_grove_config,
    "tide": iso_tide_config,
    "lens": iso_lens_config,
    "storm": iso_storm_config,
}


def _make_pool(n: int = 8) -> list[Candidate]:
    return [
        Candidate(
            id=f"c{i}",
            prompts_by_module={"qa": f"Prompt {i}"},
            score_history=[(1, 0.5 + i * 0.05), (2, 0.5 + i * 0.04)],
        )
        for i in range(n)
    ]


def _make_scores(pool: list[Candidate]) -> dict[str, dict]:
    scores = {}
    for i, c in enumerate(pool):
        mean_score = (len(pool) - i) / len(pool)
        scores[c.id] = {
            "mean": round(mean_score, 3),
            "per_example": {f"ex_{j}": max(0, mean_score - 0.1 * j) for j in range(5)},
            "per_example_feedback": {f"ex_{j}": "feedback" for j in range(5)},
            "per_example_metadata": {f"ex_{j}": {} for j in range(5)},
        }
    return scores


def _make_runtime(pool=None, scores=None):
    store = TraceStore()
    if pool and scores:
        for c in pool:
            for ex_id in scores.get(c.id, {}).get("per_example", {}):
                store.put(c.id, ex_id, ModuleTrace(
                    example_id=ex_id,
                    score=scores[c.id]["per_example"][ex_id],
                    feedback="test feedback",
                ))
    return ISORuntime(
        reflection_lm=MockReflectionLM(),
        task_lm=None,
        metric=MockMetric(),
        run_id="test-v12",
        seed=42,
        rng=random.Random(42),
        trace_store=store,
        rollout_counter=RolloutCounter(),
        round_num=2,
    )


class TestV12VariantIsolation:
    """V12: Each variant's hook functions execute without error in isolation."""

    @pytest.mark.parametrize("name,factory", list(_FACTORIES.items()))
    def test_factory_returns_valid_config(self, name, factory):
        """Config factory returns valid ISOConfig."""
        config = factory(dict(_BASE))
        assert isinstance(config, ISOConfig)
        assert config.budget == 1000
        assert config.seed == 42
        assert isinstance(config.hooks, VariantHooks)

    @pytest.mark.parametrize("name,factory", list(_FACTORIES.items()))
    def test_hooks_are_callable(self, name, factory):
        """All hook fields are callable."""
        config = factory(dict(_BASE))
        assert callable(config.hooks.prune)
        assert callable(config.hooks.reflect)
        assert callable(config.hooks.cross_mutate)

    @pytest.mark.parametrize("name,factory", list(_FACTORIES.items()))
    def test_prune_hook_returns_list(self, name, factory):
        """Prune hook returns a list shorter than or equal to input."""
        config = factory(dict(_BASE))
        pool = _make_pool(10)
        scores = _make_scores(pool)
        runtime = _make_runtime(pool, scores)

        survivors = config.hooks.prune(
            pool, scores, True, 0.5, 0.6, config, runtime,
        )
        assert isinstance(survivors, list)
        assert len(survivors) <= len(pool)
        assert len(survivors) >= config.pool_floor

    @pytest.mark.parametrize("name,factory", list(_FACTORIES.items()))
    def test_reflect_hook_returns_proposals(self, name, factory):
        """Reflect hook returns list of MutationProposal."""
        config = factory(dict(_BASE))
        pool = _make_pool(6)
        scores = _make_scores(pool)
        runtime = _make_runtime(pool, scores)

        proposals = config.hooks.reflect(
            pool, scores, 2, config, 0.5, runtime,
        )
        assert isinstance(proposals, list)
        for p in proposals:
            assert isinstance(p, MutationProposal)

    @pytest.mark.parametrize("name,factory", list(_FACTORIES.items()))
    def test_cross_mutate_hook_returns_candidates(self, name, factory):
        """Cross-mutate hook returns list of Candidate children."""
        config = factory(dict(_BASE))
        pool = _make_pool(8)
        scores = _make_scores(pool)

        # For Storm variant: the reflector-guided cross calls the mock LM with
        # "complementary"/"pairs" prompts and parses pair IDs. The mock returns
        # placeholder IDs that don't exist in the pool, so it falls back to
        # elitist cross-mutation.  We pre-register a valid pair response so the
        # reflector-guided path also produces children with correct IDs.
        mock_lm = MockReflectionLM()
        pool_ids = [c.id for c in pool]
        mock_lm.add_response(
            "complementary",
            json.dumps({
                "pairs": [
                    {
                        "parent_a_id": pool_ids[0],
                        "parent_b_id": pool_ids[1],
                        "rationale": "combine best reasoning strategies",
                    }
                ]
            }),
        )

        store = TraceStore()
        for c in pool:
            for ex_id in scores.get(c.id, {}).get("per_example", {}):
                store.put(c.id, ex_id, ModuleTrace(
                    example_id=ex_id,
                    score=scores[c.id]["per_example"][ex_id],
                    feedback="test feedback",
                ))
        runtime = ISORuntime(
            reflection_lm=mock_lm,
            task_lm=None,
            metric=MockMetric(),
            run_id="test-v12",
            seed=42,
            rng=random.Random(42),
            trace_store=store,
            rollout_counter=RolloutCounter(),
            round_num=2,
        )

        children = config.hooks.cross_mutate(
            pool, scores, True, config, runtime,
        )
        assert isinstance(children, list)
        for child in children:
            assert isinstance(child, Candidate)
            assert child.id  # Has a valid ID
            assert len(child.parent_ids) > 0  # Has parents

    def test_sprint_specific(self):
        """Sprint: prune_ratio=0.5, population-level reflection."""
        config = iso_sprint_config(dict(_BASE))
        assert config.pool_floor == 4
        assert config.hooks.prune_ratio == 0.5
        assert config.hooks.prune.__name__ == "prune_fixed_ratio"
        assert config.hooks.reflect.__name__ == "reflect_population_level"

    def test_grove_specific(self):
        """Grove: pool_floor=8, exploration cross."""
        config = iso_grove_config(dict(_BASE))
        assert config.pool_floor == 8
        assert config.hooks.cross_mutate.__name__ == "cross_mutate_exploration_preserving"

    def test_tide_specific(self):
        """Tide: cross_mutate_only_when_improving=True."""
        config = iso_tide_config(dict(_BASE))
        assert config.pool_floor == 6
        assert config.hooks.cross_mutate_only_when_improving is True

    def test_storm_specific(self):
        """Storm: pool_size_max=4, reflector-guided cross."""
        config = iso_storm_config(dict(_BASE))
        assert config.hooks.pool_size_max == 4
        assert config.hooks.cross_mutate.__name__ == "cross_mutate_reflector_guided"
        assert config.hooks.reflect.__name__ == "reflect_hybrid"

    def test_lens_specific(self):
        """Lens: pair-contrastive reflection."""
        config = iso_lens_config(dict(_BASE))
        assert config.hooks.reflect.__name__ == "reflect_pair_contrastive"
        assert config.hooks.cross_mutate_only_when_improving is True

    def test_tide_cross_mutate_suppressed_when_not_improving(self):
        """Tide: cross_mutate returns empty list when pool_improved=False."""
        config = iso_tide_config(dict(_BASE))
        pool = _make_pool(8)
        scores = _make_scores(pool)
        runtime = _make_runtime(pool, scores)

        children = config.hooks.cross_mutate(
            pool, scores, False, config, runtime,
        )
        assert children == []

    def test_lens_cross_mutate_suppressed_when_not_improving(self):
        """Lens: cross_mutate returns empty list when pool_improved=False."""
        config = iso_lens_config(dict(_BASE))
        pool = _make_pool(8)
        scores = _make_scores(pool)
        runtime = _make_runtime(pool, scores)

        children = config.hooks.cross_mutate(
            pool, scores, False, config, runtime,
        )
        assert children == []

    def test_storm_cross_mutate_active_when_improving(self):
        """Storm: cross_mutate produces children (no improving-only gate)."""
        config = iso_storm_config(dict(_BASE))
        pool = _make_pool(8)
        scores = _make_scores(pool)

        mock_lm = MockReflectionLM()
        pool_ids = [c.id for c in pool]
        mock_lm.add_response(
            "complementary",
            json.dumps({
                "pairs": [
                    {
                        "parent_a_id": pool_ids[0],
                        "parent_b_id": pool_ids[1],
                        "rationale": "pair for cross",
                    }
                ]
            }),
        )

        store = TraceStore()
        for c in pool:
            for ex_id in scores.get(c.id, {}).get("per_example", {}):
                store.put(c.id, ex_id, ModuleTrace(
                    example_id=ex_id,
                    score=scores[c.id]["per_example"][ex_id],
                    feedback="test feedback",
                ))
        runtime = ISORuntime(
            reflection_lm=mock_lm,
            task_lm=None,
            metric=MockMetric(),
            run_id="test-v12",
            seed=42,
            rng=random.Random(42),
            trace_store=store,
            rollout_counter=RolloutCounter(),
            round_num=2,
        )

        children = config.hooks.cross_mutate(
            pool, scores, False, config, runtime,
        )
        # Storm has no improving-only gate, so should produce children
        assert isinstance(children, list)
        assert len(children) > 0

    def test_grove_prune_floor_is_8(self):
        """Grove: pool never drops below floor=8 after pruning."""
        config = iso_grove_config(dict(_BASE))
        pool = _make_pool(9)
        scores = _make_scores(pool)
        runtime = _make_runtime(pool, scores)

        survivors = config.hooks.prune(
            pool, scores, True, 0.5, 0.6, config, runtime,
        )
        assert len(survivors) >= 8

    def test_all_variants_base_config_passthrough(self):
        """Extra base_config fields are preserved in the resulting ISOConfig."""
        base = {"budget": 500, "seed": 7, "max_rounds": 5}
        for name, factory in _FACTORIES.items():
            config = factory(dict(base))
            assert config.budget == 500, f"{name}: budget mismatch"
            assert config.seed == 7, f"{name}: seed mismatch"
            assert config.max_rounds == 5, f"{name}: max_rounds mismatch"
