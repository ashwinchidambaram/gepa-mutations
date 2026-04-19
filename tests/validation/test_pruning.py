"""V14: Pruning rule validation."""

from __future__ import annotations

import pytest
from dataclasses import dataclass

from iso_harness.optimizer.candidate import Candidate
from iso_harness.optimizer.pruning import (
    prune_fixed_ratio,
    prune_adaptive_with_floor,
    prune_adaptive_to_regression,
)


# ---------------------------------------------------------------------------
# Minimal stand-ins (avoid importing the full ISOConfig/ISORuntime)
# ---------------------------------------------------------------------------


@dataclass
class _MockConfig:
    """Minimal config stand-in for pruning tests."""

    pool_floor: int = 4

    class hooks:
        prune_ratio = 0.5
        pool_size_max = None
        cross_mutate_only_when_improving = False


@dataclass
class _MockRuntime:
    """Minimal runtime stand-in."""

    round_num: int = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(n: int) -> list[Candidate]:
    return [Candidate(id=f"c{i}") for i in range(n)]


def _make_scores(pool: list[Candidate], descending: bool = True) -> dict[str, dict]:
    """Assign scores 0.9, 0.8, 0.7, … or reversed."""
    scores = {}
    for i, c in enumerate(pool):
        score = (len(pool) - i) / len(pool) if descending else (i + 1) / len(pool)
        scores[c.id] = {"mean": round(score, 3)}
    return scores


def _make_hooks(prune_ratio=0.5):
    return type(
        "Hooks",
        (),
        {
            "prune_ratio": prune_ratio,
            "pool_size_max": None,
            "cross_mutate_only_when_improving": False,
        },
    )()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestV14Pruning:
    """V14: Pruning rules respect floor and remove expected candidates."""

    # ------------------------------------------------------------------ #
    # prune_fixed_ratio                                                    #
    # ------------------------------------------------------------------ #

    def test_fixed_ratio_prunes_half(self):
        """prune_fixed_ratio with ratio=0.5 on 10 candidates keeps 5."""
        pool = _make_pool(10)
        scores = _make_scores(pool)
        config = _MockConfig(pool_floor=4)
        config.hooks = _make_hooks(prune_ratio=0.5)
        runtime = _MockRuntime(round_num=1)

        survivors = prune_fixed_ratio(pool, scores, True, 0.5, 0.6, config, runtime)

        assert len(survivors) == 5

    def test_fixed_ratio_respects_floor(self):
        """Floor prevents over-pruning: 6 * 0.5 = 3 < floor=4, so keep 4."""
        pool = _make_pool(6)
        scores = _make_scores(pool)
        config = _MockConfig(pool_floor=4)
        config.hooks = _make_hooks(prune_ratio=0.5)
        runtime = _MockRuntime(round_num=1)

        survivors = prune_fixed_ratio(pool, scores, True, 0.5, 0.6, config, runtime)

        assert len(survivors) == 4

    def test_fixed_ratio_keeps_best_candidates(self):
        """Survivors are the highest-scoring candidates."""
        pool = _make_pool(10)
        scores = _make_scores(pool, descending=True)  # c0=highest, c9=lowest
        config = _MockConfig(pool_floor=4)
        config.hooks = _make_hooks(prune_ratio=0.5)
        runtime = _MockRuntime(round_num=1)

        survivors = prune_fixed_ratio(pool, scores, True, 0.5, 0.6, config, runtime)

        survivor_ids = {c.id for c in survivors}
        # Top 5 should survive
        assert "c0" in survivor_ids
        assert "c1" in survivor_ids
        # Bottom 5 should be pruned
        assert "c9" not in survivor_ids
        assert "c8" not in survivor_ids

    # ------------------------------------------------------------------ #
    # prune_adaptive_with_floor                                           #
    # ------------------------------------------------------------------ #

    def test_adaptive_with_floor_prunes_25pct(self):
        """prune_adaptive_with_floor removes 25%: 12*0.75=9, floor=8, keep 9."""
        pool = _make_pool(12)
        scores = _make_scores(pool)
        config = _MockConfig(pool_floor=8)
        runtime = _MockRuntime(round_num=1)

        survivors = prune_adaptive_with_floor(
            pool, scores, True, 0.5, 0.6, config, runtime
        )

        assert len(survivors) == 9

    def test_adaptive_with_floor_enforces_floor(self):
        """floor=8 prevents dropping below 8: int(9*0.75)=6 < 8, keep 8."""
        pool = _make_pool(9)
        scores = _make_scores(pool)
        config = _MockConfig(pool_floor=8)
        runtime = _MockRuntime(round_num=1)

        survivors = prune_adaptive_with_floor(
            pool, scores, True, 0.5, 0.6, config, runtime
        )

        assert len(survivors) == 8

    def test_adaptive_with_floor_death_reason(self):
        """Pruned candidates have death_reason='pruned_adaptive_with_floor'."""
        pool = _make_pool(12)
        scores = _make_scores(pool)
        config = _MockConfig(pool_floor=4)
        runtime = _MockRuntime(round_num=2)

        survivors = prune_adaptive_with_floor(
            pool, scores, True, 0.5, 0.6, config, runtime
        )

        survivor_ids = {c.id for c in survivors}
        for c in pool:
            if c.id not in survivor_ids:
                assert c.death_reason == "pruned_adaptive_with_floor"
                assert c.death_round == 2

    # ------------------------------------------------------------------ #
    # prune_adaptive_to_regression                                        #
    # ------------------------------------------------------------------ #

    def test_adaptive_to_regression_prunes_one_when_improving(self):
        """When pool improved, prune exactly 1 candidate."""
        pool = _make_pool(10)
        scores = _make_scores(pool)
        config = _MockConfig(pool_floor=4)
        runtime = _MockRuntime(round_num=2)

        survivors = prune_adaptive_to_regression(
            pool,
            scores,
            pool_improved=True,
            prev_top3_mean=0.7,
            top3_mean=0.8,
            config=config,
            runtime=runtime,
        )

        assert len(survivors) == 9

    def test_adaptive_to_regression_scales_with_regression(self):
        """20% regression prunes 4: regression_pct=0.2, prune=max(1,int(20*0.2))=4, keep 16."""
        pool = _make_pool(20)
        scores = _make_scores(pool)
        config = _MockConfig(pool_floor=4)
        runtime = _MockRuntime(round_num=3)

        survivors = prune_adaptive_to_regression(
            pool,
            scores,
            pool_improved=False,
            prev_top3_mean=0.8,
            top3_mean=0.64,
            config=config,
            runtime=runtime,
        )

        # regression_pct = (0.8 - 0.64) / 0.8 = 0.2
        # prune_count = max(1, int(20 * 0.2)) = 4
        # cap at 20 // 4 = 5, so prune 4
        # keep 20 - 4 = 16
        assert len(survivors) == 16

    def test_first_round_adaptive_prunes_one(self):
        """On round=1 (or prev_top3_mean=0), adaptive prunes exactly 1."""
        pool = _make_pool(10)
        scores = _make_scores(pool)
        config = _MockConfig(pool_floor=4)
        runtime = _MockRuntime(round_num=1)

        survivors = prune_adaptive_to_regression(
            pool,
            scores,
            pool_improved=False,
            prev_top3_mean=0.0,
            top3_mean=0.5,
            config=config,
            runtime=runtime,
        )

        assert len(survivors) == 9

    def test_adaptive_to_regression_death_reason(self):
        """Pruned candidates have death_reason='pruned_adaptive_to_regression'."""
        pool = _make_pool(10)
        scores = _make_scores(pool)
        config = _MockConfig(pool_floor=4)
        runtime = _MockRuntime(round_num=5)

        survivors = prune_adaptive_to_regression(
            pool,
            scores,
            pool_improved=True,
            prev_top3_mean=0.7,
            top3_mean=0.8,
            config=config,
            runtime=runtime,
        )

        survivor_ids = {c.id for c in survivors}
        for c in pool:
            if c.id not in survivor_ids:
                assert c.death_reason == "pruned_adaptive_to_regression"
                assert c.death_round == 5

    # ------------------------------------------------------------------ #
    # Cross-cutting concerns                                              #
    # ------------------------------------------------------------------ #

    def test_pruned_candidates_have_death_metadata(self):
        """All pruned candidates have death_round and death_reason set."""
        pool = _make_pool(10)
        scores = _make_scores(pool)
        config = _MockConfig(pool_floor=4)
        config.hooks = _make_hooks(prune_ratio=0.5)
        runtime = _MockRuntime(round_num=3)

        survivors = prune_fixed_ratio(pool, scores, True, 0.5, 0.6, config, runtime)

        survivor_ids = {c.id for c in survivors}
        for c in pool:
            if c.id not in survivor_ids:
                assert c.death_round == 3
                assert c.death_reason == "pruned_by_fixed_ratio"

    def test_survivors_retain_state(self):
        """Surviving candidates' score_history and prompts remain unchanged."""
        pool = _make_pool(10)
        pool[0].score_history = [(1, 0.9)]
        pool[0].prompts_by_module = {"qa": "test prompt"}
        scores = _make_scores(pool)
        config = _MockConfig(pool_floor=4)
        config.hooks = _make_hooks(prune_ratio=0.5)
        runtime = _MockRuntime(round_num=1)

        survivors = prune_fixed_ratio(pool, scores, True, 0.5, 0.6, config, runtime)

        # c0 has the highest score and must survive
        top = next(c for c in survivors if c.id == "c0")
        assert top.score_history == [(1, 0.9)]
        assert top.prompts_by_module == {"qa": "test prompt"}
        assert top.death_round is None
        assert top.death_reason is None

    def test_survivors_have_no_death_metadata(self):
        """Surviving candidates' death_round and death_reason remain None."""
        pool = _make_pool(8)
        scores = _make_scores(pool)
        config = _MockConfig(pool_floor=4)
        config.hooks = _make_hooks(prune_ratio=0.5)
        runtime = _MockRuntime(round_num=2)

        survivors = prune_fixed_ratio(pool, scores, True, 0.5, 0.6, config, runtime)

        for c in survivors:
            assert c.death_round is None
            assert c.death_reason is None

    def test_floor_larger_than_pool_keeps_all(self):
        """If pool_floor >= pool size, no candidates are pruned."""
        pool = _make_pool(3)
        scores = _make_scores(pool)
        config = _MockConfig(pool_floor=10)
        config.hooks = _make_hooks(prune_ratio=0.5)
        runtime = _MockRuntime(round_num=1)

        survivors = prune_fixed_ratio(pool, scores, True, 0.5, 0.6, config, runtime)

        assert len(survivors) == 3

    def test_adaptive_to_regression_floor_enforced(self):
        """Floor prevents pruning below pool_floor even with heavy regression."""
        pool = _make_pool(10)
        scores = _make_scores(pool)
        config = _MockConfig(pool_floor=9)
        runtime = _MockRuntime(round_num=3)

        survivors = prune_adaptive_to_regression(
            pool,
            scores,
            pool_improved=False,
            prev_top3_mean=0.9,
            top3_mean=0.1,  # massive regression
            config=config,
            runtime=runtime,
        )

        # floor=9 overrides regression-based pruning
        assert len(survivors) >= 9
