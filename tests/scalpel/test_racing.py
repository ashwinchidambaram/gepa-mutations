"""Tests for SCALPEL Phase 6 - successive halving + cluster-stratified sampling."""

from __future__ import annotations

import random
from typing import Any

import pytest

from scalpel.clustering.kmeans import ClusterState
from scalpel.racing.rungs import DEFAULT_ETA, DEFAULT_RUNGS, halve
from scalpel.racing.successive_halving import (
    FailurePoolStratifier,
    RaceResult,
    RungLog,
    StratifierLike,
    SuccessiveHalving,
    SurrogateLike,
    _NoopSurrogate,
)

# --------------------------------------------------------------------- rungs


def test_default_rungs_match_spec() -> None:
    assert DEFAULT_RUNGS == (8, 16, 32, 64)
    assert DEFAULT_ETA == 2


def test_halve_basic() -> None:
    assert halve(8) == 4
    assert halve(4) == 2
    assert halve(2) == 1
    assert halve(1) == 1
    assert halve(0) == 0


def test_halve_with_eta_3() -> None:
    assert halve(9, eta=3) == 3
    assert halve(2, eta=3) == 1


# --------------------------------------------------------------- stratifier


def _mk_pool(n: int, prefix: str = "i") -> list[dict[str, Any]]:
    return [{"id": f"{prefix}{i}"} for i in range(n)]


def test_stratifier_uniform_when_no_clusters() -> None:
    pool = _mk_pool(100)
    strat = FailurePoolStratifier(seed=0)
    out = strat.sample(pool, n=10)
    assert len(out) == 10
    ids = [p["id"] for p in out]
    assert len(set(ids)) == 10


def test_stratifier_sqrt_tempering() -> None:
    """Failure counts (16, 4, 1) -> sqrt (4, 2, 1) -> shares (4/7, 2/7, 1/7).

    Pool: 50 items in cluster 0, 25 in cluster 1, 15 in cluster 2 (sized so
    quotas of 40/20/10 fit without with-replacement fallback).  Aggregating
    many draws should track the (4/7, 2/7, 1/7) split closely.
    """
    clusters = [
        ClusterState(id=0, centroid=[0.0] * 384, failure_count=16),
        ClusterState(id=1, centroid=[0.0] * 384, failure_count=4),
        ClusterState(id=2, centroid=[0.0] * 384, failure_count=1),
    ]
    pool: list[dict[str, Any]] = []
    cluster_assignments: dict[str, int] = {}
    for cid, count in [(0, 50), (1, 25), (2, 15)]:
        for j in range(count):
            iid = f"c{cid}_i{j}"
            pool.append({"id": iid})
            cluster_assignments[iid] = cid

    counts = [0, 0, 0]
    n_trials = 50
    for trial in range(n_trials):
        strat = FailurePoolStratifier(
            clusters=clusters,
            cluster_assignments=cluster_assignments,
            seed=trial,
        )
        out = strat.sample(pool, n=70)
        assert len(out) == 70
        for item in out:
            cid = cluster_assignments[item["id"]]
            counts[cid] += 1

    total = sum(counts)
    expected = [total * 4 / 7, total * 2 / 7, total * 1 / 7]
    for got, exp in zip(counts, expected, strict=True):
        assert abs(got - exp) < 0.05 * total, (counts, expected)


def test_stratifier_falls_back_to_replacement_when_quota_exceeds_cluster_size() -> None:
    """One item in cluster 0 with quota=5: should return 5 items (with repeats)."""
    clusters = [ClusterState(id=0, centroid=[0.0] * 384, failure_count=1)]
    pool = [{"id": "only"}]
    cluster_assignments = {"only": 0}
    strat = FailurePoolStratifier(
        clusters=clusters,
        cluster_assignments=cluster_assignments,
        seed=0,
    )
    out = strat.sample(pool, n=5)
    assert len(out) == 5
    assert all(p["id"] == "only" for p in out)


# ------------------------------------------------------- SuccessiveHalving


def _mk_score_fn(winner: str, win_score: float = 0.9, lose_score: float = 0.1):
    """Return a deterministic scoring callable for race tests."""

    def score_fn(cid: str, inst: Any) -> float:
        rng = random.Random(hash((cid, str(inst.get("id")))) & 0xFFFF)
        if cid == winner:
            return win_score + rng.uniform(-0.01, 0.01)
        return lose_score + rng.uniform(-0.01, 0.01)

    return score_fn


def test_race_with_clear_winners_picks_top() -> None:
    candidates = [f"c{i}" for i in range(8)]
    pool = _mk_pool(2000)
    score_fn = _mk_score_fn("c0")

    n_correct = 0
    n_trials = 100
    for seed in range(n_trials):
        sh = SuccessiveHalving(rng_seed=seed)
        result = sh.race(candidates, score_fn, pool)
        if result.survivor_id == "c0":
            n_correct += 1
    assert n_correct >= 95, f"survivor was c0 only {n_correct}/100 trials"


def test_race_total_rollouts_in_expected_band() -> None:
    candidates = [f"c{i}" for i in range(8)]
    pool = _mk_pool(2000)
    score_fn = _mk_score_fn("c0")
    sh = SuccessiveHalving(rng_seed=42)
    result = sh.race(candidates, score_fn, pool)
    # 8*8 + 4*16 + 2*32 + 1*64 = 256 with no surrogate skips.
    assert result.total_rollouts == 256
    assert result.total_skipped == 0


class _MockSurrogate:
    def __init__(self, skip_rate: float, seed: int = 0) -> None:
        self.skip_rate = skip_rate
        self._rng = random.Random(seed)

    def should_skip(self, candidate_id: str, instance: Any) -> bool:
        return self._rng.random() < self.skip_rate

    def predict(self, candidate_id: str, instance: Any) -> float:
        return 0.5


def test_race_with_surrogate_skip_30_percent() -> None:
    candidates = [f"c{i}" for i in range(8)]
    pool = _mk_pool(2000)
    score_fn = _mk_score_fn("c0")
    surrogate = _MockSurrogate(skip_rate=0.30, seed=0)
    sh = SuccessiveHalving(rng_seed=42, surrogate=surrogate)
    result = sh.race(candidates, score_fn, pool)
    assert result.total_rollouts + result.total_skipped == 256
    # Per addendum Q1/Q2: surrogate cuts ~30%, so net rollouts land ~226;
    # accept [180, 256] for RNG variance across rungs.
    assert 180 <= result.total_rollouts <= 256, result.total_rollouts


def test_race_with_one_candidate_runs_rung_0_only() -> None:
    pool = _mk_pool(2000)
    score_fn = _mk_score_fn("only_one")
    sh = SuccessiveHalving(rng_seed=0)
    result = sh.race(["only_one"], score_fn, pool)
    assert result.survivor_id == "only_one"
    assert result.total_rollouts == 8
    assert result.total_skipped == 0
    assert len(result.rung_logs) == 1
    log = result.rung_logs[0]
    assert log.rung_index == 0
    assert log.n_alive_before == 1
    assert log.n_alive_after == 1


def test_race_raises_on_empty_candidates() -> None:
    sh = SuccessiveHalving()
    with pytest.raises(ValueError, match="no candidates"):
        sh.race([], lambda c, i: 0.0, _mk_pool(10))


def test_race_result_rung_logs_correct_lengths() -> None:
    candidates = [f"c{i}" for i in range(8)]
    pool = _mk_pool(2000)
    score_fn = _mk_score_fn("c0")
    sh = SuccessiveHalving(rng_seed=0)
    result = sh.race(candidates, score_fn, pool)
    # Default rungs: 8 -> 4 -> 2 -> 1.
    assert len(result.rung_logs) == 4
    expected_alive = [(8, 4), (4, 2), (2, 1), (1, 1)]
    for log, (before, after) in zip(result.rung_logs, expected_alive, strict=True):
        assert log.n_alive_before == before
        assert log.n_alive_after == after


def test_score_samples_accumulate_across_rungs() -> None:
    """Survivor accumulates samples across all rungs: 8 + 16 + 32 + 64 = 120."""
    candidates = [f"c{i}" for i in range(8)]
    pool = _mk_pool(2000)
    call_count = {"c0": 0}

    def counting_score(cid: str, inst: Any) -> float:
        if cid == "c0":
            call_count["c0"] += 1
            return 0.9
        return 0.1

    sh = SuccessiveHalving(rng_seed=0)
    result = sh.race(candidates, counting_score, pool)
    assert result.survivor_id == "c0"
    assert call_count["c0"] == 120


def test_surrogate_protocol_runtime_check() -> None:
    assert isinstance(_NoopSurrogate(), SurrogateLike)


def test_stratifier_protocol_runtime_check() -> None:
    assert isinstance(FailurePoolStratifier(), StratifierLike)


# --------------------------------------------------------------- type hygiene


def test_pydantic_models_round_trip() -> None:
    log = RungLog(
        rung_index=0,
        rung_size=8,
        n_alive_before=8,
        n_alive_after=4,
        rollouts_run=64,
        rollouts_skipped=0,
        survivor_ids=["c0", "c1", "c2", "c3"],
    )
    result = RaceResult(
        survivor_id="c0",
        survivor_score=0.9,
        rung_logs=[log],
        total_rollouts=64,
        total_skipped=0,
    )
    dumped = result.model_dump()
    restored = RaceResult.model_validate(dumped)
    assert restored.survivor_id == "c0"
    assert len(restored.rung_logs) == 1
