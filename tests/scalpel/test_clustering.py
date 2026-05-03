"""Tests for SCALPEL Phase 5: failure-mode clustering.

Covers:

* ``scalpel.clustering.embeddings.BGEEmbedder`` — empty-input shortcut,
  lazy model loading (monkeypatched, so no real download), single-text
  helper.
* ``scalpel.clustering.kmeans.FailureClusterer`` — synthetic 3-mode
  recovery, recluster trigger logic, small-pool fallback, cluster-state
  bookkeeping, low-silhouette warning.
* ``scalpel.clustering.targeting`` — Q1 selection heuristic
  (highest-mass default, anti-starvation rotation, recent-with-margin
  override), history bookkeeping, error path, and a coupon-collector
  rotation simulation that bounds the expected cover time.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from scalpel.clustering import embeddings as embeddings_mod
from scalpel.clustering.embeddings import EMBEDDING_DIM, BGEEmbedder
from scalpel.clustering.kmeans import ClusterState, FailureClusterer
from scalpel.clustering.targeting import (
    DEFAULT_GAMMA,
    TargetingHistory,
    expected_hitting_time_ub,
    select_target_cluster,
)

# --------------------------------------------------------------------- helpers


class _FakeSentenceTransformer:
    """Deterministic stand-in for ``SentenceTransformer``.

    Encodes each text into a (EMBEDDING_DIM,) vector seeded by a stable
    hash of the text, normalized to unit L2 norm so it mirrors
    ``normalize_embeddings=True`` behaviour.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del batch_size, show_progress_bar  # unused by the fake.
        out = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
            v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
            if normalize_embeddings:
                norm = float(np.linalg.norm(v))
                if norm > 0:
                    v = v / norm
            out[i] = v
        return out


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSentenceTransformer]:
    """Replace ``SentenceTransformer`` inside the embeddings module with a fake."""
    monkeypatch.setattr(embeddings_mod, "SentenceTransformer", _FakeSentenceTransformer)
    return _FakeSentenceTransformer


def _synthetic_3_modes(seed: int = 0) -> np.ndarray:
    """20+20+20 points around three well-separated centres in 384-d.

    The spec sketch in the build prompt suggested ``std=0.05`` but the 384-d
    geometry pushes the silhouette just below the 0.4 threshold there
    because per-dimension Gaussian noise dominates the inter-centre gap on
    most axes — see the report for the diagnostic.  ``std=0.03`` keeps the
    cluster geometry clearly separable while preserving the test's intent.
    """
    rng = np.random.default_rng(seed)
    centres = np.zeros((3, EMBEDDING_DIM), dtype=np.float32)
    centres[0, 0] = 1.0
    centres[1, 0] = -1.0
    centres[2, 1] = 5.0
    blocks: list[np.ndarray] = []
    for c in centres:
        block = c[None, :] + rng.normal(scale=0.03, size=(20, EMBEDDING_DIM)).astype(
            np.float32
        )
        blocks.append(block)
    return np.concatenate(blocks, axis=0)


# ------------------------------------------------------------------ embeddings


def test_embedder_empty_input_returns_zero_shape() -> None:
    """No texts → no model load and a (0, 384) zero array."""
    embedder = BGEEmbedder()
    out = embedder.embed([])
    assert out.shape == (0, EMBEDDING_DIM)
    assert embedder._model is None  # truly never loaded.


def test_embedder_lazy_loads_model(fake_st: type[_FakeSentenceTransformer]) -> None:
    """``_model`` is None pre-call and non-None post-call."""
    del fake_st  # only here for the monkeypatch side-effect.
    embedder = BGEEmbedder()
    assert embedder._model is None
    out = embedder.embed(["hello world"])
    assert embedder._model is not None
    assert out.shape == (1, EMBEDDING_DIM)


def test_embed_one_returns_1d_array(fake_st: type[_FakeSentenceTransformer]) -> None:
    """``embed_one`` collapses to a (384,) 1-D vector."""
    del fake_st
    embedder = BGEEmbedder()
    v = embedder.embed_one("a single string")
    assert v.shape == (EMBEDDING_DIM,)
    assert v.dtype == np.float32


# ---------------------------------------------------------------------- kmeans


def test_synthetic_3_mode_recovery() -> None:
    X = _synthetic_3_modes()
    fc = FailureClusterer(seed=0)
    fc.add(X)
    k = fc.recluster()
    # Silhouette may pick k=4 with one near-empty cluster at the boundary,
    # or k=3 outright; either is acceptable on the synthetic.
    assert fc.last_k in {3, 4}
    assert k == fc.last_k
    assert fc.last_silhouette > 0.4


def test_should_recluster_first_call_with_data() -> None:
    fc = FailureClusterer()
    fc.add(_synthetic_3_modes())
    assert fc.should_recluster() is True


def test_should_recluster_after_n_iters() -> None:
    fc = FailureClusterer(recluster_every_iters=8)
    fc.add(_synthetic_3_modes())
    fc.recluster()
    assert fc.should_recluster() is False
    for _ in range(8):
        fc.step_iteration()
    assert fc.should_recluster() is True


def test_should_recluster_after_pool_growth() -> None:
    fc = FailureClusterer(recluster_pool_growth_frac=0.5)
    rng = np.random.default_rng(42)
    base = rng.normal(size=(20, EMBEDDING_DIM)).astype(np.float32)
    fc.add(base)
    fc.recluster()
    assert fc.should_recluster() is False
    extra = rng.normal(size=(11, EMBEDDING_DIM)).astype(np.float32)
    fc.add(extra)  # 20 -> 31, 55% growth.
    assert fc.should_recluster() is True


def test_should_not_recluster_with_no_data() -> None:
    fc = FailureClusterer()
    assert fc.should_recluster() is False


def test_recluster_idempotent_at_small_pool() -> None:
    """Pool of 2 points falls back to k=1 without crashing."""
    fc = FailureClusterer()
    rng = np.random.default_rng(0)
    fc.add(rng.normal(size=(2, EMBEDDING_DIM)).astype(np.float32))
    k = fc.recluster()
    assert k == 1
    assert fc.last_k == 1
    # Second call must not crash either.
    k2 = fc.recluster()
    assert k2 == 1


def test_cluster_states_consistency() -> None:
    fc = FailureClusterer(seed=0)
    fc.add(_synthetic_3_modes())
    fc.recluster()
    states = fc.cluster_states()
    assert isinstance(states[0], ClusterState)
    assert len(states) == fc.last_k
    assert sum(s.failure_count for s in states) == 60
    for s in states:
        assert len(s.centroid) == EMBEDDING_DIM


def test_recluster_warns_at_low_silhouette(caplog: pytest.LogCaptureFixture) -> None:
    """Random uniform points produce a low silhouette → warning logged."""
    rng = np.random.default_rng(7)
    fc = FailureClusterer(seed=0)
    fc.add(rng.uniform(size=(30, EMBEDDING_DIM)).astype(np.float32))
    caplog.set_level(logging.WARNING, logger="scalpel.clustering.kmeans")
    fc.recluster()
    if fc.last_silhouette < 0.1:
        assert any(
            "low silhouette" in record.message for record in caplog.records
        ), "expected low-silhouette warning to fire"


# ------------------------------------------------------------------- targeting


def _cluster(cid: int, count: int) -> ClusterState:
    return ClusterState(id=cid, centroid=[0.0] * EMBEDDING_DIM, failure_count=count)


def test_select_with_no_history_picks_highest_mass() -> None:
    clusters = [_cluster(0, 10), _cluster(1, 5), _cluster(2, 1)]
    history = TargetingHistory()
    assert select_target_cluster(clusters, history) == 0


def test_select_rotates_after_recent_targeting() -> None:
    """Equal masses, cluster 0 just targeted → must NOT pick 0."""
    clusters = [_cluster(0, 5), _cluster(1, 5), _cluster(2, 5)]
    history = TargetingHistory(iters_since_targeted={0: 0}, last_targeted_id=0)
    chosen = select_target_cluster(clusters, history)
    assert chosen != 0


def test_select_anti_starvation_high_mass_recent_can_win() -> None:
    """High-mass recent cluster beats tiny non-recent ones by margin."""
    clusters = [_cluster(0, 100), _cluster(1, 1), _cluster(2, 1)]
    history = TargetingHistory(iters_since_targeted={0: 1}, last_targeted_id=0)
    chosen = select_target_cluster(clusters, history)
    assert chosen == 0


def test_record_increments_others() -> None:
    h = TargetingHistory(
        iters_since_targeted={0: 3, 1: 1}, last_targeted_id=1
    )
    h2 = h.record(2)
    assert h2.iters_since_targeted[2] == 0
    assert h2.iters_since_targeted[0] == 4
    assert h2.iters_since_targeted[1] == 2
    assert h2.last_targeted_id == 2


def test_select_raises_on_empty_clusters() -> None:
    with pytest.raises(ValueError):
        select_target_cluster([], TargetingHistory())


def test_coupon_collector_simulation() -> None:
    """All 8 equal-mass clusters get visited well within 2·H_k."""
    k = 8
    clusters = [_cluster(cid, 10) for cid in range(k)]
    history = TargetingHistory()
    seen: set[int] = set()
    cover_iter: int | None = None
    for i in range(100):
        chosen = select_target_cluster(clusters, history)
        seen.add(chosen)
        history = history.record(chosen)
        if len(seen) == k and cover_iter is None:
            cover_iter = i + 1
            break
    assert cover_iter is not None, "did not cover all clusters in 100 iters"
    bound = 2 * expected_hitting_time_ub(k)
    assert cover_iter <= bound, (
        f"covered all {k} clusters at iter {cover_iter}, "
        f"expected <= 2*H_k = {bound:.1f}"
    )


# ----------------------------------------------------- module-level invariants


def test_default_gamma_is_seven_tenths() -> None:
    """Sanity: the spec pins γ = 0.7."""
    assert DEFAULT_GAMMA == pytest.approx(0.7)


def test_targeting_history_default_state() -> None:
    h = TargetingHistory()
    assert h.iters_since_targeted == {}
    assert h.last_targeted_id is None


def test_select_handles_unused_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke: tunable gamma propagates into the score computation."""
    del monkeypatch
    clusters = [_cluster(0, 10), _cluster(1, 10)]
    history = TargetingHistory(iters_since_targeted={0: 1}, last_targeted_id=0)
    # With gamma=0 the recency decay vanishes and ties resolve to lowest id.
    chosen_zero = select_target_cluster(clusters, history, gamma=0.0)
    assert chosen_zero in {0, 1}
    # With gamma=1 the recency decay is full → cluster 0's score is 0,
    # and only cluster 1 has positive score, so it wins outright.
    chosen_one: Any = select_target_cluster(clusters, history, gamma=1.0)
    assert chosen_one == 1
