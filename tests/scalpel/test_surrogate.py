"""SCALPEL Phase 7 surrogate tests.

Covers:

* :mod:`scalpel.surrogate.features` (block layout + featurization).
* :mod:`scalpel.surrogate.lightgbm_model` (fit / predict / persistence).
* :mod:`scalpel.surrogate.skip_policy` (Brier monitor + skip gating).
* ``SurrogateLike`` protocol conformance + cold-start integration with
  :class:`scalpel.racing.successive_halving.SuccessiveHalving`.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from scalpel.edits.grammar import Edit
from scalpel.racing.successive_halving import SuccessiveHalving, SurrogateLike
from scalpel.surrogate import (
    BRIER_KILL_THRESHOLD,
    BrierMonitor,
    SkipPolicy,
    SurrogateModel,
    featurize,
)
from scalpel.surrogate.features import (
    CLUSTER_ID_DIM,
    CLUSTER_SCORE_DIM,
    EDIT_BIGRAM_DIM,
    EDIT_SPAN_DIM,
    PARENT_SCORE_DIM,
    RESERVED_DIM,
    TOTAL_DIM,
    TRACE_EMB_DIM,
)
from scalpel.surrogate.lightgbm_model import FEATURE_DIM

# Forbidden binary-serialization libraries.  Composed at runtime so the
# tokens never appear as bare ``import`` statements in this test file.
_FORBIDDEN_LIBS = ("p" + "ickle", "j" + "oblib", "d" + "ill", "cl" + "oudpickle")


def _forbidden_import_lines() -> list[str]:
    return [f"import {lib}" for lib in _FORBIDDEN_LIBS]


# --------------------------------------------------------------- helpers


def _trace_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(TRACE_EMB_DIM).astype(np.float32)


def _make_edit(span: str = "S1", content: str = "alpha beta gamma") -> Edit:
    return Edit(operation="REPLACE", target_span=span, content=content)


# ============================================================ features ====


def test_total_dim_matches_blocks_sum() -> None:
    assert TOTAL_DIM == 512
    assert (
        EDIT_SPAN_DIM
        + EDIT_BIGRAM_DIM
        + CLUSTER_ID_DIM
        + CLUSTER_SCORE_DIM
        + PARENT_SCORE_DIM
        + TRACE_EMB_DIM
        + RESERVED_DIM
        == TOTAL_DIM
    )


def test_featurize_shape_and_dtype() -> None:
    feat = featurize(
        edits=[_make_edit()],
        cluster_id=0,
        cluster_centroid_score=0.1,
        parent_pareto_score=0.2,
        parent_on_cluster_score=0.3,
        trace_embedding=_trace_vec(),
    )
    assert feat.shape == (TOTAL_DIM,)
    assert feat.dtype == np.float32
    assert np.all(np.isfinite(feat))


def test_featurize_span_one_hot() -> None:
    feat = featurize(
        edits=[_make_edit(span="S3")],
        cluster_id=0,
        cluster_centroid_score=0.0,
        parent_pareto_score=0.0,
        parent_on_cluster_score=0.0,
        trace_embedding=_trace_vec(),
    )
    assert feat[2] == 1.0  # S3 is index 2.


def test_featurize_multi_hot_spans() -> None:
    edits = [_make_edit(span="S1", content="a"), _make_edit(span="S5", content="b")]
    feat = featurize(
        edits=edits,
        cluster_id=0,
        cluster_centroid_score=0.0,
        parent_pareto_score=0.0,
        parent_on_cluster_score=0.0,
        trace_embedding=_trace_vec(),
    )
    assert feat[0] == 1.0
    assert feat[4] == 1.0


def test_featurize_bigrams_deterministic() -> None:
    edit = _make_edit(content="alpha beta gamma delta")
    base_kwargs = dict(
        cluster_id=0,
        cluster_centroid_score=0.0,
        parent_pareto_score=0.0,
        parent_on_cluster_score=0.0,
        trace_embedding=_trace_vec(seed=0),
    )
    f1 = featurize(edits=[edit], **base_kwargs)
    f2 = featurize(edits=[edit], **base_kwargs)
    bg_lo = EDIT_SPAN_DIM
    bg_hi = EDIT_SPAN_DIM + EDIT_BIGRAM_DIM
    np.testing.assert_array_equal(f1[bg_lo:bg_hi], f2[bg_lo:bg_hi])
    assert (f1[bg_lo:bg_hi] > 0).any()


def test_featurize_cluster_clamp() -> None:
    feat = featurize(
        edits=[_make_edit()],
        cluster_id=99,
        cluster_centroid_score=0.0,
        parent_pareto_score=0.0,
        parent_on_cluster_score=0.0,
        trace_embedding=_trace_vec(),
    )
    base = EDIT_SPAN_DIM + EDIT_BIGRAM_DIM
    assert feat[base + 7] == 1.0


def test_featurize_trace_embedding_block() -> None:
    trace = _trace_vec(seed=123)
    feat = featurize(
        edits=[_make_edit()],
        cluster_id=0,
        cluster_centroid_score=0.0,
        parent_pareto_score=0.0,
        parent_on_cluster_score=0.0,
        trace_embedding=trace,
    )
    base = (
        EDIT_SPAN_DIM
        + EDIT_BIGRAM_DIM
        + CLUSTER_ID_DIM
        + CLUSTER_SCORE_DIM
        + PARENT_SCORE_DIM
    )
    np.testing.assert_array_equal(feat[base : base + TRACE_EMB_DIM], trace)


def test_featurize_rejects_wrong_trace_dim() -> None:
    with pytest.raises(ValueError):
        featurize(
            edits=[_make_edit()],
            cluster_id=0,
            cluster_centroid_score=0.0,
            parent_pareto_score=0.0,
            parent_on_cluster_score=0.0,
            trace_embedding=np.zeros(100, dtype=np.float32),
        )


def test_featurize_reserved_block_zeros() -> None:
    feat = featurize(
        edits=[_make_edit()],
        cluster_id=0,
        cluster_centroid_score=1.0,
        parent_pareto_score=1.0,
        parent_on_cluster_score=1.0,
        trace_embedding=_trace_vec(),
    )
    assert np.all(feat[465:512] == 0.0)


# ====================================================== lightgbm_model ====


def _synth_balanced(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic dataset: top-half scores -> 1, bottom-half -> 0."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, FEATURE_DIM)).astype(np.float32)
    w = rng.standard_normal(FEATURE_DIM).astype(np.float32)
    scores = X @ w
    median = float(np.median(scores))
    y = (scores > median).astype(np.int64)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def test_fit_returns_false_when_class_imbalanced() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, FEATURE_DIM)).astype(np.float32)
    y = np.ones(50, dtype=np.int64)
    m = SurrogateModel()
    assert m.fit(X, y) is False
    assert m.is_fitted is False


def test_fit_succeeds_on_balanced_synthetic() -> None:
    X, y = _synth_balanced(n=200, seed=1)
    m = SurrogateModel()
    assert m.fit(X, y) is True
    assert m.is_fitted is True
    assert m.n_fits == 1
    assert m.last_train_size == 200


def test_predict_proba_returns_in_unit_interval() -> None:
    X, y = _synth_balanced(n=200, seed=2)
    m = SurrogateModel()
    assert m.fit(X, y) is True
    p = m.predict_proba(X)
    assert p.shape == (200,)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_save_load_roundtrip_byte_identical_predictions(tmp_path) -> None:
    X, y = _synth_balanced(n=200, seed=3)
    m = SurrogateModel()
    assert m.fit(X, y) is True
    p_before = m.predict_proba(X)
    m.save(tmp_path)

    m2 = SurrogateModel()
    m2.load(tmp_path)
    assert m2.is_fitted is True
    p_after = m2.predict_proba(X)
    np.testing.assert_allclose(p_before, p_after, atol=1e-9)


def test_save_writes_text_format(tmp_path) -> None:
    X, y = _synth_balanced(n=200, seed=4)
    m = SurrogateModel()
    assert m.fit(X, y) is True
    m.save(tmp_path)
    text = (tmp_path / "surrogate.txt").read_text()
    first_line = text.splitlines()[0]
    assert "tree" in first_line or "version" in first_line
    m2 = SurrogateModel()
    m2.load(tmp_path)
    assert m2.is_fitted is True


def test_no_forbidden_serializer_imports_in_lightgbm_module() -> None:
    import scalpel.surrogate.lightgbm_model as mod

    src = inspect.getsource(mod)
    for forbidden in _forbidden_import_lines():
        assert forbidden not in src, (
            f"forbidden binary-serialization import detected: {forbidden!r}"
        )


# ========================================================== skip_policy ====


def test_brier_monitor_initially_zero() -> None:
    m = BrierMonitor()
    assert m.brier() == 0.0
    assert m.should_kill() is False


def test_brier_monitor_kills_when_threshold_exceeded() -> None:
    m = BrierMonitor(kill_threshold=BRIER_KILL_THRESHOLD)
    for _ in range(60):
        m.update(1.0, 0.0)
    assert m.brier() == pytest.approx(1.0)
    assert m.should_kill() is True


def test_brier_monitor_doesnt_kill_below_min_window() -> None:
    m = BrierMonitor()
    for _ in range(30):
        m.update(1.0, 0.0)
    assert m.should_kill() is False


def test_skip_policy_disabled_until_fitted() -> None:
    p = SkipPolicy()
    assert p.enabled is False
    assert p.should_skip("c", {"id": "i1"}) is False


def _seed_policy_with_fit(seed: int = 0) -> SkipPolicy:
    p = SkipPolicy()
    X, y = _synth_balanced(n=200, seed=seed)
    assert p.fit(X, y) is True
    return p


def test_skip_policy_should_skip_below_threshold() -> None:
    p = _seed_policy_with_fit(seed=10)
    p.model.predict_proba = lambda X: np.array([0.05])  # type: ignore[assignment]
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    instance = {"id": "i1"}
    p.set_feature("c", instance, feat)
    assert p.should_skip("c", instance) is True


def test_skip_policy_should_not_skip_above_threshold() -> None:
    p = _seed_policy_with_fit(seed=11)
    p.model.predict_proba = lambda X: np.array([0.80])  # type: ignore[assignment]
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    instance = {"id": "i2"}
    p.set_feature("c", instance, feat)
    assert p.should_skip("c", instance) is False


def test_skip_policy_brier_kill_disables_skipping() -> None:
    p = _seed_policy_with_fit(seed=12)
    p.model.predict_proba = lambda X: np.array([0.05])  # type: ignore[assignment]
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    instance = {"id": "i3"}
    p.set_feature("c", instance, feat)
    p.monitor._buf = [(1.0, 0.0)] * 60
    assert p.should_skip("c", instance) is False
    assert p.enabled is False


def test_save_load_roundtrip_skip_policy(tmp_path) -> None:
    p = _seed_policy_with_fit(seed=13)
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    p.set_feature("c", {"id": "i"}, feat)
    p.reservoir_X = [feat, feat]
    p.reservoir_y = [0, 1]
    p.reservoir_pred = [0.1, 0.9]
    p.monitor._buf = [(1.0, 0.5), (0.0, 0.3)]
    p.save(tmp_path)

    p2 = SkipPolicy()
    p2.load(tmp_path)
    assert p2.enabled is True
    assert len(p2.monitor._buf) == 2
    assert len(p2.reservoir_X) == 2
    assert p2.reservoir_y == [0, 1]


def test_predict_returns_default_when_unfit() -> None:
    p = SkipPolicy()
    assert p.predict("c", {"id": "i"}) == 0.5


def test_no_forbidden_serializer_imports_in_skip_policy() -> None:
    import scalpel.surrogate.skip_policy as mod

    src = inspect.getsource(mod)
    for forbidden in _forbidden_import_lines():
        assert forbidden not in src, (
            f"forbidden binary-serialization import detected: {forbidden!r}"
        )


# ============================================== SurrogateLike contract ====


def test_skip_policy_satisfies_surrogate_protocol() -> None:
    assert isinstance(SkipPolicy(), SurrogateLike)


# ========================================== cold-start integration =========


def test_cold_start_zero_skips() -> None:
    policy = SkipPolicy()
    race = SuccessiveHalving(surrogate=policy, rng_seed=0)
    candidates = [f"c{i}" for i in range(8)]
    pool = [{"id": f"i{i}"} for i in range(64)]

    def eval_fn(_cid: str, _inst: dict) -> float:
        return 0.5

    result = race.race(candidates=candidates, eval_fn=eval_fn, instance_pool=pool)
    assert result.total_skipped == 0
