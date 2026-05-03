"""SCALPEL Phase 7 — LightGBM binary surrogate with text-format persistence.

Per addendum Q5 the surrogate predicts ``P(success | features)`` using a
LightGBM binary classifier on 512-d features.  Hyperparameters follow Q5
verbatim.  Persistence uses LightGBM's native text-format ``save_model`` /
``Booster(model_file=...)`` plus a JSON metadata sidecar.  Binary
serialization frameworks are deliberately avoided everywhere.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Pin OpenMP thread count *before* importing :mod:`lightgbm`.  LightGBM 4.x
# under macOS segfaults when both LightGBM's libomp and sklearn's libiomp are
# loaded multi-threaded in the same process — see
# https://github.com/microsoft/LightGBM/issues/4889.  Setting the env var to
# ``"1"`` is the documented workaround and has no measurable effect on
# Linux/CUDA training.  Existing values are respected.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb  # noqa: E402  -- env var must be set before import
import numpy as np  # noqa: E402

__all__ = [
    "EARLY_STOPPING_ROUNDS",
    "FEATURE_DIM",
    "HOLDOUT_FRAC",
    "LIGHTGBM_PARAMS",
    "MAX_TREES",
    "MIN_LABELS_PER_CLASS",
    "SCHEMA_VERSION",
    "SurrogateModel",
]


# Per addendum Q5.  ``num_threads=1`` is appended to defuse the libomp/libiomp
# conflict that segfaults LightGBM after ``sklearn`` has been imported in the
# same process on macOS — a well-known LightGBM-on-macOS issue and not a
# functional change to the model.
LIGHTGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 0.1,
    "verbose": -1,
    "num_threads": 1,
}
EARLY_STOPPING_ROUNDS = 20
HOLDOUT_FRAC = 0.20
MAX_TREES = 200
MIN_LABELS_PER_CLASS = 5
FEATURE_DIM = 512
SCHEMA_VERSION = 1


class SurrogateModel:
    """LightGBM binary classifier with native text-format persistence."""

    def __init__(self) -> None:
        self.booster: lgb.Booster | None = None
        self.is_fitted: bool = False
        self.n_fits: int = 0
        self.last_train_size: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Fit the booster.

        Returns ``True`` on success and ``False`` when either class has fewer
        than :data:`MIN_LABELS_PER_CLASS` samples (in which case the model is
        left untouched and ``is_fitted`` stays at its prior value).
        """
        if X.ndim != 2 or X.shape[1] != FEATURE_DIM:
            raise ValueError(f"expected (N, {FEATURE_DIM}); got {X.shape}")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("X/y length mismatch")
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos < MIN_LABELS_PER_CLASS or n_neg < MIN_LABELS_PER_CLASS:
            return False

        # Cast to float64 + ensure C-contiguous layout: avoids the float32
        # segfault we observed under LightGBM 4.x on macOS when sklearn has
        # already loaded its OpenMP runtime in the same process.
        X64 = np.ascontiguousarray(X, dtype=np.float64)
        n_train = max(1, int((1 - HOLDOUT_FRAC) * len(X64)))
        train_set = lgb.Dataset(X64[:n_train], y[:n_train])
        valid_set = lgb.Dataset(X64[n_train:], y[n_train:], reference=train_set)
        self.booster = lgb.train(
            LIGHTGBM_PARAMS,
            train_set,
            num_boost_round=MAX_TREES,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        self.is_fitted = True
        self.n_fits += 1
        self.last_train_size = len(X)
        return True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict ``P(success)`` for each row."""
        if not self.is_fitted or self.booster is None:
            raise RuntimeError("predict_proba called before fit")
        if X.ndim != 2 or X.shape[1] != FEATURE_DIM:
            raise ValueError(f"expected (N, {FEATURE_DIM}); got {X.shape}")
        X64 = np.ascontiguousarray(X, dtype=np.float64)
        return np.asarray(self.booster.predict(X64), dtype=np.float64)

    def save(self, dir_path: str | Path) -> None:
        """Persist booster (text format) + metadata sidecar (JSON)."""
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        if self.is_fitted and self.booster is not None:
            self.booster.save_model(str(d / "surrogate.txt"))
        meta = {
            "is_fitted": self.is_fitted,
            "n_fits": self.n_fits,
            "last_train_size": self.last_train_size,
            "feature_dim": FEATURE_DIM,
            "schema_version": SCHEMA_VERSION,
        }
        (d / "surrogate_meta.json").write_text(json.dumps(meta, indent=2))

    def load(self, dir_path: str | Path) -> None:
        """Inverse of :meth:`save`."""
        d = Path(dir_path)
        meta = json.loads((d / "surrogate_meta.json").read_text())
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"surrogate schema version mismatch: {meta.get('schema_version')}"
            )
        self.n_fits = meta["n_fits"]
        self.last_train_size = meta["last_train_size"]
        if meta["is_fitted"]:
            self.booster = lgb.Booster(model_file=str(d / "surrogate.txt"))
            self.is_fitted = True
