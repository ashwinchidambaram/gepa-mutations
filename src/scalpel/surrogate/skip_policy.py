"""SCALPEL Phase 7 — Brier-monitored skip policy with isotonic recalibration.

Wraps :class:`scalpel.surrogate.lightgbm_model.SurrogateModel` with:

* an isotonic recalibrator fit on a 5% reservoir of held-out predictions, and
* a sliding-window :class:`BrierMonitor` that disables skipping when calibration
  decays (Brier > 0.22 across the most recent 200 predictions).

The class implements the ``scalpel.racing.successive_halving.SurrogateLike``
protocol (``should_skip`` / ``predict``).  All persistence is JSON +
LightGBM text format — binary serialization frameworks are not used.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from scalpel.surrogate.lightgbm_model import SurrogateModel

__all__ = [
    "BRIER_KILL_THRESHOLD",
    "BRIER_MIN_FOR_KILL",
    "BRIER_WINDOW",
    "BrierMonitor",
    "RESERVOIR_FRAC",
    "SKIP_THRESHOLD",
    "SkipPolicy",
]


SKIP_THRESHOLD = 0.15
BRIER_KILL_THRESHOLD = 0.22
BRIER_WINDOW = 200
BRIER_MIN_FOR_KILL = 50
RESERVOIR_FRAC = 0.05


class BrierMonitor:
    """Sliding-window Brier-score monitor used to disable skipping on drift."""

    def __init__(
        self,
        kill_threshold: float = BRIER_KILL_THRESHOLD,
        window: int = BRIER_WINDOW,
        min_for_kill: int = BRIER_MIN_FOR_KILL,
    ) -> None:
        self.kill_threshold = kill_threshold
        self.window = window
        self.min_for_kill = min_for_kill
        self._buf: list[tuple[float, float]] = []

    def update(self, y_true: float, y_pred: float) -> None:
        self._buf.append((float(y_true), float(y_pred)))
        if len(self._buf) > self.window:
            self._buf = self._buf[-self.window :]

    def brier(self) -> float:
        if not self._buf:
            return 0.0
        return float(np.mean([(t - p) ** 2 for t, p in self._buf]))

    def should_kill(self) -> bool:
        if len(self._buf) < self.min_for_kill:
            return False
        return self.brier() > self.kill_threshold

    def reset(self) -> None:
        self._buf = []


def _instance_id(instance: Any) -> Any:
    """Hashable id: prefer ``.id`` attr, then ``dict["id"]``, then ``id()``."""
    if hasattr(instance, "id"):
        return getattr(instance, "id")
    if isinstance(instance, dict):
        return instance.get("id", id(instance))
    return id(instance)


class SkipPolicy:
    """Surrogate skip policy.

    Implements ``scalpel.racing.successive_halving.SurrogateLike`` at runtime:
    ``should_skip(candidate_id, instance) -> bool`` and
    ``predict(candidate_id, instance) -> float``.

    Holds a :class:`SurrogateModel`, a :class:`BrierMonitor`, an isotonic
    calibrator fit on a 5% reservoir of held-out predictions, and a
    per-race feature cache mapping ``(candidate_id, instance_id)`` to a
    pre-computed 512-d feature vector.
    """

    def __init__(
        self,
        skip_threshold: float = SKIP_THRESHOLD,
        brier_kill_threshold: float = BRIER_KILL_THRESHOLD,
        reservoir_frac: float = RESERVOIR_FRAC,
        rng_seed: int = 0,
    ) -> None:
        self.model = SurrogateModel()
        self.skip_threshold = skip_threshold
        self.monitor = BrierMonitor(kill_threshold=brier_kill_threshold)
        self.reservoir_frac = reservoir_frac
        self.reservoir_X: list[np.ndarray] = []
        self.reservoir_y: list[int] = []
        self.reservoir_pred: list[float] = []
        self.calibrator: IsotonicRegression | None = None
        self._enabled: bool = False
        self._feature_cache: dict[tuple[str, Any], np.ndarray] = {}
        self._rng = random.Random(rng_seed)

    @property
    def enabled(self) -> bool:
        return self._enabled and self.model.is_fitted

    def feed_label(
        self, x: np.ndarray, y: int, predicted: float | None = None
    ) -> None:
        """Record a ground-truth label and (optionally) update Brier/reservoir."""
        if predicted is not None:
            self.monitor.update(y, predicted)
            if self._rng.random() < self.reservoir_frac:
                self.reservoir_X.append(x)
                self.reservoir_y.append(int(y))
                self.reservoir_pred.append(float(predicted))

    def calibrate(self) -> bool:
        """Fit isotonic calibrator on the reservoir; require ≥50 samples."""
        if len(self.reservoir_pred) < 50:
            return False
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(np.asarray(self.reservoir_pred), np.asarray(self.reservoir_y))
        self.calibrator = ir
        return True

    def predict_proba(self, x: np.ndarray) -> float:
        """Predict ``P(success)`` for a single feature vector."""
        if not self.model.is_fitted:
            return 0.5
        raw = float(self.model.predict_proba(x.reshape(1, -1))[0])
        if self.calibrator is not None:
            return float(self.calibrator.transform(np.array([raw]))[0])
        return raw

    # --------------------------------------------------------- SurrogateLike

    def should_skip(self, candidate_id: str, instance: Any) -> bool:
        if not self.enabled:
            return False
        if self.monitor.should_kill():
            self._enabled = False
            return False
        x = self._feature_cache.get((candidate_id, _instance_id(instance)))
        if x is None:
            return False
        return self.predict_proba(x) < self.skip_threshold

    def predict(self, candidate_id: str, instance: Any) -> float:
        if not self.enabled:
            return 0.5
        x = self._feature_cache.get((candidate_id, _instance_id(instance)))
        if x is None:
            return 0.5
        return self.predict_proba(x)

    # ------------------------------------------------------- feature cache

    def set_feature(
        self, candidate_id: str, instance: Any, x: np.ndarray
    ) -> None:
        self._feature_cache[(candidate_id, _instance_id(instance))] = x

    def clear_feature_cache(self) -> None:
        self._feature_cache.clear()

    # ----------------------------------------------------------------- fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        ok = self.model.fit(X, y)
        if ok:
            self._enabled = True
        return ok

    # ----------------------------------------------------------- persistence

    def save(self, dir_path: str | Path) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        self.model.save(d)
        cal_state = None
        if self.calibrator is not None:
            cal_state = {
                "x_thresholds": self.calibrator.X_thresholds_.tolist(),
                "y_thresholds": self.calibrator.y_thresholds_.tolist(),
            }
        state = {
            "skip_threshold": self.skip_threshold,
            "enabled": self._enabled,
            "brier_buf": [list(t) for t in self.monitor._buf],
            "reservoir_X": [x.tolist() for x in self.reservoir_X],
            "reservoir_y": list(self.reservoir_y),
            "reservoir_pred": list(self.reservoir_pred),
            "calibrator": cal_state,
        }
        (d / "skip_policy_state.json").write_text(json.dumps(state, indent=2))

    def load(self, dir_path: str | Path) -> None:
        d = Path(dir_path)
        self.model.load(d)
        state = json.loads((d / "skip_policy_state.json").read_text())
        self.skip_threshold = state["skip_threshold"]
        self._enabled = state["enabled"]
        self.monitor._buf = [tuple(t) for t in state["brier_buf"]]
        self.reservoir_X = [
            np.asarray(x, dtype=np.float32) for x in state["reservoir_X"]
        ]
        self.reservoir_y = list(state["reservoir_y"])
        self.reservoir_pred = list(state["reservoir_pred"])
        if state.get("calibrator") is not None:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(
                np.asarray(state["calibrator"]["x_thresholds"]),
                np.asarray(state["calibrator"]["y_thresholds"]),
            )
            self.calibrator = ir
        else:
            self.calibrator = None
