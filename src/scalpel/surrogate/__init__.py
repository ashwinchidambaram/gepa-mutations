"""SCALPEL surrogate (Phase 7) — LightGBM rollout-skip with Brier monitor + isotonic calibration."""

from scalpel.surrogate.features import (
    FEATURE_SCHEMA_VERSION,
    HASH_SEED,
    featurize,
)
from scalpel.surrogate.features import (
    TOTAL_DIM as FEATURE_DIM,
)
from scalpel.surrogate.lightgbm_model import LIGHTGBM_PARAMS, SurrogateModel
from scalpel.surrogate.skip_policy import (
    BRIER_KILL_THRESHOLD,
    SKIP_THRESHOLD,
    BrierMonitor,
    SkipPolicy,
)

__all__ = [
    "BRIER_KILL_THRESHOLD",
    "BrierMonitor",
    "FEATURE_DIM",
    "FEATURE_SCHEMA_VERSION",
    "HASH_SEED",
    "LIGHTGBM_PARAMS",
    "SKIP_THRESHOLD",
    "SkipPolicy",
    "SurrogateModel",
    "featurize",
]
