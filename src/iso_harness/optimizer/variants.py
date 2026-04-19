"""ISO variant configuration factories.

Each factory returns a fully configured ISOConfig for the named variant.
"""

from __future__ import annotations

from iso_harness.optimizer.config import ISOConfig, VariantHooks
from iso_harness.optimizer.pruning import (
    prune_fixed_ratio,
    prune_adaptive_with_floor,
    prune_adaptive_to_regression,
)
from iso_harness.optimizer.reflection import (
    reflect_per_candidate,
    reflect_population_level,
    reflect_pair_contrastive,
    reflect_hybrid,
)
from iso_harness.optimizer.cross_mutation import (
    cross_mutate_elitist,
    cross_mutate_exploration_preserving,
    cross_mutate_reflector_guided,
)

__all__ = [
    "iso_sprint_config",
    "iso_grove_config",
    "iso_tide_config",
    "iso_lens_config",
    "iso_storm_config",
]


def iso_sprint_config(base_config: dict) -> ISOConfig:
    """Sprint: aggressive pruning, population-level reflection, elitist cross."""
    cfg = dict(base_config)
    cfg.pop("pool_floor", None)
    cfg.pop("hooks", None)
    return ISOConfig(
        **cfg,
        pool_floor=4,
        hooks=VariantHooks(
            prune=prune_fixed_ratio,
            reflect=reflect_population_level,
            cross_mutate=cross_mutate_elitist,
            prune_ratio=0.5,
        ),
    )


def iso_grove_config(base_config: dict) -> ISOConfig:
    """Grove: gentle pruning, per-candidate reflection, exploration cross."""
    cfg = dict(base_config)
    cfg.pop("pool_floor", None)
    cfg.pop("hooks", None)
    return ISOConfig(
        **cfg,
        pool_floor=8,
        hooks=VariantHooks(
            prune=prune_adaptive_with_floor,
            reflect=reflect_per_candidate,
            cross_mutate=cross_mutate_exploration_preserving,
        ),
    )


def iso_tide_config(base_config: dict) -> ISOConfig:
    """Tide: adaptive pruning, per-candidate reflection, conditional elitist cross."""
    cfg = dict(base_config)
    cfg.pop("pool_floor", None)
    cfg.pop("hooks", None)
    hooks = VariantHooks(
        prune=prune_adaptive_to_regression,
        reflect=reflect_per_candidate,
        cross_mutate=cross_mutate_elitist,
    )
    hooks.cross_mutate_only_when_improving = True
    return ISOConfig(**cfg, pool_floor=6, hooks=hooks)


def iso_lens_config(base_config: dict) -> ISOConfig:
    """Lens: adaptive pruning, pair-contrastive reflection, conditional elitist cross."""
    cfg = dict(base_config)
    cfg.pop("pool_floor", None)
    cfg.pop("hooks", None)
    hooks = VariantHooks(
        prune=prune_adaptive_to_regression,
        reflect=reflect_pair_contrastive,
        cross_mutate=cross_mutate_elitist,
    )
    hooks.cross_mutate_only_when_improving = True
    return ISOConfig(**cfg, pool_floor=6, hooks=hooks)


def iso_storm_config(base_config: dict) -> ISOConfig:
    """Storm: adaptive pruning, hybrid reflection, reflector-guided cross."""
    cfg = dict(base_config)
    cfg.pop("pool_floor", None)
    cfg.pop("hooks", None)
    hooks = VariantHooks(
        prune=prune_adaptive_to_regression,
        reflect=reflect_hybrid,
        cross_mutate=cross_mutate_reflector_guided,
        pool_size_max=4,
    )
    return ISOConfig(**cfg, pool_floor=6, hooks=hooks)
