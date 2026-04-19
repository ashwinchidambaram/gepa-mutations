"""Pruning strategies for ISO optimizer.

Three strategies across variants. All follow the same hook signature so the
core loop can call any of them through ``config.hooks.prune(...)``.

Hook signature::

    prune(pool, scores, pool_improved, prev_top3_mean, top3_mean, config, runtime)
        -> list[Candidate]
"""

from __future__ import annotations

from iso_harness.optimizer.candidate import Candidate

__all__ = [
    "prune_fixed_ratio",
    "prune_adaptive_with_floor",
    "prune_adaptive_to_regression",
]


def prune_fixed_ratio(
    pool: list[Candidate],
    scores: dict[str, dict],
    pool_improved: bool,
    prev_top3_mean: float,
    top3_mean: float,
    config,  # ISOConfig
    runtime,  # ISORuntime
) -> list[Candidate]:
    """Sprint: prune a fixed fraction each round.

    Uses ``config.hooks.prune_ratio`` (default 0.5). Never drops below
    ``config.pool_floor``.
    """
    ratio = config.hooks.prune_ratio if config.hooks.prune_ratio is not None else 0.5
    n_keep = max(int(len(pool) * (1 - ratio)), config.pool_floor)
    ranked = sorted(pool, key=lambda c: scores[c.id]["mean"], reverse=True)
    survivors = ranked[:n_keep]
    for pruned in ranked[n_keep:]:
        pruned.death_round = runtime.round_num
        pruned.death_reason = "pruned_by_fixed_ratio"
    return survivors


def prune_adaptive_with_floor(
    pool: list[Candidate],
    scores: dict[str, dict],
    pool_improved: bool,
    prev_top3_mean: float,
    top3_mean: float,
    config,  # ISOConfig
    runtime,  # ISORuntime
) -> list[Candidate]:
    """Grove: gentle pruning (25%), never below floor (8 for Grove)."""
    ratio = 0.25
    n_keep = max(
        int(len(pool) * (1 - ratio)),
        config.pool_floor,
    )
    ranked = sorted(pool, key=lambda c: scores[c.id]["mean"], reverse=True)
    survivors = ranked[:n_keep]
    for pruned in ranked[n_keep:]:
        pruned.death_round = runtime.round_num
        pruned.death_reason = "pruned_adaptive_with_floor"
    return survivors


def prune_adaptive_to_regression(
    pool: list[Candidate],
    scores: dict[str, dict],
    pool_improved: bool,
    prev_top3_mean: float,
    top3_mean: float,
    config,  # ISOConfig
    runtime,  # ISORuntime
) -> list[Candidate]:
    """Tide/Storm/Lens: prune count scales with regression size.

    If pool improved: prune 1 candidate (light pressure).
    If pool regressed: prune proportionally to regression size (capped at 25%).
    """
    if prev_top3_mean == 0 or runtime.round_num == 1:
        # First round — gentle default
        prune_count = 1
    elif pool_improved:
        prune_count = 1
    else:
        # Regressed — prune proportionally
        regression_pct = (prev_top3_mean - top3_mean) / prev_top3_mean
        prune_count = max(1, int(len(pool) * regression_pct))
        prune_count = min(prune_count, len(pool) // 4)  # cap at 25% per round

    # Enforce floor
    n_keep = max(len(pool) - prune_count, config.pool_floor)
    ranked = sorted(pool, key=lambda c: scores[c.id]["mean"], reverse=True)
    survivors = ranked[:n_keep]
    for pruned in ranked[n_keep:]:
        pruned.death_round = runtime.round_num
        pruned.death_reason = "pruned_adaptive_to_regression"
    return survivors
