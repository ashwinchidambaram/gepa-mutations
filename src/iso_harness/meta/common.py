"""Shared data structures and helpers for Track 2 meta-optimizers.

MetaEpisode records one episode of the meta-optimizer outer loop.
MetaAction defines the hyperparameter space the meta-LLM searches over.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

logger = logging.getLogger("iso.meta")


@dataclass
class MetaEpisode:
    """One episode of the meta-optimizer outer loop.

    Attributes:
        episode_id: Unique identifier for this episode.
        meta_run_id: ID of the meta-optimizer run this episode belongs to.
        episode_num: Sequential episode number (0-based).
        proposed_config: Subset of ISOConfig fields proposed by the meta-LLM.
        meta_llm_reasoning: Natural-language reasoning from the meta-LLM.
        episode_outcome: ISO run summary dict (final_score, rollouts, tokens).
        reward: Scalar (constrained) or vector (Pareto) reward.
        playbook_state: Current playbook text (for Cartographer and Atlas).
    """

    episode_id: str = field(default_factory=lambda: str(uuid4()))
    meta_run_id: str = ""
    episode_num: int = 0
    proposed_config: dict = field(default_factory=dict)
    meta_llm_reasoning: str = ""
    episode_outcome: dict = field(default_factory=dict)
    reward: float | list[float] = 0.0
    playbook_state: str = ""


@dataclass
class MetaAction:
    """The action space the meta-LLM picks from.

    Each field has a valid range; the meta-LLM proposes values within these bounds.
    """

    pool_size_seed: int = 5  # 3-8
    mutations_per_seed: int = 2  # 1-3
    minibatch_count: int = 5  # 3-7
    minibatch_size: int = 4  # 2-8
    prune_aggressiveness: float = 0.25  # 0.1-0.6
    max_rounds: int = 10  # 3-20

    def to_dict(self) -> dict:
        return {
            "pool_size_seed": self.pool_size_seed,
            "mutations_per_seed": self.mutations_per_seed,
            "minibatch_count": self.minibatch_count,
            "minibatch_size": self.minibatch_size,
            "prune_aggressiveness": self.prune_aggressiveness,
            "max_rounds": self.max_rounds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MetaAction:
        """Create from a parsed config dict, clamping to valid ranges."""
        return cls(
            pool_size_seed=max(3, min(8, int(d.get("pool_size_seed", 5)))),
            mutations_per_seed=max(1, min(3, int(d.get("mutations_per_seed", 2)))),
            minibatch_count=max(3, min(7, int(d.get("minibatch_count", 5)))),
            minibatch_size=max(2, min(8, int(d.get("minibatch_size", 4)))),
            prune_aggressiveness=max(0.1, min(0.6, float(d.get("prune_aggressiveness", 0.25)))),
            max_rounds=max(3, min(20, int(d.get("max_rounds", 10)))),
        )


def compute_constrained_reward(outcome: dict) -> float:
    """Compute scalar reward with budget penalty.

    If rollouts exceed budget, subtract 1.0 from the score as penalty.
    """
    score = outcome.get("final_score", 0.0)
    rollouts = outcome.get("rollouts_consumed", 0)
    budget = outcome.get("budget", rollouts)  # no penalty if budget not specified

    if rollouts > budget:
        return score - 1.0
    return score


def run_iso_with_config(
    variant: str,
    benchmark: str,
    subset_size: int,
    config_overrides: dict,
    meta_lm: Any = None,
    seed: int = 0,
) -> dict:
    """Run one ISO episode with overridden config.

    This is the inner loop call used by all meta-optimizers.
    Returns an outcome dict with: final_score, rollouts_consumed, tokens_consumed, budget.

    Note: For now this is a stub that returns a mock outcome.
    In production, it will call ISO.compile() with the overridden config
    on a surrogate subset of the benchmark.
    """
    # TODO: Wire up to actual ISO.compile() when ready for live runs
    # For now, return a placeholder that meta-optimizers can work with
    logger.info(
        "run_iso_with_config: variant=%s, benchmark=%s, subset=%s, overrides=%s",
        variant,
        benchmark,
        subset_size,
        config_overrides,
    )

    # Calculate a pseudo-budget from overrides
    budget = (
        config_overrides.get("max_rounds", 10)
        * config_overrides.get("minibatch_count", 5)
        * config_overrides.get("minibatch_size", 4)
        * config_overrides.get("pool_size_seed", 5)
    )

    return {
        "final_score": 0.0,  # Stub — real runs populate this
        "rollouts_consumed": 0,
        "tokens_consumed": 0,
        "budget": budget,
    }


def update_pareto_frontier(
    frontier: list[dict],
    new_reward: list[float],
    outcome: dict,
) -> list[dict]:
    """Add new point to frontier if non-dominated; remove dominated points.

    Each point is a dict with 'reward' (list[float]) and 'outcome' (dict).
    """
    point = {"reward": new_reward, "outcome": outcome}

    # Check if dominated by existing frontier
    for existing in frontier:
        if all(e >= n for e, n in zip(existing["reward"], new_reward)) and any(
            e > n for e, n in zip(existing["reward"], new_reward)
        ):
            return frontier  # new point is dominated

    # Remove points dominated by new point
    frontier = [
        existing
        for existing in frontier
        if not (
            all(n >= e for n, e in zip(new_reward, existing["reward"]))
            and any(n > e for n, e in zip(new_reward, existing["reward"]))
        )
    ]

    frontier.append(point)
    return frontier
