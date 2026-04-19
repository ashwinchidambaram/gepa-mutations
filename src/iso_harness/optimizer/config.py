"""Configuration dataclasses for the ISO optimizer.

``ISOConfig`` carries all hyperparameters for one optimisation run.
``VariantHooks`` bundles the three callable hooks that distinguish ISO variants
(prune, reflect, cross_mutate) together with their variant-specific parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "VariantHooks",
    "ISOConfig",
]


@dataclass
class VariantHooks:
    """Callable hooks that define an ISO variant's behaviour.

    Each hook is a plain Python callable.  The ISO engine calls them at the
    appropriate points in the optimisation loop.

    Attributes:
        prune: ``(pool, config, runtime) -> list[Candidate]``
            Returns the surviving candidates after pruning.
        reflect: ``(candidate, traces, config, runtime) -> MutationProposal``
            Generates a mutation proposal for a single candidate.
        cross_mutate: ``(a, b, config, runtime) -> MutationProposal``
            Generates a mutation proposal by crossing two candidates.
        prune_ratio: Fraction of pool to prune each round (variant-specific).
        pool_size_max: Hard cap on pool size (variant-specific, or None).
        cross_mutate_only_when_improving: When True, cross-mutation is only
            triggered if population score is trending upward.
    """

    prune: Callable
    reflect: Callable
    cross_mutate: Callable
    prune_ratio: float | None = None
    pool_size_max: int | None = None
    cross_mutate_only_when_improving: bool = False


@dataclass
class ISOConfig:
    """Full configuration for one ISO optimisation run.

    Attributes:
        budget: Maximum number of task-LM rollouts allowed for this run.
        seed: Random seed (controls minibatch sampling, RNG, etc.).
        hooks: ``VariantHooks`` instance that defines variant behaviour.
        pool_floor: Minimum number of candidates kept alive after pruning.
        max_rounds: Hard cap on optimisation rounds.
        plateau_rounds_threshold: Number of consecutive non-improving rounds
            before early stopping.
        plateau_tolerance: Minimum score delta to count as improvement.
        n_discovery_examples: Examples used for inductive skill discovery.
        target_skills_min: Minimum number of skill clusters to discover.
        target_skills_max: Maximum number of skill clusters to discover.
        mutations_per_seed: Mutation proposals generated per surviving candidate.
        minibatch_count: Number of minibatches evaluated per round.
        minibatch_size: Examples per minibatch.
        merge_interval: Rounds between cross-mutation (merge) events.
    """

    budget: int
    seed: int
    hooks: VariantHooks
    pool_floor: int = 6
    max_rounds: int = 20
    plateau_rounds_threshold: int = 3
    plateau_tolerance: float = 0.001
    n_discovery_examples: int = 30
    target_skills_min: int = 3
    target_skills_max: int = 8
    mutations_per_seed: int = 2
    minibatch_count: int = 5
    minibatch_size: int = 4
    merge_interval: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (hooks stored as function names).

        Returns:
            Plain dict suitable for ``json.dumps`` and JSONL logging.
        """
        return {
            "budget": self.budget,
            "seed": self.seed,
            "pool_floor": self.pool_floor,
            "max_rounds": self.max_rounds,
            "plateau_rounds_threshold": self.plateau_rounds_threshold,
            "plateau_tolerance": self.plateau_tolerance,
            "n_discovery_examples": self.n_discovery_examples,
            "target_skills_min": self.target_skills_min,
            "target_skills_max": self.target_skills_max,
            "mutations_per_seed": self.mutations_per_seed,
            "minibatch_count": self.minibatch_count,
            "minibatch_size": self.minibatch_size,
            "merge_interval": self.merge_interval,
            "hooks": {
                "prune": self.hooks.prune.__name__ if self.hooks.prune else None,
                "reflect": self.hooks.reflect.__name__ if self.hooks.reflect else None,
                "cross_mutate": (
                    self.hooks.cross_mutate.__name__
                    if self.hooks.cross_mutate
                    else None
                ),
                "prune_ratio": self.hooks.prune_ratio,
                "pool_size_max": self.hooks.pool_size_max,
                "cross_mutate_only_when_improving": (
                    self.hooks.cross_mutate_only_when_improving
                ),
            },
        }
