"""ActiveMinibatchSampler: selects high-disagreement examples for reflection.

Maintains its own per-example score history (since EvaluationCache is indexed by
(candidate_hash, example_id) and does not expose cross-candidate queries).
Scores are fed via update_scores() called from ActiveMinibatchCallback after
each training evaluation.

During warmup, falls back to EpochShuffledBatchSampler. After warmup, selects
(1 - fallback_ratio) examples by highest variance and fallback_ratio random
examples for exploration.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler


class ActiveMinibatchSampler:
    """BatchSampler that preferentially selects high-disagreement examples.

    Maintains its own per-example score history (since EvaluationCache
    doesn't expose cross-candidate queries). Scores are fed via
    update_scores() called from the callback on each evaluation.

    For the first warmup_iterations, falls back to EpochShuffledBatchSampler.
    After warmup: selects (1-fallback_ratio) examples by highest variance,
    plus fallback_ratio random examples for exploration.

    Args:
        minibatch_size: Number of examples per minibatch.
        rng: Random number generator for reproducibility.
        warmup_iterations: Number of iterations to use fallback sampler.
        fallback_ratio: Fraction of minibatch to fill with random examples
            for exploration (default 0.3).
    """

    def __init__(
        self,
        minibatch_size: int,
        rng: random.Random,
        warmup_iterations: int = 10,
        fallback_ratio: float = 0.3,
    ) -> None:
        self.minibatch_size = minibatch_size
        self.rng = rng
        self.warmup_iterations = warmup_iterations
        self.fallback_ratio = fallback_ratio
        self._fallback = EpochShuffledBatchSampler(minibatch_size, rng)
        # {example_id: [score_1, score_2, ...]} — accumulated across iterations
        self._score_index: dict[Any, list[float]] = defaultdict(list)
        # Cache hit/miss tracking: counts across all post-warmup selections
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def update_scores(self, example_ids: list[Any], scores: list[float]) -> None:
        """Record scores for a batch of examples.

        Called from ActiveMinibatchCallback after each training evaluation so
        that the sampler can track per-example score variance over time.

        Args:
            example_ids: List of example IDs evaluated in this batch.
            scores: Corresponding scores (same length as example_ids).
        """
        for eid, s in zip(example_ids, scores):
            self._score_index[eid].append(s)

    def _compute_disagreement(self, eid: Any) -> float:
        """Compute variance-based disagreement score for a single example.

        Returns float('inf') for examples with fewer than 2 observations
        (unseen = highest priority so they get evaluated early).
        """
        scores = self._score_index.get(eid, [])
        if len(scores) < 2:
            return float("inf")
        mean = sum(scores) / len(scores)
        return sum((s - mean) ** 2 for s in scores) / len(scores)

    def next_minibatch_ids(
        self, loader: DataLoader[DataId, Any], state: GEPAState
    ) -> list[DataId]:
        """Return the next minibatch of example IDs.

        During warmup (state.i < warmup_iterations): delegates to
        EpochShuffledBatchSampler.

        After warmup: selects n_active examples with highest disagreement
        (variance) plus n_random random examples from the remainder.

        Args:
            loader: DataLoader providing all training IDs.
            state: Current GEPA optimization state (used for iteration count).

        Returns:
            List of example IDs for this minibatch.
        """
        if state.i < self.warmup_iterations:
            return self._fallback.next_minibatch_ids(loader, state)

        all_ids = list(loader.all_ids())

        if len(all_ids) <= self.minibatch_size:
            # Entire trainset fits in one minibatch — return all
            return all_ids

        n_active = int(self.minibatch_size * (1 - self.fallback_ratio))
        n_random = self.minibatch_size - n_active

        # Sort by disagreement descending; unseen (inf) come first
        sorted_ids = sorted(all_ids, key=self._compute_disagreement, reverse=True)
        active_ids = sorted_ids[:n_active]

        # Random exploration from the remaining IDs
        remaining = [x for x in all_ids if x not in set(active_ids)]
        random_ids = self.rng.sample(remaining, min(n_random, len(remaining)))

        selected = active_ids + random_ids
        for eid in selected:
            if eid in self._score_index and self._score_index[eid]:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

        return selected

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of post-warmup selections that had prior score history."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
