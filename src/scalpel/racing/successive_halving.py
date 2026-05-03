"""Successive halving racing driver for SCALPEL Phase 6.

Implements Karnin-style successive halving over K=8 sibling candidates with
cluster-stratified rollout sampling and an injected surrogate-skip dependency
that defaults to a no-op.  See ``docs/scalpel/SCALPEL.md`` §3.C and §5.6.

Deviation from the spec sketch in §5.6: the original document outlines an
``async race()`` coroutine.  This module ships a **synchronous** ``race()``
instead, because Phase 1's :class:`scalpel.llm.client.LiteLLMClient` is sync
(its ``__call__`` uses a ``threading.Semaphore`` for concurrency control) and
the rest of the codebase — see ``scripts/raycluster/run_gepa.py`` — is sync
top-to-bottom.  Callers who want intra-rung parallelism can wrap ``eval_fn``
in a ``concurrent.futures.ThreadPoolExecutor``; we deliberately do not force
async up the stack.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from scalpel.clustering.kmeans import ClusterState
from scalpel.racing.rungs import DEFAULT_ETA, DEFAULT_RUNGS, halve

__all__ = [
    "CandidateScore",
    "FailurePoolStratifier",
    "RaceResult",
    "RungLog",
    "StratifierLike",
    "SuccessiveHalving",
    "SurrogateLike",
]


# --------------------------------------------------------------------- models


class CandidateScore(BaseModel):
    """Mean ± std + sample list for a single candidate at the current rung."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    candidate_id: str
    samples: list[float] = Field(default_factory=list)

    @property
    def mean(self) -> float:
        if not self.samples:
            return 0.0
        return float(sum(self.samples) / len(self.samples))


class RungLog(BaseModel):
    """One rung's outcome — useful for telemetry."""

    rung_index: int
    rung_size: int  # rollouts per candidate at this rung.
    n_alive_before: int
    n_alive_after: int
    rollouts_run: int  # net rollouts performed (excludes surrogate-skipped).
    rollouts_skipped: int  # surrogate-imputed.
    survivor_ids: list[str]


class RaceResult(BaseModel):
    """Final outcome of a race."""

    survivor_id: str
    survivor_score: float
    rung_logs: list[RungLog]
    total_rollouts: int  # net (run, not skipped).
    total_skipped: int


# ------------------------------------------------------------------ protocols


@runtime_checkable
class SurrogateLike(Protocol):
    """Phase 7's Surrogate will implement this; Phase 6 ships a no-op default."""

    def should_skip(self, candidate_id: str, instance: Any) -> bool: ...

    def predict(self, candidate_id: str, instance: Any) -> float: ...


class _NoopSurrogate:
    def should_skip(self, candidate_id: str, instance: Any) -> bool:
        return False

    def predict(self, candidate_id: str, instance: Any) -> float:
        return 0.5  # never invoked when ``should_skip`` returns False.


@runtime_checkable
class StratifierLike(Protocol):
    """Picks ``n`` instances from ``pool``, weighted by cluster.

    Phase 7+ may swap in a smarter stratifier; Phase 6 ships with a default
    that is uniform across the pool when no clusters are provided, and
    proportional to ``failure_count^0.5`` (square-root tempering, per §5.6
    last paragraph) when clusters are provided.
    """

    def sample(self, pool: list[Any], n: int) -> list[Any]: ...


# ------------------------------------------------------------------ helpers


def _instance_id(inst: Any) -> Any:
    """Best-effort id extraction for an instance.

    Falls back to ``id(inst)`` when no ``id`` attribute or key exists.
    """
    attr_id = getattr(inst, "id", None)
    if attr_id is not None:
        return attr_id
    if hasattr(inst, "get"):
        got = inst.get("id")
        if got is not None:
            return got
    return id(inst)


# ---------------------------------------------------------------- stratifier


class FailurePoolStratifier:
    """Cluster-stratified sampler with sqrt-tempered cluster quotas."""

    def __init__(
        self,
        clusters: list[ClusterState] | None = None,
        cluster_assignments: dict[Any, int] | None = None,
        seed: int = 0,
    ) -> None:
        self.clusters = clusters or []
        self.cluster_assignments = cluster_assignments or {}
        self._rng = random.Random(seed)

    def sample(self, pool: list[Any], n: int) -> list[Any]:
        if n <= 0 or not pool:
            return []

        # No clusters / no assignments: uniform without replacement (with
        # fallback to with-replacement when ``n`` exceeds pool size).
        if not self.clusters or not self.cluster_assignments:
            if n <= len(pool):
                return self._rng.sample(pool, n)
            return [self._rng.choice(pool) for _ in range(n)]

        # Bucket pool entries by cluster id; entries with no assignment are
        # parked in a ``None`` bucket and treated as uniform fallback.
        buckets: dict[int, list[Any]] = {c.id: [] for c in self.clusters}
        unassigned: list[Any] = []
        for inst in pool:
            cid = self.cluster_assignments.get(_instance_id(inst))
            if cid is None or cid not in buckets:
                unassigned.append(inst)
            else:
                buckets[cid].append(inst)

        # Compute sqrt-tempered weights from failure counts.
        weights = [math.sqrt(max(0, c.failure_count)) for c in self.clusters]
        total_w = sum(weights)
        if total_w <= 0:
            # All-zero failure counts: degenerate; fall back to uniform.
            if n <= len(pool):
                return self._rng.sample(pool, n)
            return [self._rng.choice(pool) for _ in range(n)]

        # Largest-remainder allocation of ``n`` across cluster quotas so the
        # quotas sum to exactly ``n``.
        raw = [n * w / total_w for w in weights]
        floors = [int(math.floor(x)) for x in raw]
        remainder = n - sum(floors)
        # Sort cluster indices by descending fractional part; bump those.
        fracs = sorted(
            range(len(self.clusters)),
            key=lambda i: raw[i] - floors[i],
            reverse=True,
        )
        quotas: list[int] = list(floors)
        for i in fracs[:remainder]:
            quotas[i] += 1

        # Draw per cluster; fall back to with-replacement when quota > size.
        out: list[Any] = []
        for c, quota in zip(self.clusters, quotas, strict=True):
            if quota <= 0:
                continue
            bucket = buckets[c.id]
            if not bucket:
                # Empty bucket — sample with replacement from unassigned/pool
                # so the total still reaches ``n``.
                fallback = unassigned or pool
                out.extend(self._rng.choice(fallback) for _ in range(quota))
                continue
            if quota <= len(bucket):
                out.extend(self._rng.sample(bucket, quota))
            else:
                # Quota exceeds bucket: take the whole bucket then top up
                # with replacement from the same bucket.
                out.extend(bucket)
                deficit = quota - len(bucket)
                out.extend(self._rng.choice(bucket) for _ in range(deficit))
        return out


# ------------------------------------------------------------ race driver


class SuccessiveHalving:
    """Karnin-style successive halving over K candidates."""

    def __init__(
        self,
        rungs: tuple[int, ...] = DEFAULT_RUNGS,
        eta: int = DEFAULT_ETA,
        surrogate: SurrogateLike | None = None,
        stratifier: StratifierLike | None = None,
        rng_seed: int = 0,
    ) -> None:
        self.rungs = rungs
        self.eta = eta
        self.surrogate: SurrogateLike = (
            surrogate if surrogate is not None else _NoopSurrogate()
        )
        self.stratifier: StratifierLike = (
            stratifier if stratifier is not None else FailurePoolStratifier(seed=rng_seed)
        )
        self._rng = random.Random(rng_seed)

    def race(
        self,
        candidates: list[str],
        eval_fn: Callable[[str, Any], float],
        instance_pool: list[Any],
    ) -> RaceResult:
        """Run successive halving.

        For each rung ``r`` in ``self.rungs``:

        * Sample ``r * len(alive)`` instances stratified across the failure
          pool, then split into per-candidate slices of length ``r``.
        * For each ``(candidate, instance)`` pair, optionally short-circuit
          via ``surrogate.should_skip`` and impute ``surrogate.predict``;
          else call ``eval_fn(candidate, instance)``.
        * Sort survivors by accumulated mean, halve via ``halve`` and
          continue.  Score samples persist across rungs so a survivor's
          mean reflects every rollout it has run so far.
        * Stop when ``len(alive) <= 1`` or all rungs are consumed.
        """
        if not candidates:
            raise ValueError("no candidates to race")

        # Initialize per-candidate score accumulators.
        scores: dict[str, CandidateScore] = {
            cid: CandidateScore(candidate_id=cid) for cid in candidates
        }
        alive: list[str] = list(candidates)
        rung_logs: list[RungLog] = []
        total_run = 0
        total_skipped = 0
        # Single-candidate races run rung 0 once and exit; multi-candidate
        # races traverse every rung even after halving lands on a single
        # survivor (so the winner accumulates signal across all rungs).
        single_candidate_start = len(candidates) == 1

        for rung_idx, rung_size in enumerate(self.rungs):
            n_alive_before = len(alive)
            need = rung_size * n_alive_before
            batch = self.stratifier.sample(instance_pool, need)

            # If the stratifier returns fewer instances than requested
            # (e.g. tiny pool with no replacement), pad by resampling so
            # every alive candidate still has ``rung_size`` items.
            if len(batch) < need and instance_pool:
                while len(batch) < need:
                    batch.append(self._rng.choice(instance_pool))

            rollouts_run = 0
            rollouts_skipped = 0
            for i, cid in enumerate(alive):
                start = i * rung_size
                slice_ = batch[start : start + rung_size]
                for inst in slice_:
                    if self.surrogate.should_skip(cid, inst):
                        scores[cid].samples.append(
                            float(self.surrogate.predict(cid, inst))
                        )
                        rollouts_skipped += 1
                    else:
                        scores[cid].samples.append(float(eval_fn(cid, inst)))
                        rollouts_run += 1

            total_run += rollouts_run
            total_skipped += rollouts_skipped

            # Halve (skip if only one candidate is alive — single-cand races
            # still record the rung but do not eliminate anyone).
            if n_alive_before <= 1:
                n_alive_after = n_alive_before
            else:
                n_alive_after = halve(n_alive_before, self.eta)
                # Sort descending by accumulated mean; survivors are top-k.
                alive = sorted(
                    alive,
                    key=lambda c: scores[c].mean,
                    reverse=True,
                )[:n_alive_after]

            rung_logs.append(
                RungLog(
                    rung_index=rung_idx,
                    rung_size=rung_size,
                    n_alive_before=n_alive_before,
                    n_alive_after=n_alive_after,
                    rollouts_run=rollouts_run,
                    rollouts_skipped=rollouts_skipped,
                    survivor_ids=list(alive),
                )
            )

            # Single-candidate races stop after rung 0; multi-candidate races
            # always run all rungs so the eventual survivor accumulates
            # signal at every rung size.
            if single_candidate_start:
                break

        # Pick the highest-mean candidate among the final ``alive`` set.
        survivor_id = max(alive, key=lambda c: scores[c].mean)
        return RaceResult(
            survivor_id=survivor_id,
            survivor_score=scores[survivor_id].mean,
            rung_logs=rung_logs,
            total_rollouts=total_run,
            total_skipped=total_skipped,
        )
