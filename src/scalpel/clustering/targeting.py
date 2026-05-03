"""Cluster-targeting heuristic for SCALPEL (Addendum Q1).

Phase 5 of SCALPEL.  Implements the per-iteration cluster selector

::

    target = argmax_{cl ∈ active_clusters}
        failure_mass(cl) · (1 − recency_decay(cl))

with ``recency_decay(cl) = γ ** age`` and ``γ = 0.7``.  An anti-starvation
rule prevents a single dominant cluster from monopolising the budget by
gating recently-targeted clusters behind a 1.5× margin against the next
non-recent candidate.

By the coupon-collector argument noted in §4.4 the expected hitting time
to cover all ``k`` clusters under (uniform-mass-equivalent) rotation is
``H_k = k · sum_{i=1..k}(1/i)``, e.g. ≈ 21.7 iters for ``k=8``.

Public surface:

* :class:`TargetingHistory` — frozen pydantic per-cluster recency tracker.
* :func:`select_target_cluster` — Q1 selection rule.
* :func:`expected_hitting_time_ub` — coupon-collector benchmark for tests.

The constants :data:`DEFAULT_GAMMA`, :data:`ANTI_STARVATION_MARGIN`, and
:data:`RECENT_LOOKBACK_ITERS` give the production-default tuning.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from scalpel.clustering.kmeans import ClusterState

__all__ = [
    "ANTI_STARVATION_MARGIN",
    "DEFAULT_GAMMA",
    "RECENT_LOOKBACK_ITERS",
    "TargetingHistory",
    "expected_hitting_time_ub",
    "select_target_cluster",
]

DEFAULT_GAMMA = 0.7
ANTI_STARVATION_MARGIN = 1.5
RECENT_LOOKBACK_ITERS = 2


class TargetingHistory(BaseModel):
    """Tracks per-cluster recency for the Q1 selection rule.

    ``iters_since_targeted`` maps ``cluster_id`` to the number of iterations
    since that cluster was last selected (``0`` = just-now).  Clusters not
    in the dict have never been targeted and decay to ``0`` recency weight.
    """

    iters_since_targeted: dict[int, int] = Field(default_factory=dict)
    last_targeted_id: int | None = None

    def record(self, cluster_id: int) -> "TargetingHistory":
        """Return a new history with ``cluster_id`` newly-targeted now.

        Existing entries are incremented by 1 (one iteration older); the
        newly-targeted cluster is reset to age 0 regardless of whether it
        appeared in the prior dict.
        """
        new = {cid: c + 1 for cid, c in self.iters_since_targeted.items()}
        new[cluster_id] = 0
        return TargetingHistory(iters_since_targeted=new, last_targeted_id=cluster_id)


def select_target_cluster(
    clusters: list[ClusterState],
    history: TargetingHistory,
    gamma: float = DEFAULT_GAMMA,
    anti_starvation_margin: float = ANTI_STARVATION_MARGIN,
    recent_lookback: int = RECENT_LOOKBACK_ITERS,
) -> int:
    """Select the next cluster to target via Q1's heuristic.

    ``score(cl) = failure_mass(cl) · (1 − γ ** age)`` with
    ``failure_mass(cl) = cl.failure_count / sum_failure_counts``.  A cluster
    that has never been targeted has ``age = ∞`` (decay 0) and so scores
    its full ``failure_mass``.

    Anti-starvation: clusters last targeted within ``recent_lookback``
    iterations are eligible only if their score exceeds the best
    non-recent cluster's score by at least ``anti_starvation_margin``.
    Otherwise the best non-recent cluster wins.  When *every* cluster is
    recent, the highest-scoring cluster is returned outright (no eligible
    alternatives exist).
    """
    if not clusters:
        raise ValueError("no clusters provided")

    total_mass = sum(c.failure_count for c in clusters) or 1

    def score(cl: ClusterState) -> float:
        mass = cl.failure_count / total_mass
        if cl.id not in history.iters_since_targeted:
            decay = 0.0
        else:
            decay = gamma ** history.iters_since_targeted[cl.id]
        return mass * (1 - decay)

    recent_ids = {
        cid
        for cid, age in history.iters_since_targeted.items()
        if age < recent_lookback
    }

    scored: list[tuple[float, ClusterState]] = sorted(
        ((score(c), c) for c in clusters),
        key=lambda x: -x[0],
    )
    if not scored:
        raise ValueError("no scorable clusters")

    non_recent = [(s, c) for s, c in scored if c.id not in recent_ids]
    if non_recent:
        non_recent_top_score, non_recent_top = non_recent[0]
        recent = [(s, c) for s, c in scored if c.id in recent_ids]
        if recent and recent[0][0] >= anti_starvation_margin * non_recent_top_score:
            return recent[0][1].id
        return non_recent_top.id

    # All clusters are within the recent-lookback window — pick the best.
    return scored[0][1].id


def expected_hitting_time_ub(k: int) -> float:
    """Coupon-collector upper bound for hitting all ``k`` coupons.

    ``H_k ≈ k · sum_{i=1..k}(1/i)``.  Used as a sanity benchmark in tests
    that simulate cluster rotation under the Q1 rule.
    """
    return k * sum(1.0 / i for i in range(1, k + 1))
