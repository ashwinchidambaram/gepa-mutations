"""Online mini-batch k-means with silhouette-driven k for SCALPEL.

Phase 5 of SCALPEL.  Implements :class:`FailureClusterer`, the engine that
groups embedded ``(instance, trace, feedback)`` failures into 4–8 clusters
using ``sklearn.cluster.MiniBatchKMeans`` and selects ``k`` adaptively by
silhouette score (see ``docs/scalpel/SCALPEL.md`` §5.5).  Reclustering is
triggered every ``recluster_every_iters`` SCALPEL outer-loop iterations or
whenever the failure pool grows by ``recluster_pool_growth_frac`` since the
last clustering — whichever comes first.

Public surface:

* :class:`ClusterState` — frozen pydantic snapshot of one cluster.
* :class:`FailureClusterer` — incremental clusterer with adaptive ``k``.
"""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from scalpel.clustering.embeddings import EMBEDDING_DIM

__all__ = ["ClusterState", "FailureClusterer"]

logger = logging.getLogger(__name__)

LOW_SILHOUETTE_WARN_THRESHOLD = 0.1


class ClusterState(BaseModel):
    """Snapshot of a single cluster at the current iteration."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    id: int
    centroid: list[float]  # 384 floats; serialized as JSON later.
    failure_count: int  # number of failures currently routed to this cluster.
    summary: str = ""  # ≤25-token natural-language summary; filled later.


class FailureClusterer:
    """Online failure clusterer with silhouette-driven adaptive ``k``."""

    def __init__(
        self,
        k_min: int = 4,
        k_max: int = 8,
        batch_size: int = 64,
        seed: int = 0,
        recluster_every_iters: int = 8,
        recluster_pool_growth_frac: float = 0.5,
    ) -> None:
        self.k_min = k_min
        self.k_max = k_max
        self.batch_size = batch_size
        self.seed = seed
        self.recluster_every_iters = recluster_every_iters
        self.recluster_pool_growth_frac = recluster_pool_growth_frac

        # State.
        self._embeddings: list[np.ndarray] = []
        self._labels: np.ndarray | None = None
        self._km: MiniBatchKMeans | None = None
        self._iters_since_recluster: int = 0
        self._pool_size_at_last_recluster: int = 0
        self._last_silhouette: float = 0.0
        self._last_k: int | None = None

    # ------------------------------------------------------------------ ingest
    def add(self, embeddings: np.ndarray) -> None:
        """Append new failure embeddings.  Does not recluster."""
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Expected (N, {EMBEDDING_DIM}) embeddings, got shape {embeddings.shape}"
            )
        for row in embeddings:
            self._embeddings.append(row.astype(np.float32))

    def step_iteration(self) -> None:
        """Increment the iteration counter (call once per SCALPEL outer-loop iter)."""
        self._iters_since_recluster += 1

    # ----------------------------------------------------------------- trigger
    def should_recluster(self) -> bool:
        """Return whether the recluster trigger has fired.

        Triggers (any one is sufficient):

        * Pool is non-empty and we have not clustered yet.
        * ``self._iters_since_recluster >= self.recluster_every_iters``.
        * Pool grew by ``>= self.recluster_pool_growth_frac`` since the
          last clustering.
        """
        if not self._embeddings:
            return False
        if self._km is None:
            return True
        if self._iters_since_recluster >= self.recluster_every_iters:
            return True
        prev = max(1, self._pool_size_at_last_recluster)
        growth = (len(self._embeddings) - self._pool_size_at_last_recluster) / prev
        return growth >= self.recluster_pool_growth_frac

    # ---------------------------------------------------------------- cluster
    def recluster(self) -> int:
        """Run silhouette-driven k-selection in ``[k_min, k_max]`` and refit.

        Returns the chosen ``k``.  Idempotent: safe to call when the trigger
        has not fired.  Falls back to ``k=1`` if the pool is too small for a
        meaningful ``k_min``-cluster fit, and to ``k=k_min`` if no candidate
        ``k`` produced more than a single occupied label.
        """
        if not self._embeddings:
            self._labels = None
            return 0
        X = np.asarray(self._embeddings)
        n = len(X)

        if n < max(self.k_min, 4):
            km = MiniBatchKMeans(
                n_clusters=1,
                batch_size=self.batch_size,
                random_state=self.seed,
                n_init=3,
            ).fit(X)
            self._km = km
            self._labels = km.labels_
            self._last_k = 1
            self._last_silhouette = 0.0
            self._iters_since_recluster = 0
            self._pool_size_at_last_recluster = n
            return 1

        best_k: int = self.k_min
        best_s: float = -1.0
        best_km: MiniBatchKMeans | None = None
        for k in range(self.k_min, self.k_max + 1):
            if k >= n:
                break
            km = MiniBatchKMeans(
                n_clusters=k,
                batch_size=self.batch_size,
                random_state=self.seed,
                n_init=3,
            ).fit(X)
            if len(set(km.labels_)) < 2:
                continue
            s = float(silhouette_score(X, km.labels_, sample_size=min(2000, n)))
            if s > best_s:
                best_k, best_s, best_km = k, s, km

        if best_km is None:
            best_km = MiniBatchKMeans(
                n_clusters=self.k_min,
                batch_size=self.batch_size,
                random_state=self.seed,
                n_init=3,
            ).fit(X)
            best_k = self.k_min
            best_s = 0.0

        if best_s < LOW_SILHOUETTE_WARN_THRESHOLD:
            logger.warning(
                "low silhouette score %.3f at k=%d; clusters may be poorly separated",
                best_s,
                best_k,
            )

        self._km = best_km
        self._labels = best_km.labels_
        self._last_k = best_k
        self._last_silhouette = best_s
        self._iters_since_recluster = 0
        self._pool_size_at_last_recluster = n
        return best_k

    # --------------------------------------------------------------- snapshot
    def cluster_states(self) -> list[ClusterState]:
        """Return per-cluster snapshots for the current k-means model."""
        if self._km is None or self._labels is None:
            return []
        states: list[ClusterState] = []
        for cid in range(self._km.n_clusters):
            mask = self._labels == cid
            count = int(mask.sum())
            states.append(
                ClusterState(
                    id=cid,
                    centroid=self._km.cluster_centers_[cid].tolist(),
                    failure_count=count,
                )
            )
        return states

    # ---------------------------------------------------------------- props
    @property
    def last_silhouette(self) -> float:
        return self._last_silhouette

    @property
    def last_k(self) -> int | None:
        return self._last_k
