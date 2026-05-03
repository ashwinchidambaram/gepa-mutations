"""SCALPEL Phase 7 — 512-dim feature builder for the rollout-skip surrogate.

Per addendum Q5 in ``docs/scalpel/SCALPEL.md``: each ``(candidate, instance)``
pair is mapped to a 512-d float32 vector with the following block layout:

    [0:6]      span one-hot (multi-hot if the edit list touches multiple spans)
    [6:70]     hashed unigrams + bigrams of edit content (mmh3 mod 64)
    [70:78]    cluster id one-hot (cluster_id clamped to [0, 7])
    [78:79]    cluster_centroid_score
    [79:81]    (parent_pareto_score, parent_on_cluster_score)
    [81:465]   trace_embedding (384 dims, BGE-small failure-trace embedding)
    [465:512]  reserved zeros (for v2 self-consistency credit features)

The feature schema version is bumped whenever the block layout changes; the
hash seed is fixed so featurization is deterministic across runs.
"""

from __future__ import annotations

import mmh3
import numpy as np

from scalpel.edits.grammar import SPAN_IDS, Edit

__all__ = [
    "CLUSTER_ID_DIM",
    "CLUSTER_SCORE_DIM",
    "EDIT_BIGRAM_DIM",
    "EDIT_SPAN_DIM",
    "FEATURE_SCHEMA_VERSION",
    "HASH_SEED",
    "PARENT_SCORE_DIM",
    "RESERVED_DIM",
    "TOTAL_DIM",
    "TRACE_EMB_DIM",
    "featurize",
]


EDIT_SPAN_DIM = 6  # S1..S6 one-hot
EDIT_BIGRAM_DIM = 64  # hashed bigrams of edit payload
CLUSTER_ID_DIM = 8  # one-hot of target cluster (max 8 clusters)
CLUSTER_SCORE_DIM = 1  # parent score on this cluster
PARENT_SCORE_DIM = 2  # (parent_pareto, parent_on_cluster)
TRACE_EMB_DIM = 384  # BGE-small failure-trace embedding
RESERVED_DIM = 47  # reserved for v2 self-consistency credit features
TOTAL_DIM = 512

assert (
    EDIT_SPAN_DIM
    + EDIT_BIGRAM_DIM
    + CLUSTER_ID_DIM
    + CLUSTER_SCORE_DIM
    + PARENT_SCORE_DIM
    + TRACE_EMB_DIM
    + RESERVED_DIM
    == TOTAL_DIM
)

HASH_SEED = 42
FEATURE_SCHEMA_VERSION = 1


def featurize(
    edits: list[Edit],
    cluster_id: int,
    cluster_centroid_score: float,
    parent_pareto_score: float,
    parent_on_cluster_score: float,
    trace_embedding: np.ndarray,
) -> np.ndarray:
    """Build the 512-dim feature vector for a (candidate, instance) pair.

    See the module docstring for the block layout.  The trace embedding is
    expected to be a 384-d float vector; it is cast to ``float32`` and copied
    in verbatim.
    """
    feat = np.zeros(TOTAL_DIM, dtype=np.float32)

    # Block 1: span one-hot (multi-hot if multiple edits target different spans).
    for e in edits:
        idx = SPAN_IDS.index(e.target_span)
        feat[idx] = 1.0
    base = EDIT_SPAN_DIM

    # Block 2: hashed unigrams + bigrams of edit content (mmh3 mod 64).
    for e in edits:
        toks = e.content.split()
        for i, t in enumerate(toks):
            feat[base + (mmh3.hash(t, HASH_SEED) % EDIT_BIGRAM_DIM)] += 1.0
            if i + 1 < len(toks):
                bg = f"{t} {toks[i + 1]}"
                feat[base + (mmh3.hash(bg, HASH_SEED) % EDIT_BIGRAM_DIM)] += 1.0
    base += EDIT_BIGRAM_DIM

    # Block 3: cluster id one-hot, clamped to [0, 7].
    cid = max(0, min(7, int(cluster_id)))
    feat[base + cid] = 1.0
    base += CLUSTER_ID_DIM

    # Block 4: cluster centroid score (scalar).
    feat[base] = float(cluster_centroid_score)
    base += CLUSTER_SCORE_DIM

    # Block 5: parent scores (pareto, on-cluster).
    feat[base] = float(parent_pareto_score)
    feat[base + 1] = float(parent_on_cluster_score)
    base += PARENT_SCORE_DIM

    # Block 6: trace embedding (384 dims).
    if trace_embedding.shape != (TRACE_EMB_DIM,):
        raise ValueError(
            f"trace_embedding must be ({TRACE_EMB_DIM},); got {trace_embedding.shape}"
        )
    feat[base : base + TRACE_EMB_DIM] = trace_embedding.astype(np.float32)
    # Block 7: reserved (already zero-initialized).

    return feat
