"""SCALPEL checkpoint persistence — JSON + LightGBM text format only.

The optimizer's state is split across three artifacts in the target
directory ``path``:

* ``state.json`` — candidates, iteration logs, targeting history, lesson
  book bullets, RNG state, run id, module names, telemetry counters.
* ``surrogate/`` — :meth:`scalpel.surrogate.skip_policy.SkipPolicy.save`
  output (LightGBM ``surrogate.txt`` text format + JSON sidecars).
* ``cluster_state.json`` — failure-clusterer internals (``_embeddings``
  serialized as ``list[list[float]]``, ``_labels`` as ``list[int]`` or
  ``None``, plus the recluster bookkeeping ints).

No binary serialization frameworks are used (the forbidden import set is
asserted at test time).  Numpy arrays travel as JSON ``list[list[float]]``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from scalpel.clustering.targeting import TargetingHistory
from scalpel.optimizer import SCALPEL, Candidate, IterationLog

__all__ = ["save_state", "load_state"]


def _candidate_to_jsonable(c: Candidate) -> dict[str, Any]:
    return json.loads(c.model_dump_json())


def _iteration_to_jsonable(log: IterationLog) -> dict[str, Any]:
    return json.loads(log.model_dump_json())


def save_state(optimizer: SCALPEL, path: str | Path) -> None:
    """Persist the entire optimizer state to a directory."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "run_id": optimizer._run_id,
        "module_names": list(optimizer._module_names),
        "module_round_robin_idx": optimizer._module_round_robin_idx,
        "cumulative_tokens": optimizer._cumulative_tokens,
        "candidates": [_candidate_to_jsonable(c) for c in optimizer._candidates],
        "iteration_logs": [
            _iteration_to_jsonable(log) for log in optimizer._iteration_logs
        ],
        "targeting_history": json.loads(optimizer.targeting_history.model_dump_json()),
        "lesson_book": optimizer.lesson_book.to_jsonable(),
        "rng_state": list(optimizer._rng.getstate()[1]),
        "rng_state_pos": optimizer._rng.getstate()[2],
        # np.random.get_state(legacy=True) returns a 5-tuple at runtime; the
        # stub's union-typed return value isn't narrowed, hence the cast.
        "numpy_rng_state": _np_rng_state_to_jsonable(np.random.get_state(legacy=True)),  # type: ignore[arg-type]
    }

    (p / "state.json").write_text(json.dumps(state, indent=2, default=str))

    # Surrogate (LightGBM text format + JSON sidecars).
    optimizer.surrogate.save(p / "surrogate")

    # Cluster state.
    cluster_state: dict[str, Any] = {
        "embeddings": [emb.tolist() for emb in optimizer.clusterer._embeddings],
        "labels": (
            optimizer.clusterer._labels.tolist()
            if optimizer.clusterer._labels is not None
            else None
        ),
        "last_k": optimizer.clusterer._last_k,
        "last_silhouette": float(optimizer.clusterer._last_silhouette),
        "iters_since_recluster": optimizer.clusterer._iters_since_recluster,
        "pool_size_at_last_recluster": optimizer.clusterer._pool_size_at_last_recluster,
    }
    (p / "cluster_state.json").write_text(json.dumps(cluster_state, indent=2))


def load_state(optimizer: SCALPEL, path: str | Path) -> None:
    """Restore optimizer state from a directory previously written by :func:`save_state`."""
    p = Path(path)
    state = json.loads((p / "state.json").read_text())

    optimizer._run_id = state["run_id"]
    optimizer._module_names = list(state["module_names"])
    optimizer._module_round_robin_idx = int(state["module_round_robin_idx"])
    optimizer._cumulative_tokens = int(state["cumulative_tokens"])

    optimizer._candidates = [
        Candidate.model_validate(c) for c in state["candidates"]
    ]
    optimizer._iteration_logs = [
        IterationLog.model_validate(log) for log in state["iteration_logs"]
    ]
    optimizer.targeting_history = TargetingHistory.model_validate(
        state["targeting_history"]
    )

    # Reset and refill the lesson book.
    optimizer.lesson_book._lessons = []
    optimizer.lesson_book._embeddings = []
    optimizer.lesson_book.from_jsonable(state["lesson_book"])

    # Restore RNG (Python).
    int_state = tuple(int(x) for x in state["rng_state"])
    pos = state["rng_state_pos"]
    optimizer._rng.setstate((3, int_state, pos))

    # Restore numpy RNG.
    np.random.set_state(_np_rng_state_from_jsonable(state["numpy_rng_state"]))

    # Surrogate.
    optimizer.surrogate.load(p / "surrogate")

    # Cluster state.
    cluster_state = json.loads((p / "cluster_state.json").read_text())
    optimizer.clusterer._embeddings = [
        np.asarray(emb, dtype=np.float32) for emb in cluster_state["embeddings"]
    ]
    optimizer.clusterer._labels = (
        np.asarray(cluster_state["labels"])
        if cluster_state["labels"] is not None
        else None
    )
    optimizer.clusterer._last_k = cluster_state["last_k"]
    optimizer.clusterer._last_silhouette = float(cluster_state["last_silhouette"])
    optimizer.clusterer._iters_since_recluster = int(
        cluster_state["iters_since_recluster"]
    )
    optimizer.clusterer._pool_size_at_last_recluster = int(
        cluster_state["pool_size_at_last_recluster"]
    )


# ----------------------------------------------------------------- numpy rng


def _np_rng_state_to_jsonable(state: tuple) -> dict[str, Any]:
    """Convert a numpy legacy RandomState tuple to a JSON-friendly dict.

    NumPy's ``np.random.get_state()`` returns a tuple
    ``("MT19937", uint32-keys-array, pos, has_gauss, cached_gaussian)``.
    """
    name = state[0]
    keys = state[1].tolist() if hasattr(state[1], "tolist") else list(state[1])
    return {
        "name": name,
        "keys": keys,
        "pos": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _np_rng_state_from_jsonable(d: dict[str, Any]) -> tuple:
    return (
        d["name"],
        np.asarray(d["keys"], dtype=np.uint32),
        int(d["pos"]),
        int(d["has_gauss"]),
        float(d["cached_gaussian"]),
    )


_ = datetime  # keep import for type-hint compatibility in future revisions
