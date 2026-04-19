"""ISO-specific checkpoint save/load.

Wraps the experiment harness's atomic JSON write with ISO-specific state
(pool, round_num, prev_top3_mean, plateau_rounds, random state).
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict
from pathlib import Path

from iso_harness.optimizer.candidate import Candidate

logger = logging.getLogger("iso")

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
]

# Fields from Candidate that are safe to serialize and needed for resume.
_CANDIDATE_RESUME_FIELDS = frozenset({
    "id",
    "parent_ids",
    "birth_round",
    "birth_mechanism",
    "skill_category",
    "prompts_by_module",
    "score_history",
    "per_instance_scores",
    "pareto_frontier_rounds",
    "death_round",
    "death_reason",
    "total_rollouts_consumed",
})


def save_checkpoint(
    pool: list[Candidate],
    runtime,  # ISORuntime
    prev_top3_mean: float,
    plateau_rounds: int,
) -> None:
    """Write round-boundary checkpoint to runs/{run_id}/checkpoint/.

    Args:
        pool: Current candidate pool.
        runtime: Active ``ISORuntime`` instance (provides run_id, round_num,
            rollout_counter, and seed).
        prev_top3_mean: Top-3 mean score from the previous round (used for
            plateau detection on resume).
        plateau_rounds: Number of consecutive non-improving rounds so far.
    """
    checkpoint_dir = Path(f"runs/{runtime.run_id}/checkpoint")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Serialize candidates, keeping only resume-relevant fields to avoid
    # serialisation errors from non-JSON-safe objects (e.g. DSPy predictions).
    serialized_pool = []
    for c in pool:
        raw = asdict(c)
        filtered = {k: v for k, v in raw.items() if k in _CANDIDATE_RESUME_FIELDS}
        serialized_pool.append(filtered)

    state = {
        "round_num": runtime.round_num,
        "pool": serialized_pool,
        "rollouts_consumed": runtime.rollout_counter.value(),
        "prev_top3_mean": prev_top3_mean,
        "plateau_rounds": plateau_rounds,
        "rng_state": random.getstate(),
        "seed": runtime.seed,
    }

    # Atomic write: write to .tmp then rename, so a partial write never leaves
    # a corrupt checkpoint file.
    path = checkpoint_dir / "iso_state.json"
    tmp = path.with_suffix(".tmp")
    content = json.dumps(state, indent=2, default=str)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))

    logger.info("Checkpoint saved at round %d", runtime.round_num)


def load_checkpoint(run_id: str) -> dict | None:
    """Load latest checkpoint, or None if no checkpoint exists.

    Args:
        run_id: Experiment run identifier used to locate the checkpoint
            directory (``runs/{run_id}/checkpoint/iso_state.json``).

    Returns:
        A dict with keys ``round_num``, ``pool`` (list of ``Candidate``),
        ``rollouts_consumed``, ``prev_top3_mean``, ``plateau_rounds``,
        ``rng_state``, and ``seed``.  Returns ``None`` if no checkpoint
        file is present or if the file cannot be parsed.

    Note:
        Candidates in the returned pool will not have ``prediction`` objects
        (those are not serialized).  They will be re-evaluated in the next
        optimisation round.
    """
    path = Path(f"runs/{run_id}/checkpoint/iso_state.json")
    if not path.exists():
        return None

    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)

        # Reconstruct Candidate objects from the serialized dicts.
        # score_history is stored as list-of-lists in JSON; convert back to
        # list-of-tuples as expected by the optimizer internals.
        reconstructed: list[Candidate] = []
        for raw in state["pool"]:
            raw["score_history"] = [
                tuple(pair) for pair in raw.get("score_history", [])
            ]
            # Keep only fields that Candidate.__init__ accepts, in case the
            # checkpoint was written by a slightly different schema version.
            known = {k: v for k, v in raw.items() if k in _CANDIDATE_RESUME_FIELDS}
            reconstructed.append(Candidate(**known))

        state["pool"] = reconstructed
        return state

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Failed to load checkpoint for %s: %s", run_id, e)
        return None
