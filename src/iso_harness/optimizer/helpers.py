"""Scoring, DSPy interaction, minibatch sampling, and logging helpers for ISO."""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from statistics import mean, stdev
from typing import Any

logger = logging.getLogger("iso")


# ---------------------------------------------------------------------------
# Logging helpers (spec Section 14.7)
# ---------------------------------------------------------------------------

def log_warning(msg: str) -> None:
    logger.warning(msg)

def log_info(msg: str) -> None:
    logger.info(msg)

def log_error(msg: str) -> None:
    logger.error(msg)


# ---------------------------------------------------------------------------
# Scoring helpers (spec Section 14.1)
# ---------------------------------------------------------------------------

def compute_top3_mean(scores: dict[str, dict]) -> float:
    """Mean of the top-3 candidate mean scores."""
    sorted_means = sorted(
        [s["mean"] for s in scores.values()], reverse=True
    )
    return mean(sorted_means[:3]) if sorted_means else 0.0


def top_k_examples(per_example: dict[str, float], k: int) -> list[tuple[str, float]]:
    """Top-k (example_id, score) by score descending."""
    return sorted(per_example.items(), key=lambda x: x[1], reverse=True)[:k]


def bottom_k_examples(per_example: dict[str, float], k: int) -> list[tuple[str, float]]:
    """Bottom-k (example_id, score) by score ascending."""
    return sorted(per_example.items(), key=lambda x: x[1])[:k]


def find_candidate(pool: list, candidate_id: str):
    """Find candidate by ID in pool, or None."""
    return next((c for c in pool if c.id == candidate_id), None)


def get_all_example_ids(scores: dict[str, dict]) -> set[str]:
    """Collect all example IDs across all candidates' scores."""
    all_ids = set()
    for candidate_scores in scores.values():
        all_ids.update(candidate_scores.get("per_example", {}).keys())
    return all_ids


# ---------------------------------------------------------------------------
# DSPy interaction helpers (spec Section 14.2)
# ---------------------------------------------------------------------------

def is_multi_module(student) -> bool:
    """True if student has >=2 named predictors."""
    return len(list(student.named_predictors())) >= 2


def apply_candidate_prompts(student, candidate) -> Any:
    """
    Return a deep-copied student with each predictor's signature.instructions
    replaced per candidate.prompts_by_module. The original student is unchanged.

    Works with DSPy 2.6.27.
    """
    patched = deepcopy(student)
    for name, predictor in patched.named_predictors():
        if name in candidate.prompts_by_module:
            new_sig = predictor.signature.with_instructions(
                candidate.prompts_by_module[name]
            )
            predictor.signature = new_sig
    return patched


def extract_per_module_outputs(prediction) -> dict[str, Any]:
    """
    Extract per-module output snapshot from a DSPy Prediction.
    Returns a dict of field_name -> value for all output fields.
    """
    if prediction is None:
        return {}
    # DSPy Prediction stores outputs as attributes
    result = {}
    if hasattr(prediction, '_completions'):
        # DSPy 2.x style
        for key in prediction._completions:
            try:
                result[key] = getattr(prediction, key, None)
            except Exception:
                pass
    else:
        # Fallback: iterate over prediction's dict
        for key, value in prediction.items() if hasattr(prediction, 'items') else []:
            result[key] = value
    return result


def as_dspy_module(winner, student) -> Any:
    """Return a deep-copied student with the winning candidate's prompts applied."""
    return apply_candidate_prompts(student, winner)


def ensure_example_ids(examples: list, prefix: str = "ex") -> list:
    """
    Ensure each example has an 'id' attribute. DSPy 2.6.27 Examples don't have .id.
    Assigns f"{prefix}_{i}" if missing. Modifies examples in place and returns them.
    """
    for i, ex in enumerate(examples):
        if not hasattr(ex, 'id') or ex.id is None:
            ex.id = f"{prefix}_{i}"
    return examples


# ---------------------------------------------------------------------------
# Minibatch sampling (spec Section 14.5)
# ---------------------------------------------------------------------------

def sample_minibatches(
    trainset: list,
    n_batches: int,
    batch_size: int,
    rng: random.Random,
) -> list[list]:
    """
    Sample n_batches disjoint minibatches of batch_size from trainset.
    Uses the provided seeded RNG for reproducibility.
    Raises ValueError if trainset is too small to produce disjoint batches.
    """
    total_needed = n_batches * batch_size
    if len(trainset) < total_needed:
        raise ValueError(
            f"Trainset has {len(trainset)} examples; need at least {total_needed}"
        )
    shuffled = list(trainset)
    rng.shuffle(shuffled)
    return [
        shuffled[i * batch_size:(i + 1) * batch_size]
        for i in range(n_batches)
    ]
