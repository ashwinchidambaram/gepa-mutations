"""Merge operator for ISO optimizer.

Module-level crossover: child inherits each module from whichever parent
scored better on that module specifically. No-op for single-module systems.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from statistics import mean

from iso_harness.optimizer.candidate import Candidate
from iso_harness.optimizer.helpers import get_all_example_ids

logger = logging.getLogger("iso")


def merge_candidates(
    parents: list[Candidate],
    scores: dict[str, dict],
    runtime,  # ISORuntime
) -> Candidate | None:
    """Module-level crossover: child inherits each module from better parent.

    Args:
        parents: Expected length 2.
        scores: Full score dict from evaluate_pool_multi_minibatch.
        runtime: ISORuntime.

    Returns:
        A new Candidate, or None for single-module systems or invalid input.
    """
    if len(parents) != 2:
        return None

    parent_a, parent_b = parents
    module_names = list(parent_a.prompts_by_module.keys())

    if len(module_names) <= 1:
        return None  # Single-module — Merge is a no-op

    # Compute per-module scores
    module_scores = compute_per_module_scores(parent_a, parent_b, scores, runtime)

    # Per-module selection
    child_prompts = {}
    for module_name in module_names:
        a_score = module_scores[parent_a.id].get(module_name, 0.0)
        b_score = module_scores[parent_b.id].get(module_name, 0.0)
        if a_score >= b_score:
            child_prompts[module_name] = parent_a.prompts_by_module[module_name]
        else:
            child_prompts[module_name] = parent_b.prompts_by_module[module_name]

    return Candidate(
        parent_ids=[parent_a.id, parent_b.id],
        birth_round=runtime.round_num,
        birth_mechanism="merge",
        prompts_by_module=child_prompts,
    )


def compute_per_module_scores(
    parent_a: Candidate,
    parent_b: Candidate,
    scores: dict[str, dict],
    runtime,
) -> dict[str, dict[str, float]]:
    """Per-module scoring from evaluation metadata.

    Reads per_module_score from per_example_metadata if available.
    Falls back to whole-system inheritance (assign mean to all modules).
    """
    results: dict[str, dict[str, float]] = {parent_a.id: {}, parent_b.id: {}}
    module_names = list(parent_a.prompts_by_module.keys())

    for parent in [parent_a, parent_b]:
        per_module_accumulator: dict[str, list[float]] = defaultdict(list)

        metadata = scores.get(parent.id, {}).get("per_example_metadata", {})
        for example_id, ex_metadata in metadata.items():
            per_module = ex_metadata.get("per_module_score", {})
            for module_name, module_score in per_module.items():
                per_module_accumulator[module_name].append(module_score)

        if per_module_accumulator:
            results[parent.id] = {
                m: mean(per_module_accumulator.get(m, [scores[parent.id]["mean"]]))
                for m in module_names
            }
        else:
            # Fallback: whole-system inheritance (assign mean to all modules)
            parent_mean = scores.get(parent.id, {}).get("mean", 0.0)
            results[parent.id] = {m: parent_mean for m in module_names}

    return results


def top_pareto_candidates(
    pool: list[Candidate],
    scores: dict[str, dict],
    n: int,
) -> list[Candidate]:
    """Return top-n candidates on the Pareto frontier.

    A candidate is on the frontier if it's best on at least one example.
    """
    frontier: set[str] = set()
    all_example_ids = get_all_example_ids(scores)

    for example_id in all_example_ids:
        best = max(
            pool,
            key=lambda c: scores.get(c.id, {}).get("per_example", {}).get(example_id, -1),
        )
        frontier.add(best.id)

    frontier_candidates = [c for c in pool if c.id in frontier]
    frontier_candidates.sort(
        key=lambda c: scores.get(c.id, {}).get("mean", 0), reverse=True
    )
    return frontier_candidates[:n]
