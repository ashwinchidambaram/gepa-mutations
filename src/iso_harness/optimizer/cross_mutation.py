"""Cross-mutation strategies for ISO optimizer.

Three strategies: elitist, exploration_preserving, reflector_guided.
All follow the hook signature:
    cross_mutate(pool, scores, pool_improved, config, runtime) -> list[Candidate]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from uuid import uuid4

from iso_harness.optimizer.candidate import Candidate
from iso_harness.optimizer.formatting import format_failures
from iso_harness.optimizer.helpers import find_candidate, log_warning, top_k_examples, bottom_k_examples
from iso_harness.optimizer.parsing import parse_prompts_from_response, parse_pairs_from_response
from iso_harness.optimizer.prompts import load_prompt

logger = logging.getLogger("iso")

__all__ = [
    "PairProposal",
    "build_child",
    "cross_mutate_elitist",
    "cross_mutate_exploration_preserving",
    "cross_mutate_reflector_guided",
    "apply_recombination_rationale",
]


@dataclass
class PairProposal:
    """A proposed parent pair for reflector-guided cross-mutation.

    Attributes:
        parent_a_id: ID of the first parent candidate.
        parent_b_id: ID of the second parent candidate.
        rationale: Natural-language description of how to combine the pair.
    """

    parent_a_id: str
    parent_b_id: str
    rationale: str = ""


# ---------------------------------------------------------------------------
# Shared child-building helper
# ---------------------------------------------------------------------------


def build_child(
    parent_a: Candidate,
    parent_b: Candidate,
    mechanism: str,
    scores: dict[str, dict],
    runtime,
    child_prompts: dict[str, str] | None = None,
) -> Candidate:
    """Create a new Candidate from two parents.

    If ``child_prompts`` is None, calls the reflection LM with the
    ``elitist_cross`` prompt template to blend the two parents.

    Args:
        parent_a: First parent candidate.
        parent_b: Second parent candidate.
        mechanism: Birth mechanism label (e.g. ``"cross_mutation_elitist"``).
        scores: Mapping of candidate_id -> score dict (must contain ``"mean"``).
        runtime: Active ``ISORuntime`` instance.
        child_prompts: Pre-computed blended prompts, or None to generate via LM.

    Returns:
        A fresh ``Candidate`` with a new UUID and both parent IDs recorded.
    """
    if child_prompts is None:
        prompt_template = load_prompt("elitist_cross")
        prompt = prompt_template.format(
            parent_a_prompts=parent_a.prompts_by_module,
            parent_a_score=scores[parent_a.id]["mean"],
            parent_b_prompts=parent_b.prompts_by_module,
            parent_b_score=scores[parent_b.id]["mean"],
        )
        response = runtime.reflection_lm(prompt)
        child_prompts = parse_prompts_from_response(response)

    return Candidate(
        id=str(uuid4()),
        parent_ids=[parent_a.id, parent_b.id],
        birth_round=runtime.round_num,
        birth_mechanism=mechanism,
        skill_category=None,
        prompts_by_module=child_prompts,
        score_history=[],
        per_instance_scores={},
        pareto_frontier_rounds=[],
        death_round=None,
        death_reason=None,
        total_rollouts_consumed=0,
    )


# ---------------------------------------------------------------------------
# 6.1 Elitist cross-mutation (Sprint, Tide, Lens)
# ---------------------------------------------------------------------------


def cross_mutate_elitist(
    pool: list[Candidate],
    scores: dict[str, dict],
    pool_improved: bool,
    config,
    runtime,
) -> list[Candidate]:
    """Top-3 candidates are paired and recombined via reflection.

    Returns new children (not replacing parents).  When
    ``config.hooks.cross_mutate_only_when_improving`` is True, returns an
    empty list if the pool score did not improve this round.

    Args:
        pool: Current candidate pool.
        scores: Mapping of candidate_id -> score dict.
        pool_improved: Whether the best pool score improved this round.
        config: Active ``ISOConfig`` instance.
        runtime: Active ``ISORuntime`` instance.

    Returns:
        List of newly created child candidates (0, 1, or 2 entries).
    """
    if config.hooks.cross_mutate_only_when_improving and not pool_improved:
        return []

    top3 = sorted(pool, key=lambda c: scores[c.id]["mean"], reverse=True)[:3]
    if len(top3) < 2:
        return []

    pairs = [(top3[0], top3[1])]
    if len(top3) >= 3:
        pairs.append((top3[1], top3[2]))

    return [
        build_child(a, b, "cross_mutation_elitist", scores, runtime)
        for a, b in pairs
    ]


# ---------------------------------------------------------------------------
# 6.2 Exploration-preserving cross-mutation (Grove)
# ---------------------------------------------------------------------------


def cross_mutate_exploration_preserving(
    pool: list[Candidate],
    scores: dict[str, dict],
    pool_improved: bool,
    config,
    runtime,
) -> list[Candidate]:
    """Top-3 candidates are crossed with mid-tier candidates (ranks 4-8).

    Maintains diversity by pairing strong candidates with mid-tier ones rather
    than only with each other.  Falls back to elitist if no mid-tier candidates
    exist.

    Args:
        pool: Current candidate pool.
        scores: Mapping of candidate_id -> score dict.
        pool_improved: Whether the best pool score improved this round.
        config: Active ``ISOConfig`` instance.
        runtime: Active ``ISORuntime`` instance.

    Returns:
        List of newly created child candidates.
    """
    ranked = sorted(pool, key=lambda c: scores[c.id]["mean"], reverse=True)
    top3 = ranked[:3]
    mid_tier = ranked[3:8]

    if not mid_tier:
        return cross_mutate_elitist(pool, scores, pool_improved, config, runtime)

    # zip truncates to min(len(top3), len(mid_tier))
    pairs = list(zip(top3, mid_tier))

    return [
        build_child(a, b, "cross_mutation_exploration", scores, runtime)
        for a, b in pairs
    ]


# ---------------------------------------------------------------------------
# 6.3 Reflector-guided cross-mutation (Storm)
# ---------------------------------------------------------------------------


def cross_mutate_reflector_guided(
    pool: list[Candidate],
    scores: dict[str, dict],
    pool_improved: bool,
    config,
    runtime,
) -> list[Candidate]:
    """Ask the reflector LM to propose complementary pairs and recombinations.

    The reflector receives a summary of the top-8 candidates (including their
    strongest and weakest examples) and proposes pairs whose weaknesses
    complement each other.  If the reflector response cannot be parsed, falls
    back to elitist cross-mutation.

    Args:
        pool: Current candidate pool.
        scores: Mapping of candidate_id -> score dict.
        pool_improved: Whether the best pool score improved this round.
        config: Active ``ISOConfig`` instance.
        runtime: Active ``ISORuntime`` instance.

    Returns:
        List of newly created child candidates.
    """
    top_candidates = sorted(
        pool, key=lambda c: scores[c.id]["mean"], reverse=True
    )[:8]

    pool_summary = [
        {
            "candidate_id": c.id,
            "mean_score": scores[c.id]["mean"],
            "strongest_examples": top_k_examples(
                scores[c.id].get("per_example", {}), k=3
            ),
            "weakest_examples": bottom_k_examples(
                scores[c.id].get("per_example", {}), k=3
            ),
        }
        for c in top_candidates
    ]

    prompt_template = load_prompt("reflector_guided_cross")
    prompt = prompt_template.format(
        candidates_summary=json.dumps(pool_summary, indent=2),
        n_pairs_requested=2,
    )
    response = runtime.reflection_lm(prompt)

    try:
        raw_pairs = parse_pairs_from_response(response)
    except ValueError:
        log_warning(
            "cross_mutate_reflector_guided: failed to parse pair proposals; "
            "falling back to elitist"
        )
        return cross_mutate_elitist(pool, scores, pool_improved, config, runtime)

    children: list[Candidate] = []
    for raw_pair in raw_pairs:
        proposal = PairProposal(
            parent_a_id=raw_pair["parent_a_id"],
            parent_b_id=raw_pair["parent_b_id"],
            rationale=raw_pair.get("rationale", ""),
        )

        parent_a = find_candidate(pool, proposal.parent_a_id)
        parent_b = find_candidate(pool, proposal.parent_b_id)

        if parent_a is None or parent_b is None:
            log_warning(
                f"cross_mutate_reflector_guided: proposed pair "
                f"({proposal.parent_a_id!r}, {proposal.parent_b_id!r}) "
                f"contains unknown candidate ID(s); skipping"
            )
            continue

        child_prompts = apply_recombination_rationale(
            parent_a, parent_b, proposal.rationale, runtime
        )
        children.append(
            build_child(
                parent_a,
                parent_b,
                "cross_mutation_reflector_guided",
                scores,
                runtime,
                child_prompts=child_prompts,
            )
        )

    return children


# ---------------------------------------------------------------------------
# Recombination helper (used by reflector-guided strategy)
# ---------------------------------------------------------------------------


def apply_recombination_rationale(
    parent_a: Candidate,
    parent_b: Candidate,
    rationale: str,
    runtime,
) -> dict[str, str]:
    """Use the reflector LM to blend two parents according to a rationale.

    Args:
        parent_a: First parent candidate.
        parent_b: Second parent candidate.
        rationale: Natural-language description of how to combine the parents.
        runtime: Active ``ISORuntime`` instance.

    Returns:
        Blended ``prompts_by_module`` dict for the child candidate.
    """
    prompt_template = load_prompt("recombination_apply")
    prompt = prompt_template.format(
        parent_a_prompts=parent_a.prompts_by_module,
        parent_b_prompts=parent_b.prompts_by_module,
        rationale=rationale,
    )
    response = runtime.reflection_lm(prompt)
    return parse_prompts_from_response(response)
