"""Reflection mechanisms for ISO optimizer.

Four scopes: per_candidate, population_level, pair_contrastive, hybrid.
All return list[MutationProposal] following the hook signature.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from statistics import median

from iso_harness.optimizer.candidate import Candidate, MutationProposal
from iso_harness.optimizer.formatting import format_failures, format_traces, sample_prompts
from iso_harness.optimizer.helpers import compute_top3_mean, find_candidate, log_warning
from iso_harness.optimizer.parsing import parse_prompts_from_response, parse_insights_from_response
from iso_harness.optimizer.prompts import load_prompt

logger = logging.getLogger("iso")


def reflect_per_candidate(
    pool: list[Candidate],
    scores: dict[str, dict],
    round_num: int,
    config,
    prev_top3_mean: float,
    runtime,
) -> list[MutationProposal]:
    """Per-candidate reflection (Grove, Tide, Lens).

    Each candidate is reflected on independently using its own worst-performing
    traces.  A ``MutationProposal`` is emitted for every candidate whose
    reflection succeeds; candidates whose reflection call fails are skipped
    (no-op for that candidate this round).
    """
    proposals: list[MutationProposal] = []
    prompt_template = load_prompt("per_candidate_reflection")

    for candidate in pool:
        # Identify worst-performing examples for this candidate
        candidate_scores = scores[candidate.id]["per_example"]
        worst_examples = sorted(candidate_scores.items(), key=lambda x: x[1])[:3]

        # Retrieve traces for worst examples from trace store
        traces = [
            runtime.trace_store.get(candidate.id, example_id)
            for example_id, _ in worst_examples
        ]
        traces = [t for t in traces if t is not None]

        prompt = prompt_template.format(
            current_prompts=candidate.prompts_by_module,
            traces=format_traces(traces),
            mean_score=scores[candidate.id]["mean"],
            score_history=candidate.score_history,
        )

        try:
            response = runtime.reflection_lm(prompt)
            new_prompts = parse_prompts_from_response(response)
        except (ValueError, RuntimeError) as e:
            log_warning(f"Reflection failed for candidate {candidate.id}: {e}")
            continue

        proposals.append(MutationProposal(
            candidate_id=candidate.id,
            new_prompts=new_prompts,
            mechanism="per_candidate",
        ))

    return proposals


def reflect_population_level(
    pool: list[Candidate],
    scores: dict[str, dict],
    round_num: int,
    config,
    prev_top3_mean: float,
    runtime,
) -> list[MutationProposal]:
    """Population-level reflection (Sprint, Storm-when-stalling).

    One reflection call sees failures aggregated across all candidates; the
    resulting mutation is applied to every pool candidate.  Returns an empty
    list if the reflection LM call or parse fails.
    """
    # Aggregate: which examples does the pool as a whole fail on?
    pool_example_scores: dict[str, list[float]] = defaultdict(list)
    for candidate in pool:
        for example_id, score in scores[candidate.id]["per_example"].items():
            pool_example_scores[example_id].append(score)

    # Pool failures: examples where median candidate scored < 0.5
    pool_failures = {
        ex_id: scores_list
        for ex_id, scores_list in pool_example_scores.items()
        if median(scores_list) < 0.5
    }

    # Get traces for 5 most-failed examples (use median candidate's trace —
    # any candidate's trace reveals the example's structure)
    most_failed = sorted(
        pool_failures.items(), key=lambda x: median(x[1])
    )[:5]

    # Find the candidate whose score on this example is closest to the median
    traces = []
    for ex_id, score_list in most_failed:
        median_score = median(score_list)
        # Pick the candidate whose score is closest to the median
        closest_candidate = min(
            pool,
            key=lambda c: abs(
                scores[c.id]["per_example"].get(ex_id, 0.0) - median_score
            ),
        )
        trace = runtime.trace_store.get(closest_candidate.id, ex_id)
        if trace is not None:
            traces.append(trace)

    prompt_template = load_prompt("population_reflection")
    prompt = prompt_template.format(
        n_candidates=len(pool),
        failure_traces=format_traces(traces),
        pool_prompts_sample=sample_prompts(pool, n=3, rng=runtime.rng),
    )

    try:
        response = runtime.reflection_lm(prompt)
        new_prompts = parse_prompts_from_response(response)
    except (ValueError, RuntimeError) as e:
        log_warning(f"Population reflection failed: {e}")
        return []

    # Apply the same mutation to all candidates
    return [
        MutationProposal(
            candidate_id=c.id,
            new_prompts=new_prompts,
            mechanism="population_level",
        )
        for c in pool
    ]


def reflect_pair_contrastive(
    pool: list[Candidate],
    scores: dict[str, dict],
    round_num: int,
    config,
    prev_top3_mean: float,
    runtime,
) -> list[MutationProposal]:
    """Pair-contrastive reflection (Lens).

    Compares the top improver and top regressor across the last two rounds to
    extract transferable insights, then applies those insights per-candidate.
    Falls back to ``reflect_per_candidate`` when there is insufficient history
    or no clear improver/regressor pair.

    Note: "stable" candidates are intentionally excluded — they add no signal
    that the reflector can usefully exploit.
    """
    if round_num < 2:
        # Need at least 2 rounds of history to identify improvers/regressors
        return reflect_per_candidate(
            pool, scores, round_num, config, prev_top3_mean, runtime,
        )

    # Compute score deltas since last round.
    # Only candidates that survived without mutation across the last two rounds
    # will have score_history[-2]; newly-born candidates won't have enough history.
    deltas = {
        c.id: scores[c.id]["mean"] - c.score_history[-2][1]
        for c in pool
        if len(c.score_history) >= 2
    }

    if not deltas:
        return reflect_per_candidate(
            pool, scores, round_num, config, prev_top3_mean, runtime,
        )

    # Identify top improver and top regressor
    improver_id = max(deltas, key=deltas.get)
    regressor_id = min(deltas, key=deltas.get)

    if deltas[improver_id] <= 0 or deltas[regressor_id] >= 0:
        # No clear improver-regressor pair
        return reflect_per_candidate(
            pool, scores, round_num, config, prev_top3_mean, runtime,
        )

    improver = find_candidate(pool, improver_id)
    regressor = find_candidate(pool, regressor_id)

    # Retrieve worst-example traces for each via trace store
    improver_traces = runtime.trace_store.get_worst_for_candidate(
        improver.id, scores[improver.id]["per_example"], n=3,
    )
    regressor_traces = runtime.trace_store.get_worst_for_candidate(
        regressor.id, scores[regressor.id]["per_example"], n=3,
    )

    contrastive_template = load_prompt("pair_contrastive")
    contrastive_prompt = contrastive_template.format(
        improver_prompts=improver.prompts_by_module,
        improver_delta=deltas[improver_id],
        improver_traces=format_traces(improver_traces),
        regressor_prompts=regressor.prompts_by_module,
        regressor_delta=deltas[regressor_id],
        regressor_traces=format_traces(regressor_traces),
    )

    try:
        response = runtime.reflection_lm(contrastive_prompt)
        insights = parse_insights_from_response(response)
    except (ValueError, RuntimeError) as e:
        log_warning(f"Pair-contrastive reflection failed: {e}")
        return reflect_per_candidate(
            pool, scores, round_num, config, prev_top3_mean, runtime,
        )
    # insights contains: what_worked, what_failed, recommended_changes

    # Apply per-candidate using insights
    apply_template = load_prompt("pair_apply")
    proposals: list[MutationProposal] = []
    for candidate in pool:
        per_candidate_prompt = apply_template.format(
            current_prompts=candidate.prompts_by_module,
            insights=insights,
            candidate_score=scores[candidate.id]["mean"],
        )
        try:
            response = runtime.reflection_lm(per_candidate_prompt)
            new_prompts = parse_prompts_from_response(response)
        except (ValueError, RuntimeError):
            continue  # skip this candidate, keep its current prompts

        proposals.append(MutationProposal(
            candidate_id=candidate.id,
            new_prompts=new_prompts,
            mechanism="pair_contrastive",
        ))

    return proposals


def reflect_hybrid(
    pool: list[Candidate],
    scores: dict[str, dict],
    round_num: int,
    config,
    prev_top3_mean: float,
    runtime,
) -> list[MutationProposal]:
    """Hybrid reflection (Storm).

    Switches between per-candidate and population-level reflection based on
    whether the pool is currently improving:

    - Improving (top3_mean > prev_top3_mean): use per-candidate to preserve
      diversity and capitalise on individual gains.
    - Stalling (top3_mean <= prev_top3_mean): use population-level to
      coordinate a collective jump out of the plateau.
    """
    top3_mean = compute_top3_mean(scores)

    if top3_mean > prev_top3_mean:
        # Pool improving — preserve diversity with per-candidate reflection
        return reflect_per_candidate(
            pool, scores, round_num, config, prev_top3_mean, runtime,
        )
    else:
        # Pool stalling — use population-level to coordinate a jump
        return reflect_population_level(
            pool, scores, round_num, config, prev_top3_mean, runtime,
        )
