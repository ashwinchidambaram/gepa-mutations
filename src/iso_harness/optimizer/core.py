"""Core ISO optimization loop.

iso_compile() is the main entry point called by ISO.compile().
"""

from __future__ import annotations

import logging
from uuid import uuid4

from iso_harness.optimizer.candidate import Candidate, MutationProposal
from iso_harness.optimizer.evaluation import evaluate_pool_multi_minibatch, evaluate_on_valset
from iso_harness.optimizer.helpers import (
    apply_candidate_prompts,
    compute_top3_mean,
    is_multi_module,
    log_info,
    log_warning,
    sample_minibatches,
)
from iso_harness.optimizer.merge import merge_candidates, top_pareto_candidates
from iso_harness.optimizer.runtime import set_current_runtime
from iso_harness.optimizer.skill_discovery import (
    discover_skills,
    instantiate_candidate_from_skill,
    mutate_candidate,
)
from iso_harness.experiment.context import set_context

logger = logging.getLogger("iso")


def iso_compile(student, trainset, valset, config, runtime):
    """Core ISO optimization loop.

    Args:
        student: DSPy module to optimize.
        trainset: Training examples.
        valset: Validation examples.
        config: ISOConfig with hooks and hyperparameters.
        runtime: ISORuntime with LMs, metric, trace store, etc.

    Returns:
        Optimized dspy.Module with winning candidate's prompts applied.
    """
    set_current_runtime(runtime)
    set_context(run_id=runtime.run_id, phase="optimization")

    # --- Phase I: Skill discovery ---
    log_info("Phase I: Skill discovery")
    skill_clusters = discover_skills(
        student=student,
        trainset=trainset,
        n_discovery_examples=config.n_discovery_examples,
        runtime=runtime,
    )
    log_info(f"Discovered {len(skill_clusters)} skill clusters")

    # Instantiate one candidate per skill cluster
    initial_pool = []
    for skill in skill_clusters:
        candidate = instantiate_candidate_from_skill(skill, student, runtime)
        initial_pool.append(candidate)

    # Mutate each skill-candidate into mutations_per_seed variants to grow the pool.
    # With default mutations_per_seed=2, each seed produces 3 total candidates
    # (1 original + 2 mutations). Pool after this step: 9-24 candidates (3-8 seeds × 3).
    pool = []
    for candidate in initial_pool:
        pool.append(candidate)
        for _ in range(config.mutations_per_seed):
            pool.append(mutate_candidate(candidate, scope="independent", runtime=runtime))

    log_info(f"Initial pool size: {len(pool)}")

    # --- Phase II: Tournament rounds ---
    log_info("Phase II: Tournament rounds")
    runtime.round_num = 0
    prev_top3_mean = 0.0
    plateau_rounds = 0

    while not terminate(config, runtime, plateau_rounds, pool):
        runtime.round_num += 1
        set_context(round_num=runtime.round_num)
        log_info(f"Round {runtime.round_num}: pool size = {len(pool)}")

        # Step 1: Multi-minibatch evaluation
        minibatches = sample_minibatches(
            trainset,
            n_batches=config.minibatch_count,
            batch_size=config.minibatch_size,
            rng=runtime.rng,
        )
        scores_by_candidate = evaluate_pool_multi_minibatch(
            pool, minibatches, student, runtime,
        )

        # Step 2: Round statistics
        top3_mean = compute_top3_mean(scores_by_candidate)
        pool_improved = top3_mean > prev_top3_mean
        log_info(
            f"Round {runtime.round_num}: top3_mean={top3_mean:.4f} "
            f"(prev={prev_top3_mean:.4f}, {'improved' if pool_improved else 'stalled'})"
        )

        # Step 3: Pruning (variant-specific)
        pool = config.hooks.prune(
            pool, scores_by_candidate, pool_improved,
            prev_top3_mean, top3_mean, config, runtime,
        )
        log_info(f"After pruning: {len(pool)} candidates")

        # Check for plateau
        if abs(top3_mean - prev_top3_mean) < config.plateau_tolerance:
            plateau_rounds += 1
        else:
            plateau_rounds = 0

        # Step 4: Reflection (variant-specific)
        mutation_proposals = config.hooks.reflect(
            pool, scores_by_candidate, runtime.round_num, config, prev_top3_mean, runtime,
        )
        pool = apply_mutations(pool, mutation_proposals, runtime)

        # Step 5: Cross-mutation (variant-specific)
        children = config.hooks.cross_mutate(
            pool, scores_by_candidate, pool_improved, config, runtime,
        )
        # Cap pool growth per round
        if config.hooks.pool_size_max:
            children = children[:config.hooks.pool_size_max]
        pool.extend(children)

        # Step 6: Merge (shared across variants, multi-module only)
        if runtime.round_num % config.merge_interval == 0 and is_multi_module(student):
            merged = merge_candidates(
                top_pareto_candidates(pool, scores_by_candidate, n=2),
                scores_by_candidate,
                runtime,
            )
            if merged:
                pool.append(merged)

        # Update for next iteration
        prev_top3_mean = top3_mean

        # Budget check
        if runtime.rollout_counter.value() >= config.budget:
            log_info(f"Budget exhausted at round {runtime.round_num}")
            break

    # --- Phase III: Validation and winner selection ---
    log_info("Phase III: Validation and winner selection")
    final_scores = evaluate_on_valset(pool, valset, student, runtime)

    if not final_scores:
        log_warning("No candidates survived to validation")
        return student  # Return unmodified student

    winner_id = max(final_scores, key=final_scores.get)
    winner = next(c for c in pool if c.id == winner_id)
    log_info(f"Winner: {winner_id} with val score {final_scores[winner_id]:.4f}")

    return apply_candidate_prompts(student, winner)


def terminate(config, runtime, plateau_rounds, pool) -> bool:
    """Check termination conditions."""
    if runtime.rollout_counter.value() >= config.budget:
        return True
    if len(pool) <= config.pool_floor:
        return True
    if plateau_rounds >= config.plateau_rounds_threshold:
        return True
    if runtime.round_num >= config.max_rounds:
        return True
    return False


def apply_mutations(
    pool: list[Candidate],
    proposals: list[MutationProposal],
    runtime,
) -> list[Candidate]:
    """Apply mutation proposals by creating new candidates and retiring parents.

    Per spec Section 2.4: creates NEW Candidate objects (not in-place mutation).
    Score history is NOT carried over. Parent is retired with death_reason.
    """
    proposals_by_id = {p.candidate_id: p for p in proposals}
    new_pool = []
    for candidate in pool:
        if candidate.id in proposals_by_id:
            proposal = proposals_by_id[candidate.id]
            new_candidate = Candidate(
                id=str(uuid4()),
                parent_ids=[candidate.id],
                birth_round=runtime.round_num,
                birth_mechanism=f"mutation_{proposal.mechanism}",
                skill_category=candidate.skill_category,
                prompts_by_module={
                    **candidate.prompts_by_module,
                    **proposal.new_prompts,
                },
                score_history=[],
                per_instance_scores={},
                pareto_frontier_rounds=[],
                death_round=None,
                death_reason=None,
                total_rollouts_consumed=0,
            )
            # Mark parent as retired rather than pruned
            candidate.death_round = runtime.round_num
            candidate.death_reason = "retired_for_mutation"
            new_pool.append(new_candidate)
        else:
            new_pool.append(candidate)
    return new_pool
