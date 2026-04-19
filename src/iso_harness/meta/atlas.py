"""META-Atlas: Pareto-reward meta-optimizer.

Maintains a Pareto frontier of score/rollouts/tokens trade-offs.
Uses vector rewards instead of scalar to discover the full
efficiency frontier.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from iso_harness.meta.common import (
    MetaEpisode,
    run_iso_with_config,
    update_pareto_frontier,
)
from iso_harness.meta.prompts import load_meta_prompt
from iso_harness.optimizer.formatting import (
    action_space_description,
    format_context,
    format_frontier,
    summarize_frontier,
)
from iso_harness.optimizer.parsing import (
    extract_reasoning,
    parse_config_from_response,
    parse_playbook_from_response,
)

logger = logging.getLogger("iso.meta")


def run_atlas(
    inner_variant: str,
    benchmark: str,
    n_episodes: int,
    meta_lm,
    surrogate_subset_size: int = 20,
    meta_run_id: str | None = None,
) -> list[MetaEpisode]:
    """Run Atlas meta-optimizer with Pareto-reward.

    Args:
        inner_variant: ISO variant for inner loop (e.g., "iso_tide").
        benchmark: Benchmark name.
        n_episodes: Number of meta-episodes to run.
        meta_lm: LLM callable for meta-optimization.
        surrogate_subset_size: Dataset subset size per episode.
        meta_run_id: Optional run ID (auto-generated if None).

    Returns:
        List of MetaEpisode objects.
    """
    meta_run_id = meta_run_id or f"atlas-{str(uuid4())[:8]}"
    episodes = []
    playbook = ""
    frontier = []

    prompt_template = load_meta_prompt("atlas_meta")
    playbook_template = load_meta_prompt("playbook_update")

    for ep_num in range(n_episodes):
        meta_prompt = prompt_template.format(
            playbook=playbook or "(empty)",
            current_frontier=format_frontier(frontier),
            frontier_summary=summarize_frontier(frontier),
            action_space=action_space_description(),
        )

        response = meta_lm(meta_prompt)
        response_text = str(response)

        try:
            config_proposal = parse_config_from_response(response_text)
        except ValueError:
            config_proposal = {}
        reasoning = extract_reasoning(response_text)

        outcome = run_iso_with_config(
            variant=inner_variant,
            benchmark=benchmark,
            subset_size=surrogate_subset_size,
            config_overrides=config_proposal,
        )

        # Pareto reward: vector
        reward = [
            outcome.get("final_score", 0.0),
            -outcome.get("rollouts_consumed", 0),
            -outcome.get("tokens_consumed", 0),
        ]

        episode = MetaEpisode(
            meta_run_id=meta_run_id,
            episode_num=ep_num,
            proposed_config=config_proposal,
            meta_llm_reasoning=reasoning,
            episode_outcome=outcome,
            reward=reward,
            playbook_state=playbook,
        )
        episodes.append(episode)

        # Update frontier
        frontier = update_pareto_frontier(frontier, reward, outcome)

        # Periodic playbook update
        if (ep_num + 1) % 10 == 0:
            playbook_prompt = playbook_template.format(
                current_playbook=playbook or "(empty)",
                recent_episodes=format_context(episodes[-10:]),
            )
            playbook_response = meta_lm(playbook_prompt)
            playbook = parse_playbook_from_response(str(playbook_response))

        logger.info(f"Atlas episode {ep_num}: reward={reward}, frontier_size={len(frontier)}")

    return episodes
