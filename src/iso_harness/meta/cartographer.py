"""META-Cartographer: meta-optimizer with persistent playbook.

Maintains a playbook of optimization knowledge that is updated
every N episodes. The playbook provides persistent memory across
the meta-optimization run.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from iso_harness.meta.common import MetaEpisode, compute_constrained_reward, run_iso_with_config
from iso_harness.meta.prompts import load_meta_prompt
from iso_harness.optimizer.formatting import format_context, action_space_description
from iso_harness.optimizer.parsing import parse_config_from_response, extract_reasoning, parse_playbook_from_response

logger = logging.getLogger("iso.meta")


def run_cartographer(
    inner_variant: str,
    benchmark: str,
    n_episodes: int,
    meta_lm,
    surrogate_subset_size: int = 20,
    playbook_update_interval: int = 10,
    meta_run_id: str | None = None,
) -> list[MetaEpisode]:
    """Run Cartographer meta-optimizer with persistent playbook.

    Args:
        inner_variant: ISO variant for inner loop (e.g., "iso_tide").
        benchmark: Benchmark name.
        n_episodes: Number of meta-episodes to run.
        meta_lm: LLM callable for meta-optimization.
        surrogate_subset_size: Dataset subset size per episode.
        playbook_update_interval: How often (in episodes) to update the playbook.
        meta_run_id: Optional run ID (auto-generated if None).

    Returns:
        List of MetaEpisode objects.
    """
    meta_run_id = meta_run_id or f"cartographer-{str(uuid4())[:8]}"
    episodes = []
    playbook = ""

    prompt_template = load_meta_prompt("cartographer_meta")

    for ep_num in range(n_episodes):
        meta_prompt = prompt_template.format(
            playbook=playbook or "(empty — no knowledge yet)",
            recent_episodes=format_context(episodes[-5:]),
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

        reward = compute_constrained_reward(outcome)

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

        # Update playbook every N episodes
        if (ep_num + 1) % playbook_update_interval == 0:
            playbook = _update_playbook(playbook, episodes[-playbook_update_interval:], meta_lm)

        logger.info(f"Cartographer episode {ep_num}: reward={reward:.4f}")

    return episodes


def _update_playbook(
    current_playbook: str,
    recent_episodes: list[MetaEpisode],
    meta_lm,
) -> str:
    """Update the playbook with insights from recent episodes."""
    prompt_template = load_meta_prompt("playbook_update")
    prompt = prompt_template.format(
        current_playbook=current_playbook or "(empty)",
        recent_episodes=format_context(recent_episodes),
    )
    response = meta_lm(prompt)
    return parse_playbook_from_response(str(response))
