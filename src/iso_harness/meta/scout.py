"""META-Scout: in-context-only meta-optimizer.

Uses a sliding window of recent episodes for in-context learning.
No persistent state between episodes beyond the context window.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from iso_harness.meta.common import MetaEpisode, compute_constrained_reward, run_iso_with_config
from iso_harness.meta.prompts import load_meta_prompt
from iso_harness.optimizer.formatting import format_context, action_space_description
from iso_harness.optimizer.parsing import parse_config_from_response, extract_reasoning

logger = logging.getLogger("iso.meta")


def run_scout(
    inner_variant: str,
    benchmark: str,
    n_episodes: int,
    meta_lm,
    surrogate_subset_size: int = 20,
    meta_run_id: str | None = None,
) -> list[MetaEpisode]:
    """Run Scout meta-optimizer.

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
    meta_run_id = meta_run_id or f"scout-{str(uuid4())[:8]}"
    episodes = []
    context = []  # Sliding window of (config, outcome) pairs

    prompt_template = load_meta_prompt("scout_meta")

    for ep_num in range(n_episodes):
        # Build meta-prompt from context (last 20 episodes)
        meta_prompt = prompt_template.format(
            context=format_context(context[-20:]),
            action_space=action_space_description(),
        )

        response = meta_lm(meta_prompt)
        response_text = str(response)

        try:
            config_proposal = parse_config_from_response(response_text)
        except ValueError:
            config_proposal = {}  # Use defaults on parse failure
        reasoning = extract_reasoning(response_text)

        # Run ISO with proposed config
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
            playbook_state="",
        )
        episodes.append(episode)
        context.append((config_proposal, outcome))

        logger.info(f"Scout episode {ep_num}: reward={reward:.4f}")

    return episodes
