"""Prompt formatting helpers for ISO optimizer.

All formatting functions produce concise, token-efficient representations
for inclusion in reflection LM prompts.
"""

from __future__ import annotations

import json
from typing import Any

_MAX_FIELD_LEN = 500  # Truncate long fields to save tokens


def _truncate(s: str, max_len: int = _MAX_FIELD_LEN) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."


def format_failures(failures: list) -> str:
    """Pretty-print failure traces for inclusion in prompts.

    Each failure is a ModuleTrace-like object with: example_id, score, feedback, module_outputs
    """
    if not failures:
        return "(no failures)"

    lines = []
    for i, f in enumerate(failures):
        example_id = getattr(f, 'example_id', f.get('example_id', '?') if isinstance(f, dict) else '?')
        score = getattr(f, 'score', f.get('score', 0.0) if isinstance(f, dict) else 0.0)
        feedback = getattr(f, 'feedback', f.get('feedback', '') if isinstance(f, dict) else '')
        module_outputs = getattr(f, 'module_outputs', f.get('module_outputs', {}) if isinstance(f, dict) else {})

        lines.append(f"--- Failure {i + 1} ---")
        lines.append(f"Example ID: {example_id}")
        lines.append(f"Score: {score:.3f}")
        lines.append(f"Feedback: {_truncate(str(feedback))}")
        if module_outputs:
            lines.append("Module outputs:")
            for mod_name, output in module_outputs.items():
                lines.append(f"  {mod_name}: {_truncate(str(output))}")
        lines.append("")

    return "\n".join(lines)


def format_traces(traces: list) -> str:
    """Format traces for reflection prompts. Same format as format_failures."""
    return format_failures(traces)


def format_context(episodes: list) -> str:
    """Format meta-optimizer episodes for in-context learning.

    Each episode has: episode_num, proposed_config, reward, meta_llm_reasoning, episode_outcome
    """
    if not episodes:
        return "(no prior episodes)"

    lines = []
    for ep in episodes:
        if hasattr(ep, 'episode_num'):
            num = ep.episode_num
            config = ep.proposed_config
            reward = ep.reward
            reasoning = _truncate(ep.meta_llm_reasoning, 200)
            outcome = ep.episode_outcome
        elif isinstance(ep, tuple) and len(ep) == 2:
            # (config, outcome) pair format
            config, outcome = ep
            num = "?"
            reward = outcome.get("final_score", 0)
            reasoning = ""
        else:
            continue

        lines.append(f"Episode {num}:")
        lines.append(f"  Config: {json.dumps(config, default=str) if isinstance(config, dict) else str(config)}")
        lines.append(f"  Reward: {reward}")
        if reasoning:
            lines.append(f"  Reasoning: {reasoning}")
        if isinstance(outcome, dict):
            score = outcome.get("final_score", "?")
            rollouts = outcome.get("rollouts_consumed", "?")
            lines.append(f"  Outcome: score={score}, rollouts={rollouts}")
        lines.append("")

    return "\n".join(lines)


def format_frontier(frontier: list) -> str:
    """Format Pareto frontier for Atlas meta-optimizer.

    Each point has: reward (list[float]), outcome (dict)
    """
    if not frontier:
        return "(empty frontier)"

    lines = ["Score | Rollouts | Tokens"]
    lines.append("------|----------|-------")
    for point in frontier:
        reward = point.get("reward", [0, 0, 0])
        outcome = point.get("outcome", {})
        score = reward[0] if len(reward) > 0 else 0
        rollouts = -reward[1] if len(reward) > 1 else 0  # negated in reward
        tokens = -reward[2] if len(reward) > 2 else 0
        lines.append(f"{score:.3f} | {int(rollouts)} | {int(tokens)}")

    return "\n".join(lines)


def summarize_frontier(frontier: list) -> str:
    """Natural-language summary of Pareto frontier."""
    if not frontier:
        return "Frontier is empty — no episodes completed yet."

    n = len(frontier)
    scores = [p.get("reward", [0])[0] for p in frontier]
    best_score = max(scores) if scores else 0

    # Find the cheapest point
    rollouts = [-p.get("reward", [0, 0])[1] for p in frontier if len(p.get("reward", [])) > 1]
    min_rollouts = min(rollouts) if rollouts else 0

    return (
        f"{n} points on frontier. "
        f"Best score: {best_score:.3f}. "
        f"Cheapest: {int(min_rollouts)} rollouts."
    )


def sample_prompts(pool: list, n: int) -> list[str]:
    """Return prompt text from up to n random candidates in the pool."""
    import random as _random
    sample = _random.sample(pool, min(n, len(pool))) if pool else []
    result = []
    for candidate in sample:
        prompts = getattr(candidate, 'prompts_by_module', {})
        if isinstance(prompts, dict):
            for mod_name, prompt_text in prompts.items():
                result.append(f"[{mod_name}] {_truncate(str(prompt_text), 300)}")
    return result


def action_space_description() -> str:
    """Formatted MetaAction field descriptions with ranges for meta-optimizer prompts."""
    return """Action space (choose values for each parameter):
- pool_size_seed: int [3-8] — Number of initial skill clusters (more = more diverse but slower)
- mutations_per_seed: int [1-3] — Mutations per skill candidate (more = larger initial pool)
- minibatch_count: int [3-7] — Number of evaluation minibatches per round (more = more reliable scores but more rollouts)
- minibatch_size: int [2-8] — Examples per minibatch (more = more reliable but more rollouts)
- prune_aggressiveness: float [0.1-0.6] — Fraction of pool to prune each round (higher = faster convergence but less diversity)
- max_rounds: int [3-20] — Maximum optimization rounds (more = more refinement but more rollouts)

Output your proposal as JSON:
```json
{
  "pool_size_seed": <int>,
  "mutations_per_seed": <int>,
  "minibatch_count": <int>,
  "minibatch_size": <int>,
  "prune_aggressiveness": <float>,
  "max_rounds": <int>
}
```"""
