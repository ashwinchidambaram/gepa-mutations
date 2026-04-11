"""Synthesis step for Contrastive Synthesis Reflection (CSR).

Given a list of contrastive pairs (where one candidate outperformed another),
distills the underlying principle into a single actionable instruction that can
be injected into the reflective dataset as side_info — replacing the raw pair
snippets used by ContrastiveReflectionProposer.

Adds one extra LLM call per iteration (~500 tokens). If synthesis fails or
returns an empty string, falls back to an empty principle (no injection).
"""

from __future__ import annotations

from typing import Any

SYNTHESIS_PROMPT = """You are analyzing prompt optimization results. Given these contrastive examples where one prompt variant outperformed another on specific examples:

{pairs_text}

Distill the KEY PRINCIPLE that explains WHY the better prompt worked on these examples. State it as a single, actionable instruction (1-2 sentences) that could be used to improve similar prompts.

Principle:"""


def format_pairs(pairs: list[dict[str, Any]]) -> str:
    """Format contrastive pairs into a human-readable text block.

    Args:
        pairs: List of contrastive pair dicts from find_contrastive_candidates().
            Each dict has keys: current_score, contrastive_score, score_gap,
            candidate (dict with prompt fields).

    Returns:
        Formatted string describing each pair.
    """
    lines: list[str] = []
    for i, p in enumerate(pairs, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Current prompt score: {p['current_score']:.2f}")
        lines.append(f"  Better prompt score: {p['contrastive_score']:.2f}")
        lines.append(f"  Score gap: {p['score_gap']:.2f}")
        if "candidate" in p:
            snippet = str(p["candidate"].get("system_prompt", ""))[:200]
            lines.append(f"  Better prompt snippet: {snippet}...")
    return "\n".join(lines)


def synthesize(pairs: list[dict[str, Any]], reflection_lm: Any) -> str:
    """Distill contrastive pairs into an abstract improvement principle.

    Calls reflection_lm with a synthesis prompt that asks for a single
    actionable instruction explaining why the better candidates worked.

    Args:
        pairs: Contrastive pairs from find_contrastive_candidates(). Empty list
            returns empty string immediately (no LLM call).
        reflection_lm: The language model to call for synthesis. Expected to be
            callable with a prompt string and return a string response.

    Returns:
        Synthesized principle string, or empty string if pairs is empty or
        the LLM call fails.
    """
    if not pairs:
        return ""

    pairs_text = format_pairs(pairs)
    prompt = SYNTHESIS_PROMPT.format(pairs_text=pairs_text)

    try:
        response = reflection_lm(prompt)
        # Handle both string responses and list-of-completions responses
        if isinstance(response, list):
            response = response[0] if response else ""
        return str(response).strip()
    except Exception:
        return ""
