"""Core SMNO logic: diverse prompt generation and pruning rounds.

Slime Mold Network Optimization (SMNO) searches the prompt space by:
1. Generating a diverse initial pool of candidate prompts.
2. Progressively pruning the pool across 4 rounds, evaluating on increasingly
   larger subsets of training examples and keeping fewer survivors each round.
3. Surviving prompts are mutated between rounds using failure information
   gathered during evaluation.
"""

from __future__ import annotations

import re
import random
from typing import Any

from gepa_mutations.metrics.collector import MetricsCollector
from gepa_mutations.metrics.standalone_eval import evaluate_prompt


# ---------------------------------------------------------------------------
# Prompt generation helpers
# ---------------------------------------------------------------------------

_GENERATION_STRATEGIES = [
    (
        "analytical",
        "Use a systematic, analytical approach. Break down the problem into "
        "components, apply structured reasoning, and synthesize a clear answer.",
    ),
    (
        "creative",
        "Use creative, lateral thinking. Consider unconventional angles, "
        "analogies, or reframings to arrive at insight-driven answers.",
    ),
    (
        "minimal",
        "Be maximally concise. Provide only the essential reasoning needed "
        "to arrive at a correct, well-supported answer.",
    ),
    (
        "expert",
        "Respond as a domain expert with deep knowledge. Draw on best "
        "practices, precise terminology, and authoritative reasoning.",
    ),
]


def _parse_prompts(text: str) -> list[str]:
    """Extract individual prompts from a numbered or bulleted LLM response.

    Tries several common list formats:
    - "1. ...", "2. ..." numbered items
    - "**Prompt 1:** ..." or "Prompt 1: ..." labelled items
    - "- ..." or "* ..." bullet points
    Falls back to splitting on double-newlines if no structure is found.
    """
    # Try numbered list: "1." or "1)"
    numbered = re.findall(
        r"^\s*\d+[.)]\s+(.+?)(?=\n\s*\d+[.)]|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if len(numbered) >= 2:
        return [p.strip() for p in numbered if p.strip()]

    # Try labelled: "Prompt N:" or "**Prompt N:**"
    labelled = re.findall(
        r"\*{0,2}Prompt\s+\d+\*{0,2}:\*{0,2}\s*(.+?)(?=\*{0,2}Prompt\s+\d+|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if len(labelled) >= 2:
        return [p.strip() for p in labelled if p.strip()]

    # Try bullet list
    bullets = re.findall(r"^\s*[-*•]\s+(.+?)(?=\n\s*[-*•]|\Z)", text, re.MULTILINE | re.DOTALL)
    if len(bullets) >= 2:
        return [p.strip() for p in bullets if p.strip()]

    # Fallback: double-newline separated paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return paragraphs


def generate_diverse_prompts(
    reflection_lm: Any,
    seed_prompt: str,
    n: int,
    task_description: str,
    rng: random.Random,
) -> list[str]:
    """Generate n diverse prompt candidates via the reflection LM.

    Makes 4 LLM calls, each requesting ~ceil(n/4) prompts with a different
    creative strategy. Pads with variants of the seed if fewer than n are
    returned; trims to exactly n if more are returned.

    Args:
        reflection_lm: Callable LM (already wrapped with TrackedLM).
        seed_prompt: Starting prompt for the task.
        n: Number of prompt candidates to generate (exclusive of seed).
        task_description: Short description of the benchmark task.
        rng: Seeded random instance for reproducibility.

    Returns:
        List of exactly n prompt strings.
    """
    per_call = max(1, (n + 3) // 4)  # ceil(n/4)
    candidates: list[str] = []

    for strategy_name, strategy_hint in _GENERATION_STRATEGIES:
        prompt_text = (
            f"You are optimizing system prompts for an AI assistant.\n\n"
            f"Task description: {task_description}\n\n"
            f"Current seed prompt:\n{seed_prompt}\n\n"
            f"Strategy hint ({strategy_name}): {strategy_hint}\n\n"
            f"Generate exactly {per_call} distinct system prompts for this task. "
            f"Each prompt should be different in style or approach from the others. "
            f"Do NOT include explanations or commentary — output only the prompts.\n"
            f"Format your response as a numbered list:\n"
            f"1. <prompt text>\n"
            f"2. <prompt text>\n"
            f"..."
        )
        try:
            response = reflection_lm(prompt_text)
            parsed = _parse_prompts(response)
            candidates.extend(parsed)
        except Exception:
            # If the LLM call fails, we'll pad later
            pass

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        key = c[:200]  # use first 200 chars as dedup key
        if key not in seen:
            seen.add(key)
            unique.append(c)

    # Pad with minor variants of seed if we have fewer than n
    if len(unique) < n:
        variations = [
            f"{seed_prompt}\nBe thorough and precise.",
            f"Think step by step. {seed_prompt}",
            f"{seed_prompt}\nExplain your reasoning clearly.",
            f"You are an expert assistant. {seed_prompt}",
            f"{seed_prompt}\nProvide a well-structured answer.",
        ]
        for v in variations:
            if len(unique) >= n:
                break
            if v not in unique:
                unique.append(v)

    # Shuffle for diversity, then trim
    rng.shuffle(unique)
    return unique[:n]


# ---------------------------------------------------------------------------
# Mutation helper
# ---------------------------------------------------------------------------


def mutate_prompt(
    reflection_lm: Any,
    prompt: str,
    score: float,
    failures: list[dict[str, Any]],
) -> str:
    """Generate an improved version of a prompt given failure information.

    Args:
        reflection_lm: Callable LM (already wrapped with TrackedLM).
        prompt: Current prompt text to improve.
        score: Aggregate score achieved by this prompt on the evaluation batch.
        failures: List of {"input": str, "expected": str, "got": str} dicts
                  for examples where the prompt failed.

    Returns:
        Improved prompt string, or the original if the LLM fails.
    """
    failure_text = ""
    if failures:
        lines = []
        for f in failures[:5]:  # cap at 5 examples to keep prompt short
            lines.append(
                f"  Input: {str(f.get('input', ''))[:300]}\n"
                f"  Expected: {str(f.get('expected', ''))[:200]}\n"
                f"  Got: {str(f.get('got', ''))[:200]}"
            )
        failure_text = "\n\nFailed examples:\n" + "\n\n".join(lines)

    mutation_prompt = (
        f"You are improving a system prompt for an AI assistant.\n\n"
        f"Current prompt:\n{prompt}\n\n"
        f"This prompt achieved a score of {score:.3f} (higher is better, max 1.0)."
        f"{failure_text}\n\n"
        f"Improve this prompt to fix the failures while keeping what already works. "
        f"Output ONLY the improved prompt text — no explanations or commentary."
    )
    try:
        result = reflection_lm(mutation_prompt)
        result = result.strip()
        if result:
            return result
    except Exception:
        pass
    return prompt


# ---------------------------------------------------------------------------
# Pruning round
# ---------------------------------------------------------------------------


def run_pruning_round(
    adapter: Any,
    candidates: list[str],
    trainset: list,
    n_examples: int,
    collector: MetricsCollector,
    rng: random.Random,
) -> tuple[list[str], list[float], list[int]]:
    """Evaluate all candidates on a random subset of trainset examples.

    Args:
        adapter: GEPAAdapter for evaluation.
        candidates: List of prompt strings to evaluate.
        trainset: Full training dataset.
        n_examples: How many training examples to evaluate each prompt on.
        collector: MetricsCollector for rollout counting.
        rng: Seeded random for reproducible example selection.

    Returns:
        (sorted_candidates, sorted_scores, sorted_original_indices)
        Sorted descending by score (best first).
    """
    n_examples = min(n_examples, len(trainset))
    indices = rng.sample(range(len(trainset)), k=n_examples)

    scores: list[float] = []
    for prompt_text in candidates:
        candidate = {"system_prompt": prompt_text}
        score, _ = evaluate_prompt(adapter, trainset, candidate, collector, indices=indices)
        scores.append(score)

    # Sort descending by score
    ranked = sorted(zip(scores, candidates, range(len(candidates))), key=lambda x: -x[0])
    sorted_scores = [r[0] for r in ranked]
    sorted_candidates = [r[1] for r in ranked]
    sorted_indices = [r[2] for r in ranked]

    return sorted_candidates, sorted_scores, sorted_indices
