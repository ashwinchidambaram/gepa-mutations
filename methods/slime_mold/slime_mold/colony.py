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
from dataclasses import dataclass, field
from typing import Any

from gepa_mutations.metrics.collector import MetricsCollector
from gepa_mutations.metrics.standalone_eval import evaluate_prompt


# ---------------------------------------------------------------------------
# Strategy dataclass
# ---------------------------------------------------------------------------


@dataclass
class Strategy:
    """A strategy/skill description used to generate specialized prompts."""
    name: str
    description: str
    failure_pattern: str = ""
    stress_examples: list[int] = field(default_factory=list)  # example indices
    source: str = "discovered"  # "discovered" | "prescribed"


# ---------------------------------------------------------------------------
# Prescribed strategies (8 universal problem-solving strategies)
# ---------------------------------------------------------------------------

PRESCRIBED_STRATEGIES: list[Strategy] = [
    Strategy(
        name="divide_and_conquer",
        description="Break the problem into smaller sub-problems, solve the easiest first, assemble results.",
        failure_pattern="Fails when sub-problems are tightly coupled or decomposition loses holistic context.",
        source="prescribed",
    ),
    Strategy(
        name="working_backward",
        description="Start from the desired end state and reason backward step by step to the current state.",
        failure_pattern="Fails when the end state is ambiguous or there are multiple valid endpoints.",
        source="prescribed",
    ),
    Strategy(
        name="analogy",
        description="Map the problem onto known solved patterns from other domains and adapt existing solutions.",
        failure_pattern="Fails when the problem is genuinely novel with no structural analogues.",
        source="prescribed",
    ),
    Strategy(
        name="abstraction",
        description="Strip away specifics to find core logic, solve the simplified version, then re-add detail.",
        failure_pattern="Fails when the messy details ARE the problem (edge cases, format constraints).",
        source="prescribed",
    ),
    Strategy(
        name="root_cause_analysis",
        description="Ask 'why?' iteratively (5 Whys) to find the fundamental issue before attempting a solution.",
        failure_pattern="Fails on tasks that are generative rather than diagnostic (no root cause to find).",
        source="prescribed",
    ),
    Strategy(
        name="trial_and_error",
        description="Systematically generate hypotheses, test against constraints, and refine.",
        failure_pattern="Fails on tasks with no feedback signal or where first-shot accuracy matters.",
        source="prescribed",
    ),
    Strategy(
        name="constraint_satisfaction",
        description="Enumerate all rules and limits upfront, then solve within the narrowed space.",
        failure_pattern="Fails when constraints interact in non-obvious ways or when creative solutions require relaxing constraints.",
        source="prescribed",
    ),
    Strategy(
        name="lateral_thinking",
        description="Challenge assumptions, reframe the problem, and explore unconventional approaches.",
        failure_pattern="Fails on straightforward tasks where the obvious approach is correct.",
        source="prescribed",
    ),
]


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
# Discovery helpers
# ---------------------------------------------------------------------------


def _parse_discovered_skills(text: str) -> list[Strategy]:
    """Parse a numbered-list discovery output into Strategy objects.

    Expected format:
        1. <name>: <description>. Failure: <failure pattern>
        2. <name>: <description>. Failure: <failure pattern>
        ...

    Falls back gracefully on malformed input — returns whatever was parseable.
    """
    # Match numbered items: "1. " or "1) " at start of line
    # Capture everything until next numbered item or end of string
    pattern = r'^\s*\d+[.)]\s+(.+?)(?=\n\s*\d+[.)]|\Z)'
    matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)

    skills: list[Strategy] = []
    for match in matches:
        content = match.strip()
        if not content:
            continue

        name = ""
        description = ""
        failure = ""

        # Extract failure section first (if present)
        fail_match = re.search(r'\bFailure:\s*(.+?)$', content, re.IGNORECASE | re.DOTALL)
        if fail_match:
            failure = fail_match.group(1).strip().rstrip(".")
            content = content[:fail_match.start()].rstrip(" .")

        # Now split name / description by first colon
        colon_idx = content.find(":")
        if colon_idx != -1:
            name = content[:colon_idx].strip()
            description = content[colon_idx + 1:].strip()
        else:
            # No colon — use first 3 words as name, rest as description
            words = content.split()
            name = " ".join(words[:3]) if words else content[:50]
            description = " ".join(words[3:]) if len(words) > 3 else ""

        if name:  # only keep if we got a name
            skills.append(Strategy(
                name=name,
                description=description,
                failure_pattern=failure,
                source="discovered",
            ))

    return skills


def discover_strategies(
    reflection_lm: Any,
    benchmark: str,
    task_description: str,
    examples: list,
    k: int | None = None,
    rng: Any = None,
    max_examples: int = 10,
) -> tuple[list[Strategy], str, bool]:
    """Phase 1: LLM examines examples, identifies distinct skills.

    Args:
        reflection_lm: Callable LM (already wrapped with TrackedLM).
        benchmark: Benchmark name (for logging).
        task_description: Brief task description.
        examples: List of benchmark examples.
        k: If int, ask for exactly k skills. If None, adaptive (LLM decides).
        rng: Optional random.Random for deterministic example sampling.
        max_examples: Max examples to show the LM.

    Returns:
        (strategies, raw_output, fallback_used)
        - strategies: list of Strategy objects
        - raw_output: raw LLM text (for logging/debugging)
        - fallback_used: True if we fell back to PRESCRIBED_STRATEGIES
    """
    # Sample examples
    if rng is not None and len(examples) > max_examples:
        sampled = rng.sample(examples, max_examples)
    else:
        sampled = examples[:max_examples]

    # Format examples
    formatted = []
    for i, ex in enumerate(sampled):
        input_str = getattr(ex, "input", None) or getattr(ex, "question", None) or str(ex)
        answer_str = getattr(ex, "answer", None) or getattr(ex, "output", "") or ""
        formatted.append(
            f"Example {i + 1}:\n"
            f"  Input: {str(input_str)[:300]}\n"
            f"  Expected answer: {str(answer_str)[:200]}"
        )
    examples_text = "\n\n".join(formatted)

    # Build prompt
    k_text = f"exactly {k}" if k is not None else "the number of skills you actually see (typically 3-10)"
    prompt = f"""Task: Identify the distinct skills or capabilities required to solve the following task.

Benchmark: {benchmark}
Task description: {task_description}

Examples from this benchmark:
{examples_text}

Identify {k_text} distinct skills or reasoning capabilities needed to solve these examples well.
For each skill, provide:
  - Name (2-3 words)
  - Brief description (1 sentence)
  - Failure pattern (what going wrong looks like)

Format as a numbered list:
1. <skill name>: <description>. Failure: <failure pattern>
2. <skill name>: <description>. Failure: <failure pattern>
...
"""

    # Attempt 1
    raw_output = ""
    try:
        raw_output = reflection_lm(prompt)
    except Exception:
        # Attempt 2 with clearer instruction
        try:
            retry_prompt = prompt + "\n\nIMPORTANT: Output ONLY the numbered list of skills, no preamble or explanation."
            raw_output = reflection_lm(retry_prompt)
        except Exception:
            # Fall back to prescribed
            return PRESCRIBED_STRATEGIES, raw_output, True

    # Parse
    skills = _parse_discovered_skills(raw_output)

    # Retry once if too few
    expected = k if k is not None else 3  # minimum threshold for adaptive
    if len(skills) < expected:
        retry_prompt = prompt + "\n\nIMPORTANT: Output ONLY the numbered list of skills, no preamble or explanation."
        try:
            raw_output = reflection_lm(retry_prompt)
            skills = _parse_discovered_skills(raw_output)
        except Exception:
            pass

    # Fallback if still inadequate
    if len(skills) < expected:
        return PRESCRIBED_STRATEGIES, raw_output, True

    return skills, raw_output, False


def generate_specialized_prompt(
    reflection_lm: Any,
    strategy: Strategy,
    task_description: str,
    seed_prompt: str,
    examples: list,
    n: int = 4,
    rng: Any = None,
    max_examples: int = 5,
) -> list[str]:
    """Phase 2: Generate n prompts specialized for one skill/strategy.

    Returns a list of prompt strings (typically length n, may be fewer on parse failure).
    """
    # Sample stress-test examples
    if rng is not None and len(examples) > max_examples:
        sampled = rng.sample(examples, max_examples)
    else:
        sampled = examples[:max_examples]

    formatted = []
    for i, ex in enumerate(sampled):
        input_str = getattr(ex, "input", None) or getattr(ex, "question", None) or str(ex)
        formatted.append(f"{i + 1}. {str(input_str)[:200]}")
    examples_text = "\n".join(formatted)

    prompt = f"""You are writing system prompts for an AI assistant on this task.

Task: {task_description}

Seed prompt:
{seed_prompt}

The prompts you write must EXCEL at: {strategy.name}
Description: {strategy.description}
What failure looks like: {strategy.failure_pattern}

Examples that stress-test this skill:
{examples_text}

Write {n} distinct system prompts that each:
1. Handles the general task competently.
2. Has specific, structural techniques for {strategy.name}.
3. Would score highest on examples requiring {strategy.name}, even if slightly weaker on other dimensions.

Format as a numbered list (output ONLY the prompts, no explanation):
1. <prompt>
2. <prompt>
...
"""

    try:
        response = reflection_lm(prompt)
    except Exception:
        return []

    # Reuse existing _parse_prompts helper
    return _parse_prompts(response)[:n]


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
