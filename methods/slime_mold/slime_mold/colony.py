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
    prompt = f"""# Task: Identify the Distinctive Skills of a Specific Benchmark

You are a senior evaluator analyzing a language benchmark. Your job: produce a skill decomposition that captures what makes THIS benchmark's examples hard, in a way that a prompt engineer could use to build specialized prompts. Generic skills are useless — they waste the analysis.

---

## Benchmark
{benchmark}

## Task description
{task_description}

## Examples
{examples_text}

---

## Process (follow each step in your response — think step by step)

### Step 1 — Pattern observation (private, do NOT output)
Before proposing any skill, silently observe:
- What is the *structural shape* of inputs? (paragraphs, evidence lists, questions, PII-bearing text, constraints, math problems, …)
- What is the *answer shape*? (short span, yes/no label, redacted text, final number, free text, …)
- What *specific operations* do multiple examples require? (e.g., "follow a bridge entity from paragraph A to paragraph B", "count constraints", "substitute names with typed placeholders")
- What *failure modes* does the answer shape invite? (e.g., answering from the wrong paragraph, producing the right label with wrong justification, leaking PII)

### Step 2 — Generate {k_text} candidate skills grounded in patterns from Step 1
Each skill MUST satisfy ALL THREE criteria:

**(A) Specific to this benchmark**
The skill name and description must reference the structural/semantic pattern of THESE examples. Do NOT use terms that could apply to any reading-comprehension task. If you could imagine the exact same skill name fitting HotpotQA, HOVER, PUPA, IFBench, LiveBench, and AIME equally well, it's generic — throw it out.

**(B) Targets a different failure mode than your other skills**
Two skills should never both be about "understanding the text" or "reading carefully". If two of your skills could both be replaced by the phrase "pay attention", collapse them into one specific skill and replace the other with something new.

**(C) Implementable as a prompt technique**
Imagine you are writing a prompt that EXCELS at this skill. What would that prompt say or do differently from a generic baseline? If you cannot name a concrete prompting technique (a verification step, a decomposition rule, a canonical output form, a scratchpad structure), the skill is too vague.

### Step 3 — Self-critique (for each candidate, silently ask)
- "If I rename this skill, does its meaning change? If I just renamed 'Contextual Understanding' to 'Contextual Reading', I've learned nothing — that's generic."
- "Can I point to a specific example number above that STRESSES this skill?"
- "What concrete prompting technique would this skill inspire?"
- If any answer is weak, discard or replace the skill.

### Step 4 — Output (this is what you actually write)

Output ONLY a numbered list. Each item has three parts separated clearly:

```
1. <specific skill name — 2 to 5 words, distinctive to this benchmark>: <one sentence describing the skill, referencing a pattern visible in the examples>. Failure: <one sentence describing what a prompt that LACKS this skill would produce incorrectly on a specific example>.
2. ...
```

---

## Hard bans — do NOT output any skill named or described as:
- "Contextual Understanding" / "Context Comprehension" / "Contextual Reading"
- "Attention to Detail" / "Careful Reading"
- "Fact Extraction" / "Information Extraction" (unless you specify WHAT KIND of fact/information from where)
- "Entity Recognition" (unless you specify the role entities play in THIS task)
- "Logical Reasoning" / "Logical Deduction"
- "Conciseness" / "Brevity"
- "Pattern Recognition" (too vague — specify WHICH patterns)
- "Data Filtering" (too vague)
- Any skill whose description contains phrases like "ability to comprehend", "ability to understand", "ability to interpret" without a domain-specific operator

If your draft includes any of these, rewrite that skill from scratch, grounded in a specific pattern you observed in the examples.

## Anti-examples (what a lazy answer looks like — these are BAD)
```
1. Contextual Understanding: Comprehending the context. Failure: Misinterpreting the context.
2. Attention to Detail: Reading carefully. Failure: Missing details.
```
☝️ Those are USELESS. If you produce output like that, you have failed the task.

## Good-example shapes (what a GOOD answer looks like — for illustration only; do not copy these specific skills unless they genuinely apply to the examples above):
```
[Multi-hop QA example]
1. Bridge-entity chaining: the input contains multiple paragraphs; the answer requires picking an entity from paragraph A, looking up a fact about it in paragraph B, and returning that fact. Failure: answering from paragraph A alone and missing the bridge.
2. Distractor paragraph suppression: several paragraphs mention the query entity but only one contains the answer. Failure: answering from the first-mentioned paragraph rather than the one with the target relation.
```
```
[Fact verification example]
1. Evidence-polarity aggregation: each retrieved evidence item is labeled supporting / contradicting / neutral; the verdict is the majority-signed product. Failure: treating any mention as supporting, ignoring contradicting evidence.
2. Single-hop vs multi-hop claim decomposition: some claims require verifying 1 fact, others require conjoining 2–3 facts. Failure: declaring supported based on partial verification.
```

Now do the work. Produce {k_text} skills specific to THIS benchmark's examples above."""

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
# Cross-pollination helpers
# ---------------------------------------------------------------------------


def collect_hard_examples(
    failure_matrix: dict[int, set[int]],
    n_candidates: int,
    threshold: float = 0.7,
) -> list[int]:
    """Identify examples that a high fraction of candidates failed on.

    Args:
        failure_matrix: {candidate_idx: set_of_failed_example_ids} from build_failure_matrix()
        n_candidates: Total number of candidates evaluated in the round.
        threshold: Fraction (0.0-1.0). Examples failed by >= this fraction of
                   candidates are "hard". Default 0.7.

    Returns:
        Sorted list of example ids that are "hard" (universally difficult).
    """
    if n_candidates == 0 or not failure_matrix:
        return []

    # Count how many candidates failed on each example
    failure_count: dict[int, int] = {}
    for failed_set in failure_matrix.values():
        for ex_id in failed_set:
            failure_count[ex_id] = failure_count.get(ex_id, 0) + 1

    # An example is "hard" if failure_count / n_candidates >= threshold
    min_failures = threshold * n_candidates
    hard = [ex_id for ex_id, count in failure_count.items() if count >= min_failures]
    return sorted(hard)


def discover_refresh_strategies(
    reflection_lm: Any,
    benchmark: str,
    task_description: str,
    hard_examples: list,
    existing_strategies: list,
    k_new: int = 2,
    max_examples: int = 10,
) -> tuple[list, str]:
    """Run a focused discovery pass on hard examples to find missed skills.

    Args:
        reflection_lm: Wrapped LM for reflection calls.
        benchmark: Benchmark name.
        task_description: Brief task description.
        hard_examples: List of benchmark examples that were difficult in R1.
        existing_strategies: Current strategies (to avoid rediscovery).
        k_new: Target number of NEW skills to identify.
        max_examples: Maximum number of hard examples to include in prompt.

    Returns:
        (new_strategies, raw_output)
    """
    # Format hard examples
    formatted = []
    for i, ex in enumerate(hard_examples[:max_examples]):
        input_str = getattr(ex, "input", None) or getattr(ex, "question", None) or str(ex)
        answer_str = getattr(ex, "answer", None) or getattr(ex, "output", "") or ""
        formatted.append(
            f"Example {i+1}:\n  Input: {str(input_str)[:300]}\n  Expected: {str(answer_str)[:200]}"
        )
    examples_text = "\n\n".join(formatted)

    # Format existing strategies
    existing_text = "\n".join(
        f"- {s.name}: {s.description}" for s in existing_strategies
    )

    prompt = f"""Task: Identify NEW skills required to solve difficult examples.

Benchmark: {benchmark}
Task description: {task_description}

The following examples were difficult — most of our current strategies failed on them:
{examples_text}

Our current strategies:
{existing_text}

Identify {k_new} NEW skills or reasoning capabilities that would help on these difficult
examples — skills that are DISTINCT from the existing strategies listed above.

For each new skill, provide:
  - Name (2-3 words)
  - Brief description (1 sentence)
  - Failure pattern (what going wrong looks like)

Format as a numbered list:
1. <skill name>: <description>. Failure: <failure pattern>
2. ...
"""

    raw_output = ""
    try:
        raw_output = reflection_lm(prompt)
    except Exception:
        return [], raw_output

    # Reuse existing parser
    skills = _parse_discovered_skills(raw_output)
    # Mark as discovered (from refresh pass)
    for s in skills[:k_new]:
        s.source = "refresh_discovered"
    return skills[:k_new], raw_output


def build_failure_matrix(
    per_example_scores: dict[int, dict[int, float]],
    threshold: float = 0.5,
) -> dict[int, set[int]]:
    """Build a matrix of which candidates failed on which examples.

    Args:
        per_example_scores: {candidate_idx: {example_id: score}}
        threshold: scores >= threshold count as passes, < as failures

    Returns:
        {candidate_idx: set_of_failed_example_ids}
    """
    return {
        cand_idx: {ex_id for ex_id, score in ex_scores.items() if score < threshold}
        for cand_idx, ex_scores in per_example_scores.items()
    }


def find_donor(
    survivor_idx: int,
    survivor_strategy: str,
    failure_matrix: dict[int, set[int]],
    per_example_scores: dict[int, dict[int, float]],
    strategies: dict[int, str],
    threshold: float = 0.5,
) -> dict | None:
    """Find the best donor candidate for the survivor's failures.

    Tiebreaker: (1) cross-strategy first, (2) higher score on survivor failures, (3) lower idx.
    Returns None if no donor covers any of the survivor's failures.

    Returns a dict with all fields for cross_pollination_events logging:
        donor_candidate_idx, donor_strategy, shared_failures_covered,
        donor_score_on_failures, cross_strategy, no_donor_found
    """
    survivor_failures = failure_matrix.get(survivor_idx, set())
    if not survivor_failures:
        return None

    best: tuple | None = None  # (cross_rank, coverage, mean_score, neg_cand_idx, cand_idx)

    for cand_idx, cand_strategy in strategies.items():
        if cand_idx == survivor_idx:
            continue

        cand_scores = per_example_scores.get(cand_idx, {})
        covered = {
            ex_id for ex_id in survivor_failures
            if cand_scores.get(ex_id, 0.0) >= threshold
        }
        coverage = len(covered)
        if coverage == 0:
            continue

        scores_on_failures = [
            cand_scores.get(ex_id, 0.0) for ex_id in survivor_failures
        ]
        mean_score = sum(scores_on_failures) / len(scores_on_failures) if scores_on_failures else 0.0

        is_cross = cand_strategy != survivor_strategy
        cross_rank = 1 if is_cross else 0

        # Tiebreaker: higher is better for cross_rank, coverage, mean_score; lower cand_idx wins
        key = (cross_rank, coverage, mean_score, -cand_idx)
        if best is None or key > best[:4]:
            best = (*key, cand_idx)

    if best is None:
        return None

    cross_rank, coverage, mean_score, _, cand_idx = best
    return {
        "donor_candidate_idx": cand_idx,
        "donor_strategy": strategies.get(cand_idx),
        "shared_failures_covered": coverage,
        "donor_score_on_failures": mean_score,
        "cross_strategy": cross_rank == 1,
        "no_donor_found": False,
    }


def mutate_prompt_with_context(
    reflection_lm: Any,
    prompt: str,
    score: float,
    failures: list[dict[str, Any]],
    survivor_strategy: str | None = None,
    donor_strategy: str | None = None,
    donor_prompt: str | None = None,
) -> str:
    """Generate an improved prompt with strategy context + optional donor technique.

    If donor_strategy and donor_prompt are provided, includes them as cross-pollination hint.
    Otherwise falls back to strategy-aware blind mutation.
    """
    failure_text = ""
    if failures:
        lines = []
        for f in failures[:5]:
            lines.append(
                f"  Input: {str(f.get('input', ''))[:300]}\n"
                f"  Expected: {str(f.get('expected', ''))[:200]}\n"
                f"  Got: {str(f.get('got', ''))[:200]}"
            )
        failure_text = "\n\nFailed examples:\n" + "\n\n".join(lines)

    strategy_ctx = ""
    if survivor_strategy:
        strategy_ctx = f"\n\nThis prompt was designed using the {survivor_strategy} approach."

    donor_ctx = ""
    if donor_strategy and donor_prompt:
        donor_truncated = donor_prompt[:500] if len(donor_prompt) > 500 else donor_prompt
        donor_ctx = (
            f"\n\nAnother prompt using the {donor_strategy} approach succeeded on "
            f"some of the examples this prompt failed on. Here is that prompt's approach:\n"
            f"---\n{donor_truncated}\n---\n\n"
            f"Consider incorporating insights from the {donor_strategy} approach "
            f"while preserving the strengths of this prompt's {survivor_strategy or 'original'} approach."
        )

    mutation_prompt = (
        f"You are improving a system prompt for an AI assistant.\n\n"
        f"Current prompt:\n{prompt}\n\n"
        f"This prompt achieved a score of {score:.3f} (higher is better, max 1.0)."
        f"{strategy_ctx}"
        f"{failure_text}"
        f"{donor_ctx}\n\n"
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
) -> tuple[list[str], list[float], list[int], dict[int, dict[int, float]]]:
    """Evaluate all candidates on a random subset of trainset examples.

    Args:
        adapter: GEPAAdapter for evaluation.
        candidates: List of prompt strings to evaluate.
        trainset: Full training dataset.
        n_examples: How many training examples to evaluate each prompt on.
        collector: MetricsCollector for rollout counting.
        rng: Seeded random for reproducible example selection.

    Returns:
        (sorted_candidates, sorted_scores, sorted_original_indices, per_example_scores)
        Sorted descending by score (best first).
        per_example_scores: {sorted_position: {example_id: score}}
    """
    n_examples = min(n_examples, len(trainset))
    indices = rng.sample(range(len(trainset)), k=n_examples)

    scores: list[float] = []
    per_example_matrix: dict[int, dict[int, float]] = {}
    for orig_idx, prompt_text in enumerate(candidates):
        candidate = {"system_prompt": prompt_text}
        score, per_ex_scores = evaluate_prompt(adapter, trainset, candidate, collector, indices=indices)
        scores.append(score)
        per_example_matrix[orig_idx] = {
            ex_id: ex_score for ex_id, ex_score in zip(indices, per_ex_scores)
        }

    # Sort descending by score
    ranked = sorted(zip(scores, candidates, range(len(candidates))), key=lambda x: -x[0])
    sorted_scores = [r[0] for r in ranked]
    sorted_candidates = [r[1] for r in ranked]
    sorted_orig_indices = [r[2] for r in ranked]

    # Remap per_example_matrix to use sorted positions as keys
    sorted_matrix = {
        sorted_pos: per_example_matrix[orig_idx]
        for sorted_pos, orig_idx in enumerate(sorted_orig_indices)
    }

    return sorted_candidates, sorted_scores, sorted_orig_indices, sorted_matrix
