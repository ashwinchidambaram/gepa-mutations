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
    technique: str = ""  # Discovery-only: the prompting technique this skill maps to
    stress_examples: list[int] = field(default_factory=list)  # example indices
    source: str = "discovered"  # "discovered" | "prescribed" | "refresh_discovered"


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


def _parse_failure_modes(text: str) -> dict[int, str]:
    """Parse the `### Failure modes` section into {1: desc, 2: desc, ...}."""
    section = re.search(
        r'###\s*Failure modes\s*\n(.+?)(?=\n###|\Z)',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not section:
        return {}

    modes: dict[int, str] = {}
    for line in section.group(1).splitlines():
        m = re.match(r'^\s*[-*•]?\s*\*{0,2}F(\d+)\*{0,2}\s*[:.\-—]\s*(.+?)$', line.strip())
        if m:
            modes[int(m.group(1))] = m.group(2).strip().rstrip(".")
    return modes


def _parse_new_skill_block(section: str, failure_modes: dict[int, str]) -> list[Strategy]:
    """Parse new-format skills:
        1. **<Name>** — Addresses: F<n(s)>. <What it does>. Technique: <technique>.
    """
    pattern = r'^\s*\d+[.)]\s+(.+?)(?=\n\s*\d+[.)]|\Z)'
    matches = re.findall(pattern, section, re.MULTILINE | re.DOTALL)

    skills: list[Strategy] = []
    for raw in matches:
        content = raw.strip()
        if not content:
            continue

        # Name: prefer **bold** at the start; otherwise take text up to first em dash / hyphen / colon
        name = ""
        rest = content
        bold = re.match(r'\*\*(.+?)\*\*\s*[—\-:]\s*(.+)', content, re.DOTALL)
        if bold:
            name = bold.group(1).strip()
            rest = bold.group(2).strip()
        else:
            sep = re.search(r'\s[—\-:]\s', content)
            if sep:
                name = content[:sep.start()].strip().strip("*")
                rest = content[sep.end():].strip()
            else:
                words = content.split()
                name = " ".join(words[:4]) if words else content[:50]
                rest = " ".join(words[4:]) if len(words) > 4 else ""

        # Addresses: F1, F2, F3
        addressed: list[int] = []
        addr = re.search(r'Addresses:\s*([^.]+?)(?:\.|$)', rest, re.IGNORECASE)
        if addr:
            for fn in re.findall(r'F\s*(\d+)', addr.group(1)):
                addressed.append(int(fn))
            rest = (rest[:addr.start()] + rest[addr.end():]).strip()

        # Technique: <name>
        technique = ""
        tech = re.search(r'Technique:\s*(.+?)(?:\.\s|\.$|$)', rest, re.IGNORECASE | re.DOTALL)
        if tech:
            technique = tech.group(1).strip().rstrip(".").strip("*")
            rest = (rest[:tech.start()] + rest[tech.end():]).strip().rstrip(".")

        description = rest.strip().lstrip("—-:.").strip().rstrip(".")

        # Build failure_pattern by looking up addressed F<n> modes
        failure = "; ".join(
            failure_modes[n] for n in addressed if n in failure_modes
        )

        if name:
            skills.append(Strategy(
                name=name,
                description=description,
                failure_pattern=failure,
                technique=technique,
                source="discovered",
            ))

    return skills


def _parse_old_skill_block(text: str) -> list[Strategy]:
    """Backward-compat parser for old format:
        1. <name>: <description>. Failure: <failure pattern>
    """
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

        fail_match = re.search(r'\bFailure:\s*(.+?)$', content, re.IGNORECASE | re.DOTALL)
        if fail_match:
            failure = fail_match.group(1).strip().rstrip(".")
            content = content[:fail_match.start()].rstrip(" .")

        colon_idx = content.find(":")
        if colon_idx != -1:
            name = content[:colon_idx].strip().strip("*")
            description = content[colon_idx + 1:].strip()
        else:
            words = content.split()
            name = " ".join(words[:3]) if words else content[:50]
            description = " ".join(words[3:]) if len(words) > 3 else ""

        if name:
            skills.append(Strategy(
                name=name,
                description=description,
                failure_pattern=failure,
                source="discovered",
            ))

    return skills


def _parse_discovered_skills(text: str) -> list[Strategy]:
    """Parse discovery output into Strategy objects.

    Primary format (new template):
        ### Failure modes
        - F1: ...
        - F2: ...
        ### Skills
        1. **<Name>** — Addresses: F<n(s)>. <What it does>. Technique: <technique>.

    Falls back to the legacy "1. <name>: <desc>. Failure: <pat>" format if no
    Skills section is found, so old smoke outputs and unit tests still parse.
    Returns whatever is parseable; never raises.
    """
    failure_modes = _parse_failure_modes(text)

    skills_section = re.search(
        r'###\s*Skills\s*\n(.+?)(?=\n###|\Z)',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if skills_section:
        return _parse_new_skill_block(skills_section.group(1), failure_modes)

    return _parse_old_skill_block(text)


def discover_strategies(
    reflection_lm: Any,
    benchmark: str,
    task_description: str,
    output_shape: str,
    examples: list,
    k: int | None = None,
    rng: Any = None,
    max_examples: int = 10,
) -> tuple[list[Strategy], str, bool]:
    """Phase 1: LLM examines examples, identifies distinct skills.

    Args:
        reflection_lm: Callable LM (already wrapped with TrackedLM).
        benchmark: Benchmark name (used only for logging — NOT shown to the LLM,
            to prevent memorization-based skill recall).
        task_description: 1-sentence task instruction (e.g. "Decide whether the
            claim is supported by the evidence."). Phrased without the benchmark
            name.
        output_shape: Short phrase describing the output (e.g.
            "a SUPPORTED or NOT_SUPPORTED label").
        examples: List of benchmark examples.
        k: If int, ask for exactly k skills. If None, adaptive (4-6).
        rng: Optional random.Random for deterministic example sampling.
        max_examples: Max examples to show the LM.

    Returns:
        (strategies, raw_output, fallback_used)
        - strategies: list of Strategy objects
        - raw_output: raw LLM text (for logging/debugging)
        - fallback_used: True if we fell back to PRESCRIBED_STRATEGIES
    """
    del benchmark  # intentionally unused — kept in signature for telemetry call sites

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

    k_text = f"exactly {k} skills" if k is not None else "between 4 and 6 skills"

    prompt = f"""# Task: Identify Specialized Skills for This Task

You are analyzing a task to find skills whose corresponding prompts will make *different kinds of mistakes*. The downstream system generates multiple prompt candidates — one per skill — and needs them to fail on *different* examples, not the same ones.

This means: skills that are merely reworded versions of each other are useless. Skills that sound impressive but share the same underlying technique are useless. What matters is that each skill, when turned into a prompt, stresses a structurally different part of the task.

---

## Task shape
This task takes an input and produces {output_shape}. The examples below show the exact patterns.

## Task instruction
{task_description}

## Examples
{examples_text}

---

## Step 1 — List 4–6 failure modes

Name specific ways a language model could get examples like these wrong. Each failure mode must name a *wrong behavior* and what *triggers* it, and cite example numbers.

Good: "Answers using the first paragraph that mentions the query entity, even when the target relation is in a later paragraph (Examples 2, 5)."

Bad: "Misreads the question." — names no behavior, no trigger, cites nothing.

## Step 2 — Propose {k_text}

For each skill, provide exactly these four fields:

- **Name** (2–5 words, distinctive — no generic labels)
- **Addresses**: F<number(s)> from your failure mode list
- **What it does** (one sentence, references patterns visible in the examples)
- **Technique** — pick ONE from the list below. *No two skills may use the same technique.* This constraint exists to force your skills to differ in kind, not just in wording.

## Technique list

Pick one per skill. Each skill must use a *different* technique.

- **Decomposition** — split input into labeled sub-parts before answering (e.g., "list each constraint as a bullet")
- **Scratchpad** — write a named intermediate form before the final answer (e.g., "(entity, source_paragraph) pairs")
- **Verification step** — re-check the draft against a specific criterion after writing it
- **Canonical output form** — enforce a fixed answer template
- **Ordering rule** — require sub-operations in a specific sequence (e.g., "identify bridge entity before searching for target fact")
- **Grounding citation** — cite the source span supporting each claim
- **Consistency discipline** — enforce a rule across repeated transformations (e.g., "same input entity → same placeholder, always")
- **Negative check** — explicitly rule out a named wrong-answer pattern
- **Constraint enumeration** — list every requirement from the input before drafting, check each off while writing

## Hard bans on skill names

Do not produce skills named or renamed as:
- Contextual Understanding / Careful Reading / Attention to Detail
- Logical Reasoning / Logical Deduction
- Pattern Recognition (unless you name *which* pattern)
- Fact Extraction / Information Extraction (unless you specify *what* and *from where*)
- Entity Recognition (unless you specify the entity's *role*)
- Conciseness / Brevity / Data Filtering

Also banned: any description using "ability to comprehend/understand/interpret" without a specific operator.

If you find yourself writing one of these, the skill is wrong — rewrite it from scratch, grounded in a specific failure mode.

---

## Output format

### Failure modes
- F1: ...
- F2: ...
(4 to 6 total)

### Skills
1. **<Name>** — Addresses: F<n>. <What it does>. Technique: <technique name>.
2. ...

Produce {k_text}. Every skill must use a *different* technique from the list above."""

    _retry_nudge = (
        "\n\nIMPORTANT: Follow the output format exactly — start with `### Failure modes`, "
        "then `### Skills` with a numbered list. No preamble or explanation."
    )

    # Attempt 1
    raw_output = ""
    try:
        raw_output = reflection_lm(prompt)
    except Exception:
        # Attempt 2 with clearer instruction
        try:
            raw_output = reflection_lm(prompt + _retry_nudge)
        except Exception:
            # Fall back to prescribed
            return PRESCRIBED_STRATEGIES, raw_output, True

    # Parse
    skills = _parse_discovered_skills(raw_output)

    # Retry once if too few
    expected = k if k is not None else 3  # minimum threshold for adaptive
    if len(skills) < expected:
        try:
            raw_output = reflection_lm(prompt + _retry_nudge)
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
    output_shape: str,
    hard_examples: list,
    existing_strategies: list,
    k_new: int = 2,
    max_examples: int = 10,
) -> tuple[list, str]:
    """Run a focused discovery pass on hard examples to find missed skills.

    Args:
        reflection_lm: Wrapped LM for reflection calls.
        benchmark: Benchmark name (used only for telemetry — NOT shown to LLM).
        task_description: 1-sentence task instruction.
        output_shape: Short phrase describing the output.
        hard_examples: List of benchmark examples that were difficult in R1.
        existing_strategies: Current strategies (to avoid rediscovery and to
            forbid reusing their techniques).
        k_new: Target number of NEW skills to identify.
        max_examples: Maximum number of hard examples to include in prompt.

    Returns:
        (new_strategies, raw_output)
    """
    del benchmark  # intentionally unused — kept for telemetry call sites

    # Format hard examples
    formatted = []
    for i, ex in enumerate(hard_examples[:max_examples]):
        input_str = getattr(ex, "input", None) or getattr(ex, "question", None) or str(ex)
        answer_str = getattr(ex, "answer", None) or getattr(ex, "output", "") or ""
        formatted.append(
            f"Example {i+1}:\n  Input: {str(input_str)[:300]}\n  Expected: {str(answer_str)[:200]}"
        )
    examples_text = "\n\n".join(formatted)

    # Format existing strategies (and the techniques they already use, so the
    # refresh pass is forbidden from reusing them).
    existing_lines = []
    used_techniques = []
    for s in existing_strategies:
        tech = getattr(s, "technique", "") or "(unspecified)"
        existing_lines.append(f"- **{s.name}** (Technique: {tech}): {s.description}")
        if getattr(s, "technique", ""):
            used_techniques.append(s.technique)
    existing_text = "\n".join(existing_lines) if existing_lines else "(none)"
    used_tech_text = ", ".join(sorted(set(used_techniques))) if used_techniques else "(none)"

    prompt = f"""# Task: Identify Additional Specialized Skills for Hard Examples

The downstream system already has a set of skills (listed below). Each was meant to address a different failure mode using a different prompting technique. Despite that, the examples below all failed for most candidates — meaning the existing skills do NOT cover the failure modes these examples expose.

Your job: identify {k_new} NEW skills that address failure modes the existing skills miss. Skills that are merely reworded versions of existing ones are useless. Skills that reuse a technique already used below are useless.

---

## Task shape
This task takes an input and produces {output_shape}. The hard examples below show the patterns.

## Task instruction
{task_description}

## Existing skills (DO NOT re-discover these)
{existing_text}

## Techniques already used (you may NOT pick any of these)
{used_tech_text}

## Hard examples (failed for most existing candidates)
{examples_text}

---

## Step 1 — Diagnose what the existing skills miss

List 2–3 failure modes visible in the hard examples that the existing skills do NOT address. Each must name a wrong behavior, what triggers it, and cite example numbers.

## Step 2 — Propose {k_new} new skills

Each skill must:
- Address a failure mode from Step 1 (NOT one already covered by an existing skill)
- Use a technique NOT in the "Techniques already used" list

Pick from the technique list below. Each new skill must use a *different* technique from each other AND from the techniques already used.

## Technique list

- **Decomposition**, **Scratchpad**, **Verification step**, **Canonical output form**, **Ordering rule**, **Grounding citation**, **Consistency discipline**, **Negative check**, **Constraint enumeration**

## Hard bans on skill names

Banned: Contextual Understanding / Careful Reading / Attention to Detail / Logical Reasoning / Logical Deduction / Pattern Recognition (without naming *which*) / Fact Extraction / Information Extraction (unspecified) / Entity Recognition (unspecified) / Conciseness / Brevity / Data Filtering / any "ability to comprehend/understand/interpret" phrasing.

---

## Output format

### Failure modes
- F1: ...
- F2: ...

### Skills
1. **<Name>** — Addresses: F<n>. <What it does>. Technique: <technique name>.
2. ...

Produce {k_new} skills. Each must use a different technique, and none may reuse a technique listed under "Techniques already used"."""

    raw_output = ""
    try:
        raw_output = reflection_lm(prompt)
    except Exception:
        return [], raw_output

    # Reuse parser
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
