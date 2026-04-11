"""Prompt decomposer for PDMO (Prompt Decomposition and Modular Optimization).

Decomposes a system prompt into 4 independent modules via the reflection LM.
"""

from __future__ import annotations

import re

MODULE_NAMES = [
    "task_framing",
    "reasoning_strategy",
    "format_constraints",
    "error_prevention",
]

DECOMPOSE_PROMPT = """You are a prompt engineering expert. Decompose the following system prompt into 4 independent modules. Each module should be self-contained and address one aspect of the task.

Original prompt: {seed_prompt}

Task examples:
{examples}

Return EXACTLY 4 modules in this format:

[MODULE:task_framing]
(The role definition and task description)
[/MODULE:task_framing]

[MODULE:reasoning_strategy]
(The step-by-step reasoning approach)
[/MODULE:reasoning_strategy]

[MODULE:format_constraints]
(Output format requirements and constraints)
[/MODULE:format_constraints]

[MODULE:error_prevention]
(Edge cases, common mistakes, and error prevention)
[/MODULE:error_prevention]"""


def decompose(
    seed_prompt: str,
    examples: list,
    reflection_lm: object,
) -> dict[str, str]:
    """Decompose a prompt into 4 independent modules via the reflection LM.

    Args:
        seed_prompt: The system prompt to decompose.
        examples: Training examples (uses first 5 for context).
        reflection_lm: LM callable (str | list -> str).

    Returns:
        Dict mapping module_name -> module_text for each of the 4 modules.
    """
    examples_text = "\n".join(
        f"- {str(getattr(ex, 'input', ex))[:200]}" for ex in examples[:5]
    )
    prompt = DECOMPOSE_PROMPT.format(
        seed_prompt=seed_prompt,
        examples=examples_text,
    )
    response = reflection_lm(prompt)
    return parse_modules(response)


def parse_modules(text: str) -> dict[str, str]:
    """Parse [MODULE:name]...[/MODULE:name] blocks from text.

    If no modules are found (e.g., the LM didn't follow the format), falls
    back to putting the entire text in task_framing.

    Args:
        text: Raw LM response expected to contain module blocks.

    Returns:
        Dict mapping module_name -> module_text.
    """
    modules: dict[str, str] = {}
    for name in MODULE_NAMES:
        pattern = rf"\[MODULE:{name}\](.*?)\[/MODULE:{name}\]"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            modules[name] = match.group(1).strip()
        else:
            modules[name] = ""

    # Fallback: if no modules found, put everything in task_framing
    if all(v == "" for v in modules.values()):
        modules["task_framing"] = text.strip()

    return modules
