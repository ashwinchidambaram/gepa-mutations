"""Prompt composer for PDMO (Prompt Decomposition and Modular Optimization).

Provides utilities to:
  - compose_modules(): join module texts into a single system_prompt
  - smooth_composition(): ask the reflection LM to smooth transitions
"""

from __future__ import annotations

from modular.decomposer import MODULE_NAMES

SMOOTH_PROMPT = """You are a prompt engineering expert. The following system prompt was created by composing 4 independently optimized modules. Smooth the transitions, remove redundancy, and ensure consistency. Do NOT change the core instructions — only improve readability and flow.

Composed prompt:
{prompt}

Smoothed prompt:"""


def compose_modules(modules: dict[str, str]) -> str:
    """Compose module dict into a single system_prompt string with paragraph breaks.

    Non-empty modules are joined in canonical order (task_framing,
    reasoning_strategy, format_constraints, error_prevention).

    Args:
        modules: Dict mapping module_name -> module_text.

    Returns:
        Single system prompt string with modules separated by blank lines.
    """
    parts = []
    for name in MODULE_NAMES:
        if name in modules and modules[name].strip():
            parts.append(modules[name].strip())
    return "\n\n".join(parts)


def smooth_composition(composed: str, reflection_lm: object) -> str:
    """Ask the reflection LM to smooth a composed prompt.

    Improves readability and flow without changing core instructions.

    Args:
        composed: The raw composed system prompt.
        reflection_lm: LM callable (str | list -> str).

    Returns:
        Smoothed system prompt string (falls back to composed if LM fails).
    """
    prompt = SMOOTH_PROMPT.format(prompt=composed)
    try:
        result = reflection_lm(prompt)
        if result and result.strip():
            return result.strip()
    except Exception:
        pass
    return composed
