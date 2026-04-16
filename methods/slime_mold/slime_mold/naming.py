"""Method name derivation for the SMNO (Slime Mold Network Optimization) runner.

This module is intentionally kept free of heavy imports so it can be imported
in tests without requiring the full dependency stack.

See Phase 3 of the Inductive Strategy Discovery experiment plan.
"""

from __future__ import annotations


def _derive_method_name(
    strategy_mode: str,
    k: int | None,
    mutation_mode: str,
    refresh_mode: str,
) -> str:
    """Derive the method name for result storage from configuration flags.

    Args:
        strategy_mode: One of "personality", "prescribed8", "inductive".
        k: Number of skills for inductive discovery (3, 5, or None for adaptive).
        mutation_mode: One of "blind", "crosspollin".
        refresh_mode: One of "none", "expand", "replace".

    Returns:
        String method name used for run directory naming.

    Examples:
        >>> _derive_method_name("personality", None, "blind", "none")
        'slime_mold'
        >>> _derive_method_name("inductive", 5, "crosspollin", "expand")
        'slime_mold_inductive_k5_refresh_expand'
    """
    # Personality baseline (ignores k, mutation, refresh)
    if strategy_mode == "personality":
        return "slime_mold"

    # Prescribed-8 baseline (uses fixed 8 strategies, blind mutation, no refresh)
    if strategy_mode == "prescribed8":
        return "slime_mold_prescribed8"

    # Inductive: build up the name from parts
    if strategy_mode == "inductive":
        parts = ["slime_mold", "inductive"]
        parts.append("kadaptive" if k is None else f"k{k}")

        # Refresh takes precedence over crosspollin in naming
        if refresh_mode in ("expand", "replace"):
            parts.append(f"refresh_{refresh_mode}")
        elif mutation_mode == "crosspollin":
            parts.append("crosspollin")
        # else: blind mutation, no additional suffix

        return "_".join(parts)

    raise ValueError(f"Unknown strategy_mode: {strategy_mode!r}")
