"""modular: PDMO (Prompt Decomposition and Modular Optimization) for prompt search.

Decomposes a seed prompt into 4 independent modules (task_framing,
reasoning_strategy, format_constraints, error_prevention), then optimizes
each module independently via sequential mini-GEPA runs, composes the results,
smooths the composition, and runs a final joint refinement pass.
"""

from modular.composer import compose_modules, smooth_composition
from modular.decomposer import decompose, parse_modules
from modular.runner import run_modular

__all__ = [
    "run_modular",
    "decompose",
    "parse_modules",
    "compose_modules",
    "smooth_composition",
]
