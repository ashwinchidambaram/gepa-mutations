"""iso: ISO (Iterative Search Optimization) for prompt search.

Uses progressive pruning of diverse candidate prompts, analogous to how
slime mold networks optimize nutrient transport paths over multiple rounds.
"""

from iso.colony import generate_diverse_prompts, mutate_prompt, run_pruning_round
from iso.runner import run_iso

__all__ = [
    "run_iso",
    "generate_diverse_prompts",
    "mutate_prompt",
    "run_pruning_round",
]
