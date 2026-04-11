"""slime_mold: SMNO (Slime Mold Network Optimization) for prompt search.

Uses progressive pruning of diverse candidate prompts, analogous to how
slime mold networks optimize nutrient transport paths over multiple rounds.
"""

from slime_mold.colony import generate_diverse_prompts, mutate_prompt, run_pruning_round
from slime_mold.runner import run_slime_mold

__all__ = [
    "run_slime_mold",
    "generate_diverse_prompts",
    "mutate_prompt",
    "run_pruning_round",
]
