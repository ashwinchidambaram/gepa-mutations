"""ecological_succession: ESO (Ecological Succession Optimization) for prompt search.

Uses a progressive difficulty curriculum analogous to ecological succession:
- Phase 1 (Pioneer): Optimize on easy examples
- Phase 2 (Shrub): Optimize on easy + medium examples
- Phase 3 (Forest): Optimize on all examples (full complexity)
"""

from ecological_succession.difficulty import estimate_difficulty, partition_by_difficulty
from ecological_succession.runner import run_ecological_succession

__all__ = [
    "run_ecological_succession",
    "estimate_difficulty",
    "partition_by_difficulty",
]
