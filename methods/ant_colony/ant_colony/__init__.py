"""ant_colony: ACPCO (Ant Colony Prompt Component Optimization).

Optimizes prompts by decomposing the search space into atomic components,
using pheromone-based selection to assemble high-quality prompt combinations.
"""

from ant_colony.components import Component, ComponentLibrary
from ant_colony.colony import AntColony
from ant_colony.runner import run_ant_colony

__all__ = [
    "run_ant_colony",
    "Component",
    "ComponentLibrary",
    "AntColony",
]
