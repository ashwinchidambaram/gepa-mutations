"""Component library for ACPCO.

Manages the pool of atomic prompt components and their pheromone levels.
Each component is a short (1-2 sentence) instruction from one of five
predefined categories. Components are selected by "ants" using pheromone-
weighted sampling to compose candidate prompts.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

# The five component categories used by ACPCO
CATEGORIES = [
    "task_framing",
    "reasoning_strategy",
    "format_instructions",
    "error_prevention",
    "domain_specific",
]

# Category display order for composed prompts
_CATEGORY_ORDER = {cat: i for i, cat in enumerate(CATEGORIES)}


@dataclass
class Component:
    """A single atomic prompt instruction with its pheromone level.

    Attributes:
        text: The instruction text (1-2 sentences).
        category: One of the five ACPCO categories.
        pheromone: Current pheromone level (starts at 1.0, updated over rounds).
    """

    text: str
    category: str
    pheromone: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "category": self.category, "pheromone": self.pheromone}


def _parse_component_list(text: str) -> list[str]:
    """Extract individual component texts from an LLM response.

    Handles numbered lists, bullet lists, and paragraph-separated formats.
    """
    # Numbered list: "1.", "1)"
    numbered = re.findall(
        r"^\s*\d+[.)]\s+(.+?)(?=\n\s*\d+[.)]|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if len(numbered) >= 2:
        return [c.strip() for c in numbered if c.strip()]

    # Bullet list
    bullets = re.findall(
        r"^\s*[-*•]\s+(.+?)(?=\n\s*[-*•]|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if len(bullets) >= 2:
        return [c.strip() for c in bullets if c.strip()]

    # Double-newline separated
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return paragraphs


class ComponentLibrary:
    """Manages all atomic prompt components and their pheromone levels.

    Usage::

        library = ComponentLibrary()
        library.generate(reflection_lm, seed_prompt, CATEGORIES)
        selected = library.sample_components(rng, n=10)
        prompt_text = library.compose_prompt(selected)
        library.update_pheromone(selected, score=0.7, median_score=0.6,
                                  evaporation_rate=0.1)

    """

    def __init__(self) -> None:
        self._components: list[Component] = []

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        reflection_lm: Any,
        seed_prompt: str,
        categories: list[str] | None = None,
        n_per_category: int = 10,
    ) -> list[Component]:
        """Generate prompt components via the reflection LM.

        Makes one LLM call per category, requesting n_per_category
        atomic instructions each. Stores all resulting components in the
        library and returns them.

        Args:
            reflection_lm: Callable LM (already wrapped with TrackedLM).
            seed_prompt: Seed prompt describing the task.
            categories: Category names to generate. Defaults to CATEGORIES.
            n_per_category: Number of components to generate per category.

        Returns:
            All components now stored in the library.
        """
        categories = categories or CATEGORIES
        self._components = []

        for category in categories:
            prompt_text = (
                f"You are designing modular system prompt instructions for an AI assistant.\n\n"
                f"Task context (current seed prompt):\n{seed_prompt}\n\n"
                f"Category: {category.replace('_', ' ').title()}\n\n"
                f"Generate exactly {n_per_category} short, atomic prompt instructions "
                f"(1-2 sentences each) for the category '{category}' for this task. "
                f"Each instruction should be:\n"
                f"  - Self-contained and independent (can stand alone)\n"
                f"  - Combinable with instructions from other categories\n"
                f"  - Specific and actionable\n\n"
                f"Output ONLY the instructions as a numbered list:\n"
                f"1. <instruction>\n"
                f"2. <instruction>\n"
                f"..."
            )
            try:
                response = reflection_lm(prompt_text)
                texts = _parse_component_list(response)
            except Exception:
                texts = []

            for text in texts[:n_per_category]:
                if text.strip():
                    self._components.append(Component(text=text.strip(), category=category))

            # If fewer than requested, pad with generic instructions
            existing_count = sum(1 for c in self._components if c.category == category)
            while existing_count < n_per_category:
                self._components.append(
                    Component(
                        text=f"Apply careful {category.replace('_', ' ')} to this task.",
                        category=category,
                    )
                )
                existing_count += 1

        return self._components

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_components(self, rng: random.Random, n: int) -> list[Component]:
        """Sample n components weighted by pheromone level.

        Uses weighted sampling without replacement. Components with higher
        pheromone are more likely to be selected.

        Args:
            rng: Seeded random instance for reproducibility.
            n: Number of components to sample.

        Returns:
            List of sampled Component instances.
        """
        if not self._components:
            return []

        n = min(n, len(self._components))
        weights = [max(c.pheromone, 1e-9) for c in self._components]

        # Weighted sampling without replacement
        selected: list[Component] = []
        pool = list(zip(weights, self._components))

        for _ in range(n):
            if not pool:
                break
            total = sum(w for w, _ in pool)
            r = rng.random() * total
            cumulative = 0.0
            chosen_idx = len(pool) - 1
            for i, (w, _) in enumerate(pool):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = i
                    break
            _, chosen = pool.pop(chosen_idx)
            selected.append(chosen)

        return selected

    # ------------------------------------------------------------------
    # Pheromone update
    # ------------------------------------------------------------------

    def update_pheromone(
        self,
        selected_components: list[Component],
        score: float,
        median_score: float,
        evaporation_rate: float = 0.1,
    ) -> None:
        """Update pheromone for selected components and apply evaporation.

        Adds (score - median_score) to components that performed above median.
        Then decays all components by (1 - evaporation_rate).

        Args:
            selected_components: Components used in this ant's prompt.
            score: This ant's score on the evaluation batch.
            median_score: Median score across all ants in this round.
            evaporation_rate: Fraction to decay pheromone each round.
        """
        # Reinforce above-median performers
        if score > median_score:
            delta = score - median_score
            for component in selected_components:
                component.pheromone += delta

        # Evaporation: decay all components
        for component in self._components:
            component.pheromone *= (1.0 - evaporation_rate)
            # Floor to prevent starvation
            component.pheromone = max(component.pheromone, 1e-6)

    # ------------------------------------------------------------------
    # Composition and ranking
    # ------------------------------------------------------------------

    def top_components(self, n: int) -> list[Component]:
        """Return the top-n components sorted by pheromone descending.

        Args:
            n: Number of top components to return.

        Returns:
            Sorted list of top-n Component instances.
        """
        sorted_comps = sorted(self._components, key=lambda c: -c.pheromone)
        return sorted_comps[:n]

    def compose_prompt(self, components: list[Component]) -> str:
        """Compose a prompt string from a list of components.

        Orders components by category (following CATEGORIES order), then
        joins them with newlines.

        Args:
            components: List of Component instances to compose.

        Returns:
            Composed prompt string.
        """
        # Sort by category order, then by pheromone descending within category
        sorted_comps = sorted(
            components,
            key=lambda c: (_CATEGORY_ORDER.get(c.category, 99), -c.pheromone),
        )
        return "\n".join(c.text for c in sorted_comps)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def components(self) -> list[Component]:
        return list(self._components)

    def __len__(self) -> int:
        return len(self._components)
