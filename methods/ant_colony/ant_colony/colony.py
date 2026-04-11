"""AntColony: main optimization loop for ACPCO.

Each round, `n_ants` synthetic ants independently sample components from the
library (weighted by pheromone), compose prompts from those components, and
evaluate them on a mini-batch. Pheromone is updated based on performance
relative to the round's median, then decayed by evaporation.

After all rounds, the top-15 components by pheromone are passed to the
reflection LM for a final polish pass.
"""

from __future__ import annotations

import random
import statistics
from typing import Any

from gepa_mutations.metrics.collector import MetricsCollector
from gepa_mutations.metrics.standalone_eval import evaluate_prompt

from ant_colony.components import Component, ComponentLibrary


class AntColony:
    """Runs the ACPCO optimization loop.

    Args:
        library: Initialized ComponentLibrary with generated components.
        n_ants: Number of ants (candidate prompts) per round. Default: 3.
        n_components: Number of components each ant selects. Default: 10.
        n_rounds: Total number of optimization rounds. Default: 50.
        evaporation_rate: Pheromone decay fraction per round. Default: 0.1.
    """

    def __init__(
        self,
        library: ComponentLibrary,
        n_ants: int = 3,
        n_components: int = 10,
        n_rounds: int = 50,
        evaporation_rate: float = 0.1,
    ) -> None:
        self.library = library
        self.n_ants = n_ants
        self.n_components = n_components
        self.n_rounds = n_rounds
        self.evaporation_rate = evaporation_rate

        self._best_score: float = 0.0
        self._best_prompt: str = ""
        self._best_components: list[Component] = []
        self._pheromone_history: list[dict[str, Any]] = []

    def run_round(
        self,
        adapter: Any,
        trainset: list,
        n_eval_examples: int,
        collector: MetricsCollector,
        rng: random.Random,
    ) -> list[tuple[float, list[Component]]]:
        """Run a single ant colony round.

        Each ant samples components, composes a prompt, and evaluates it.
        Pheromone is updated after all ants are scored.

        Args:
            adapter: GEPAAdapter for prompt evaluation.
            trainset: Full training dataset.
            n_eval_examples: Number of examples to evaluate each ant on.
            collector: MetricsCollector for rollout tracking.
            rng: Seeded random for reproducible sampling.

        Returns:
            List of (score, selected_components) for each ant, sorted by score
            descending.
        """
        n_eval_examples = min(n_eval_examples, len(trainset))
        indices = rng.sample(range(len(trainset)), k=n_eval_examples)

        results: list[tuple[float, list[Component]]] = []

        for _ in range(self.n_ants):
            # Each ant selects 8-12 components (randomized within range)
            n_select = rng.randint(
                max(1, self.n_components - 2),
                min(len(self.library), self.n_components + 2),
            )
            selected = self.library.sample_components(rng, n=n_select)
            if not selected:
                results.append((0.0, []))
                continue

            prompt_text = self.library.compose_prompt(selected)
            candidate = {"system_prompt": prompt_text}
            score, _ = evaluate_prompt(adapter, trainset, candidate, collector, indices=indices)
            results.append((score, selected))

        # Compute median score across ants for this round
        scores = [r[0] for r in results]
        median_score = statistics.median(scores) if scores else 0.0

        # Update pheromone for each ant's components
        for score, selected in results:
            if selected:
                self.library.update_pheromone(
                    selected_components=selected,
                    score=score,
                    median_score=median_score,
                    evaporation_rate=self.evaporation_rate,
                )

        # Track global best
        for score, selected in results:
            if score > self._best_score and selected:
                self._best_score = score
                self._best_components = selected
                self._best_prompt = self.library.compose_prompt(selected)

        return sorted(results, key=lambda x: -x[0])

    def run(
        self,
        adapter: Any,
        trainset: list,
        collector: MetricsCollector,
        rng: random.Random,
        budget: int | None = None,
    ) -> str:
        """Run all optimization rounds and return the best prompt text.

        Args:
            adapter: GEPAAdapter for prompt evaluation.
            trainset: Full training dataset.
            collector: MetricsCollector for rollout tracking.
            rng: Seeded random for reproducibility.
            budget: Optional hard rollout cap. Stops early when reached.

        Returns:
            Best prompt text seen across all rounds.
        """
        for round_idx in range(self.n_rounds):
            if budget is not None and collector.rollout_count >= budget:
                break

            self.run_round(
                adapter=adapter,
                trainset=trainset,
                n_eval_examples=10,
                collector=collector,
                rng=rng,
            )

            # Record pheromone history every round
            top5 = self.library.top_components(5)
            self._pheromone_history.append({
                "round": round_idx + 1,
                "top_5_components": [c.to_dict() for c in top5],
                "best_score_so_far": self._best_score,
            })

        return self._best_prompt or ""

    def polish_prompt(
        self,
        reflection_lm: Any,
        best_components: list[Component] | None = None,
    ) -> str:
        """Use the reflection LM to compose a polished prompt from top components.

        Args:
            reflection_lm: Callable LM for the polish pass.
            best_components: Components to polish. Defaults to top-15 by pheromone.

        Returns:
            Polished prompt string, or the raw composed prompt if the LLM fails.
        """
        if best_components is None:
            best_components = self.library.top_components(15)

        raw_prompt = self.library.compose_prompt(best_components)

        polish_request = (
            f"You are refining a system prompt for an AI assistant. "
            f"The following instructions have been identified as the most effective "
            f"components for this task:\n\n"
            f"{raw_prompt}\n\n"
            f"Please combine and polish these instructions into a single, coherent "
            f"system prompt. Remove any redundancy, ensure smooth transitions, and "
            f"preserve all key guidance. Output ONLY the final polished prompt — "
            f"no explanations or commentary."
        )
        try:
            result = reflection_lm(polish_request)
            result = result.strip()
            if result:
                return result
        except Exception:
            pass

        return raw_prompt

    # ------------------------------------------------------------------
    # Metrics accessors
    # ------------------------------------------------------------------

    @property
    def pheromone_history(self) -> list[dict[str, Any]]:
        return self._pheromone_history

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_prompt(self) -> str:
        return self._best_prompt

    @property
    def best_components(self) -> list[Component]:
        return self._best_components
