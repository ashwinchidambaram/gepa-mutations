"""Tournament bracket logic for PTS (Tournament Selection) prompt optimization.

Implements a single-elimination tournament where prompts compete head-to-head.
Each matchup evaluates both prompts on the SAME randomly sampled examples —
higher mean score wins. On a tie the first prompt (lower seeding) advances.

Round schedule (per the spec):
    R1: 32 matchups, 5 examples each
    R2: 16 matchups, 7 examples each
    R3:  8 matchups, 10 examples each
    R4:  4 matchups, 15 examples each
    SF:  2 matchups, 20 examples each
    Final: 1 matchup, full valset
"""

from __future__ import annotations

import random
from typing import Any

from gepa_mutations.metrics.collector import MetricsCollector
from gepa_mutations.metrics.standalone_eval import evaluate_prompt

# Number of examples per matchup for rounds 1–5 (R1–SF); final uses full valset
ROUND_EXAMPLES = [5, 7, 10, 15, 20]


class Tournament:
    """Single-elimination tournament over a pool of prompt candidates.

    Args:
        candidates: List of prompt strings (should be a power of 2 for a
            perfectly balanced bracket; extras get byes automatically).
        rng: Random generator for reproducible example sampling.
    """

    def __init__(self, candidates: list[str], rng: random.Random) -> None:
        self.candidates = list(candidates)
        self.rng = rng
        self._matchup_results: list[dict[str, Any]] = []
        self.best_known_score: float = 0.0

    def run_round(
        self,
        adapter: Any,
        trainset: list,
        n_examples: int,
        collector: MetricsCollector,
        round_num: int = 0,
        budget: int | None = None,
    ) -> list[str]:
        """Run one round of the bracket, returning the list of winners.

        Pairs candidates sequentially (index 0 vs 1, index 2 vs 3, etc.).
        If there is an odd number of candidates, the last one gets a bye.
        If budget is set and would be exceeded before a matchup, the first
        candidate (lower seed) of each remaining matchup advances without
        evaluation.

        Args:
            adapter: GEPAAdapter for scoring.
            trainset: Training examples to sample from.
            n_examples: Number of examples per matchup (same for all matchups in this round).
            collector: MetricsCollector for rollout counting.
            round_num: Current round index (0-based), used for logging.
            budget: Optional hard rollout cap. Stops evaluating when reached.

        Returns:
            List of winning prompt strings.
        """
        winners: list[str] = []
        n = len(self.candidates)
        n_matchups = n // 2

        # Sample the example indices once per round and reuse across all matchups
        # so that head-to-head comparisons within a round are on the same examples.
        max_idx = len(trainset)
        indices_per_matchup: list[list[int]] = []
        for _ in range(n_matchups):
            sample_size = min(n_examples, max_idx)
            sampled = self.rng.sample(range(max_idx), sample_size)
            indices_per_matchup.append(sampled)

        for match_idx in range(n_matchups):
            prompt_a = self.candidates[match_idx * 2]
            prompt_b = self.candidates[match_idx * 2 + 1]
            indices = indices_per_matchup[match_idx]

            # If budget is exhausted, advance prompt_a without evaluation
            if budget is not None and collector.rollout_count >= budget:
                winners.append(prompt_a)
                self._matchup_results.append({
                    "round": round_num,
                    "match_idx": match_idx,
                    "winner_idx": match_idx * 2,
                    "loser_idx": match_idx * 2 + 1,
                    "winner_score": self.best_known_score,
                    "loser_score": 0.0,
                    "n_examples": 0,
                    "budget_exhausted": True,
                })
                continue

            score_a, _ = evaluate_prompt(
                adapter, trainset, {"system_prompt": prompt_a}, collector, indices=indices
            )
            score_b, _ = evaluate_prompt(
                adapter, trainset, {"system_prompt": prompt_b}, collector, indices=indices
            )

            # Higher score wins; on tie, prompt_a (first) advances
            if score_b > score_a:
                winner = prompt_b
                loser = prompt_a
                winner_score, loser_score = score_b, score_a
                winner_idx = match_idx * 2 + 1
                loser_idx = match_idx * 2
            else:
                winner = prompt_a
                loser = prompt_b
                winner_score, loser_score = score_a, score_b
                winner_idx = match_idx * 2
                loser_idx = match_idx * 2 + 1

            self.best_known_score = max(self.best_known_score, winner_score)
            winners.append(winner)
            self._matchup_results.append({
                "round": round_num,
                "match_idx": match_idx,
                "winner_idx": winner_idx,
                "loser_idx": loser_idx,
                "winner_score": winner_score,
                "loser_score": loser_score,
                "n_examples": len(indices),
            })

        # Bye: if odd number of candidates, the last one advances automatically
        if n % 2 == 1:
            bye_prompt = self.candidates[-1]
            winners.append(bye_prompt)
            self._matchup_results.append({
                "round": round_num,
                "match_idx": n_matchups,
                "winner_idx": n - 1,
                "loser_idx": -1,  # No opponent
                "winner_score": 1.0,  # Bye
                "loser_score": 0.0,
                "n_examples": 0,
                "bye": True,
            })

        # Advance winners to next round
        self.candidates = winners
        return winners

    def run_tournament(
        self,
        adapter: Any,
        trainset: list,
        valset: list,
        collector: MetricsCollector,
        budget: int | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Run the full tournament bracket and return the champion.

        Runs rounds according to ROUND_EXAMPLES schedule, then the final
        matchup on the full valset. If budget is set, stops early and returns
        the current leader when the budget is exhausted.

        Args:
            adapter: GEPAAdapter for scoring.
            trainset: Training examples for preliminary rounds.
            valset: Validation set for the final matchup.
            collector: MetricsCollector for rollout counting.
            budget: Optional hard rollout cap. Stops the tournament early
                when the cap is reached.

        Returns:
            Tuple of (champion_prompt_str, matchup_results_list).
        """
        from rich.console import Console
        console = Console()

        total_rounds = len(ROUND_EXAMPLES) + 1  # preliminary rounds + final

        for round_idx, n_examples in enumerate(ROUND_EXAMPLES):
            if len(self.candidates) <= 1:
                break
            if budget is not None and collector.rollout_count >= budget:
                console.print("  [yellow]Budget exhausted; stopping tournament early[/yellow]")
                break

            round_label = f"R{round_idx + 1}"
            n_matchups = len(self.candidates) // 2
            console.print(
                f"  [{round_label}] {len(self.candidates)} candidates, "
                f"{n_matchups} matchups, {n_examples} examples each"
            )

            self.run_round(
                adapter=adapter,
                trainset=trainset,
                n_examples=n_examples,
                collector=collector,
                round_num=round_idx,
                budget=budget,
            )
            console.print(f"    -> {len(self.candidates)} survivors")

        # Final: 1 matchup on full valset
        if len(self.candidates) >= 2:
            if budget is not None and collector.rollout_count >= budget:
                # Budget exhausted — declare the first remaining candidate champion
                console.print("  [yellow]Budget exhausted; skipping final, using current leader[/yellow]")
                self._matchup_results.append({
                    "round": len(ROUND_EXAMPLES),
                    "match_idx": 0,
                    "winner_idx": 0,
                    "loser_idx": 1,
                    "winner_score": self.best_known_score,
                    "loser_score": 0.0,
                    "n_examples": 0,
                    "is_final": True,
                    "budget_exhausted": True,
                })
                self.candidates = [self.candidates[0]]
            else:
                console.print(
                    f"  [Final] {len(self.candidates)} candidates, full valset ({len(valset)} examples)"
                )
                # For the final we use the full valset — pass indices as full range
                full_val_indices = list(range(len(valset)))
                round_num = len(ROUND_EXAMPLES)

                prompt_a = self.candidates[0]
                prompt_b = self.candidates[1]

                score_a, _ = evaluate_prompt(
                    adapter, valset, {"system_prompt": prompt_a}, collector, indices=full_val_indices
                )
                score_b, _ = evaluate_prompt(
                    adapter, valset, {"system_prompt": prompt_b}, collector, indices=full_val_indices
                )

                if score_b > score_a:
                    champion = prompt_b
                    winner_idx, loser_idx = 1, 0
                    winner_score, loser_score = score_b, score_a
                else:
                    champion = prompt_a
                    winner_idx, loser_idx = 0, 1
                    winner_score, loser_score = score_a, score_b

                self.best_known_score = max(self.best_known_score, winner_score)
                self._matchup_results.append({
                    "round": round_num,
                    "match_idx": 0,
                    "winner_idx": winner_idx,
                    "loser_idx": loser_idx,
                    "winner_score": winner_score,
                    "loser_score": loser_score,
                    "n_examples": len(full_val_indices),
                    "is_final": True,
                })
                self.candidates = [champion]
                console.print(f"  Champion score (full valset): {winner_score:.4f}")
        elif len(self.candidates) == 1:
            # Only one candidate survived all rounds
            champion = self.candidates[0]
        else:
            raise RuntimeError("Tournament ended with no candidates — check input pool size.")

        return self.candidates[0], self._matchup_results
