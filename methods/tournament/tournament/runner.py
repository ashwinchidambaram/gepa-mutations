"""Runner for PTS (Tournament Selection) prompt optimization.

Algorithm:
1. Generate 63 diverse prompts via tracked_reflection. Candidate #0 = seed prompt. Total = 64.
   - 4 generation calls of ~16 prompts each with different strategies.
2. Single-elimination bracket:
   - R1: 32 matchups, 5 examples each (320 rollouts)
   - R2: 16 matchups, 7 examples each (224 rollouts)
   - R3:  8 matchups, 10 examples each (160 rollouts)
   - R4:  4 matchups, 15 examples each (120 rollouts)
   - SF:  2 matchups, 20 examples each (80 rollouts)
   - Final: 1 matchup, full valset
3. Evaluate champion on test via evaluate_on_test(), save.
"""

from __future__ import annotations

import argparse
import random
import re
import time
from typing import Any

from rich.console import Console

from gepa_mutations.base import (
    build_qa_task_lm,
    build_reflection_lm,
    evaluate_on_test,
)
from gepa_mutations.benchmarks.evaluators import get_adapter
from gepa_mutations.benchmarks.loader import load_benchmark
from gepa_mutations.config import PAPER_ROLLOUTS, Settings, model_id, model_tag as get_model_tag
from gepa_mutations.metrics.collector import MetricsCollector
from gepa_mutations.metrics.tracked_lm import TrackedLM
from gepa_mutations.runner.experiment import (
    BENCHMARK_SEED_PROMPTS,
    SEED_PROMPT,
    ExperimentResult,
)
from gepa_mutations.storage.local import save_result

from tournament.bracket import ROUND_EXAMPLES, Tournament

console = Console()

# Target total pool size (must be a power of 2 for a perfect bracket)
_POOL_SIZE = 64
# Number of LLM generation calls
_N_GEN_CALLS = 4
# Prompts per generation call
_PROMPTS_PER_CALL = 16

# Generation strategy templates (one per call)
_GENERATION_STRATEGIES = [
    "Generate {n} diverse system prompts for the following task. "
    "Vary the reasoning strategies (e.g. chain-of-thought, step-by-step, direct answer, analogy-based). "
    "Separate each prompt with '---'. Task: {task}",

    "Generate {n} diverse system prompts for the following task. "
    "Vary the output format instructions (e.g. bullet points, numbered steps, paragraphs, JSON, etc.). "
    "Separate each prompt with '---'. Task: {task}",

    "Generate {n} diverse system prompts for the following task. "
    "Vary the level of detail and specificity (from very concise to highly detailed). "
    "Separate each prompt with '---'. Task: {task}",

    "Generate {n} diverse system prompts for the following task. "
    "Vary the error prevention strategies (e.g. double-checking, listing common mistakes, constraints). "
    "Separate each prompt with '---'. Task: {task}",
]


def _parse_generated_prompts(response: str) -> list[str]:
    """Parse a batch of generated prompts from an LLM response.

    Handles two common formats:
    1. Prompts separated by '---' dividers.
    2. Numbered items: "1. ...", "2. ...", etc.

    Args:
        response: Raw LLM response string.

    Returns:
        List of individual prompt strings (non-empty only).
    """
    if not response or not response.strip():
        return []

    # Try "---" separator first
    if "---" in response:
        parts = response.split("---")
        prompts = [p.strip() for p in parts if p.strip()]
        if len(prompts) >= 2:
            return prompts

    # Try numbered items: lines starting with a digit and period/paren
    numbered_pattern = re.compile(
        r"(?m)^(?=\s*\d+[\.\)]\s)",
    )
    parts = numbered_pattern.split(response)
    parts = [p.strip() for p in parts if p.strip()]

    # Strip leading "N. " or "N) " from each item
    cleaned = []
    for part in parts:
        cleaned_part = re.sub(r"^\d+[\.\)]\s*", "", part).strip()
        if cleaned_part:
            cleaned.append(cleaned_part)

    if len(cleaned) >= 2:
        return cleaned

    # Fall back: treat entire response as a single prompt
    return [response.strip()]


def _generate_candidate_pool(
    tracked_reflection: TrackedLM,
    seed_prompt: str,
    rng: random.Random,
    pool_size: int = _POOL_SIZE,
) -> list[str]:
    """Generate a pool of candidates (seed + generated).

    Makes 4 generation calls with diverse strategies. Adjusts prompts per
    call based on pool_size. If a call fails or returns fewer than expected,
    uses the seed prompt as a fallback to pad the pool to exactly pool_size.

    Args:
        tracked_reflection: Token-tracked reflection LM.
        seed_prompt: Canonical seed prompt for this benchmark.
        rng: Random generator (for shuffling the pool).
        pool_size: Total pool size including seed candidate.

    Returns:
        List of exactly pool_size prompt strings. Candidate #0 is the seed.
    """
    generated: list[str] = []
    target_generated = pool_size - 1
    n_per_call = max(1, (target_generated + _N_GEN_CALLS - 1) // _N_GEN_CALLS)

    for call_idx, strategy_template in enumerate(_GENERATION_STRATEGIES):
        if len(generated) >= target_generated:
            break
        n_needed = min(n_per_call, target_generated - len(generated))
        generation_prompt = strategy_template.format(n=n_needed, task=seed_prompt)

        try:
            response = tracked_reflection(generation_prompt)
            parsed = _parse_generated_prompts(response)
            if parsed:
                generated.extend(parsed[:n_needed])
                console.print(f"  Call {call_idx + 1}: got {min(len(parsed), n_needed)} prompts")
            else:
                console.print(f"  [yellow]Call {call_idx + 1}: empty response; skipping[/yellow]")
        except Exception as e:
            console.print(f"  [yellow]Call {call_idx + 1}: failed ({e}); skipping[/yellow]")

    if len(generated) < target_generated:
        shortfall = target_generated - len(generated)
        console.print(
            f"  [yellow]Only {len(generated)} prompts generated; padding with {shortfall} seed copies[/yellow]"
        )
        generated.extend([seed_prompt] * shortfall)

    # Trim to exactly target_generated and shuffle (seed is added separately at front)
    generated = generated[:target_generated]
    rng.shuffle(generated)

    # Candidate #0 is always the seed prompt
    pool = [seed_prompt] + generated
    console.print(f"  Total pool size: {len(pool)}")
    return pool


def run_tournament(
    benchmark: str,
    seed: int,
    subset: int | None = None,
    max_metric_calls: int | None = None,
    pool_size: int = _POOL_SIZE,
    settings: Settings | None = None,
) -> ExperimentResult:
    """Run the PTS (Tournament Selection) prompt optimization experiment.

    Args:
        benchmark: Benchmark name (hotpotqa, ifbench, hover, pupa, etc.).
        seed: Random seed for reproducibility.
        subset: If set, use only this many train/val examples.
        max_metric_calls: Rollout budget override (defaults to paper budget).
        settings: Environment settings (loaded from .env if not provided).

    Returns:
        ExperimentResult with scores, best prompt, and diagnostics.
    """
    settings = settings or Settings()
    start_time = time.time()
    collector = MetricsCollector()
    rng = random.Random(seed)

    # =========================================================================
    # 1. Load benchmark data
    # =========================================================================
    console.print(f"[bold]Loading benchmark: {benchmark}[/bold]")
    data = load_benchmark(benchmark, seed=0)  # ALWAYS seed=0 for data loading
    console.print(f"  Train: {len(data.train)}, Val: {len(data.val)}, Test: {len(data.test)}")

    trainset = data.train[:subset] if subset else data.train
    valset = data.val[:subset] if subset else data.val
    testset = data.test

    # =========================================================================
    # 2. Build LMs and adapter
    # =========================================================================
    qa_lm = build_qa_task_lm(settings)
    tracked_task_lm = TrackedLM(qa_lm, collector, role="task")
    adapter = get_adapter(benchmark, task_lm=tracked_task_lm)
    reflection_lm = build_reflection_lm(settings)
    tracked_reflection = TrackedLM(reflection_lm, collector, role="reflection")

    # =========================================================================
    # 3. Seed prompt (canonical source)
    # =========================================================================
    seed_prompt = BENCHMARK_SEED_PROMPTS.get(benchmark, SEED_PROMPT)

    # =========================================================================
    # 4. Budget
    # =========================================================================
    budget = max_metric_calls or PAPER_ROLLOUTS["gepa"].get(benchmark, 5000)

    console.print(f"[bold]Running PTS (Tournament Selection)[/bold]")
    console.print(f"  Benchmark: {benchmark}, Seed: {seed}")
    console.print(f"  Rollout budget: {budget}")
    console.print(f"  Target pool size: {pool_size}")

    # =========================================================================
    # 5. Generate candidate pool
    # =========================================================================
    console.print("\n[bold]Step 1: Generating candidate pool...[/bold]")
    pool = _generate_candidate_pool(tracked_reflection, seed_prompt, rng, pool_size=pool_size)

    # =========================================================================
    # 6. Run tournament bracket
    # =========================================================================
    console.print("\n[bold]Step 2: Running tournament bracket...[/bold]")
    bracket = Tournament(candidates=pool, rng=rng)
    champion_prompt, matchup_results = bracket.run_tournament(
        adapter=adapter,
        trainset=trainset,
        valset=valset,
        collector=collector,
        budget=budget,
    )

    # Extract champion's final score from the last matchup (the final)
    final_matchup = next(
        (m for m in reversed(matchup_results) if m.get("is_final")),
        matchup_results[-1] if matchup_results else None,
    )
    best_val_score = final_matchup["winner_score"] if final_matchup else 0.0
    # If budget was exhausted and val_score is carried from best_known_score,
    # it may still be 0.0 if no real matchups ran. Fall back to max real winner score.
    if best_val_score == 0.0 and matchup_results:
        real_scores = [m["winner_score"] for m in matchup_results if not m.get("budget_exhausted")]
        if real_scores:
            best_val_score = max(real_scores)

    console.print(f"\n  Champion val score: {best_val_score:.4f}")
    collector.record_val_score(iteration=1, score=best_val_score)

    best_prompt_dict = {"system_prompt": champion_prompt}

    # =========================================================================
    # 7. Populate method-specific metrics
    # =========================================================================
    n_rounds = len(set(m["round"] for m in matchup_results))
    bracket_depth = len(set(m["round"] for m in matchup_results))
    collector.method_specific.update({
        "pool_size": len(pool),
        "rounds": n_rounds,
        "matchup_results": matchup_results,
        "bracket_depth": bracket_depth,
    })

    # =========================================================================
    # 8. Evaluate on test set
    # =========================================================================
    console.print(f"\n[bold]Evaluating on test set ({len(testset)} examples)...[/bold]")
    test_eval = evaluate_on_test(benchmark, best_prompt_dict, testset, settings)
    console.print(f"  Test score: {test_eval.score:.4f} ({test_eval.score * 100:.2f}%)")

    # =========================================================================
    # 9. Save results
    # =========================================================================
    metrics_data = collector.finalize(
        test_score=test_eval.score,
        best_prompt=best_prompt_dict,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
        model=model_id(settings),
        model_tag=get_model_tag(settings),
        benchmark=benchmark,
        seed=seed,
        method="tournament",
    )

    config_snap = {
        "benchmark": benchmark,
        "seed": seed,
        "subset": subset,
        "method": "tournament",
        "model": model_id(settings),
        "temperature": settings.gepa_temperature,
        "top_p": settings.gepa_top_p,
        "top_k": settings.gepa_top_k,
        "rollout_budget": budget,
        "pool_size": pool_size,
        "n_gen_calls": _N_GEN_CALLS,
        "prompts_per_call": _PROMPTS_PER_CALL,
        "round_examples": ROUND_EXAMPLES,
        "bracket_depth": bracket_depth,
    }

    exp_result = ExperimentResult(
        benchmark=benchmark,
        seed=seed,
        test_score=test_eval.score,
        val_score=best_val_score,
        best_prompt=best_prompt_dict,
        rollout_count=collector.rollout_count,
        config_snapshot=config_snap,
        wall_clock_seconds=time.time() - start_time,
        method="tournament",
        metrics=metrics_data,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
    )

    mtagval = get_model_tag(settings)
    save_result(
        benchmark=benchmark,
        seed=seed,
        result_data=exp_result.to_dict(),
        config_data=exp_result.config_snapshot,
        metrics_data=metrics_data,
        method="tournament",
        model_tag=mtagval,
    )
    console.print(f"  Results saved to runs/{mtagval + '/' if mtagval else ''}{benchmark}/tournament/{seed}/")

    return exp_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PTS (Tournament Selection) optimization")
    parser.add_argument("--benchmark", "-b", required=True, help="Benchmark name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subset", "-s", type=int, default=None, help="Use only N train/val examples")
    parser.add_argument("--max-metric-calls", "-m", type=int, default=None, help="Rollout budget override")
    parser.add_argument("--pool-size", type=int, default=_POOL_SIZE, help="Candidate pool size (default: 64)")
    args = parser.parse_args()
    run_tournament(args.benchmark, args.seed, args.subset, args.max_metric_calls, pool_size=args.pool_size)
