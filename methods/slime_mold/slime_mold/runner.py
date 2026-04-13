"""Runner for the SMNO (Slime Mold Network Optimization) prompt experiment.

SMNO searches the prompt space through progressive pruning:
1. Generate 19 diverse candidates + seed = 20 total.
2. Four pruning rounds that evaluate on progressively larger subsets
   and keep fewer survivors each round (20 -> 10 -> 5 -> 3 -> 1).
3. Between rounds, surviving prompts are mutated using failure information.
4. Champion is evaluated on full val set, then test set, and results are saved.

Budget: ~540 rollouts + ~22 LLM calls.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Any

from rich.console import Console

from gepa_mutations.base import build_qa_task_lm, build_reflection_lm, evaluate_on_test
from gepa_mutations.benchmarks.evaluators import get_adapter
from gepa_mutations.benchmarks.loader import load_benchmark
from gepa_mutations.config import PAPER_ROLLOUTS, Settings, model_tag as get_model_tag
from gepa_mutations.metrics.collector import MetricsCollector
from gepa_mutations.metrics.standalone_eval import evaluate_prompt
from gepa_mutations.metrics.tracked_lm import TrackedLM
from gepa_mutations.runner.experiment import BENCHMARK_SEED_PROMPTS, SEED_PROMPT, ExperimentResult
from gepa_mutations.storage.local import save_result

from slime_mold.colony import generate_diverse_prompts, mutate_prompt, run_pruning_round

console = Console()

METHOD_NAME = "slime_mold"

# Progressive pruning schedule: (n_examples_per_eval, keep_top_k)
_PRUNING_SCHEDULE = [
    (10, 10),   # R1: 20 candidates × 10 examples = 200 rollouts → keep 10
    (15, 5),    # R2: 10 candidates × 15 examples = 150 rollouts → keep 5
    (20, 3),    # R3: 5 candidates × 20 examples = 100 rollouts → keep 3
    (30, 1),    # R4: 3 candidates × 30 examples = 90 rollouts → keep 1
]


def run_slime_mold(
    benchmark: str = "hotpotqa",
    seed: int = 42,
    subset: int | None = None,
    max_metric_calls: int | None = None,
    settings: Settings | None = None,
) -> ExperimentResult:
    """Run the SMNO (Slime Mold Network Optimization) experiment.

    Args:
        benchmark: Benchmark name (hotpotqa, ifbench, hover, pupa, etc.).
        seed: Random seed for reproducibility.
        subset: If set, limit train/val to this many examples.
        max_metric_calls: Rollout budget override (defaults to paper budget).
        settings: Environment settings (loaded from .env if not provided).

    Returns:
        ExperimentResult with test/val scores, best prompt, and diagnostics.
    """
    settings = settings or Settings()
    start_time = time.time()
    rng = random.Random(seed)
    collector = MetricsCollector()

    # =========================================================================
    # 1. Load benchmark data
    # =========================================================================
    console.print(f"[bold]Loading benchmark: {benchmark}[/bold]")
    data = load_benchmark(benchmark, seed=0)
    console.print(f"  Train: {len(data.train)}, Val: {len(data.val)}, Test: {len(data.test)}")

    trainset = data.train[:subset] if subset is not None else data.train
    valset = data.val[:subset] if subset is not None else data.val
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
    # 3. Budget
    # =========================================================================
    if max_metric_calls is None:
        max_metric_calls = PAPER_ROLLOUTS["gepa"].get(benchmark, 5000)

    # =========================================================================
    # 4. Seed prompt and task description
    # =========================================================================
    seed_prompt = BENCHMARK_SEED_PROMPTS.get(benchmark, SEED_PROMPT)
    seed_candidate = {"system_prompt": seed_prompt}
    task_description = f"Benchmark: {benchmark}. Seed prompt: {seed_prompt}"

    console.print(f"\n[bold]Running SMNO optimization[/bold]")
    console.print(f"  Benchmark: {benchmark}, Seed: {seed}")
    console.print(f"  Train: {len(trainset)}, Val: {len(valset)}")
    console.print(f"  Rollout budget: {max_metric_calls}")

    # Evaluate seed prompt on test and val sets BEFORE optimization
    console.print(f"\n[bold]Evaluating seed prompt on test set ({len(testset)} examples)...[/bold]")
    seed_test_eval = evaluate_on_test(benchmark, seed_candidate, testset, settings)
    console.print(f"  Seed test score: {seed_test_eval.score:.4f}")
    console.print(f"\n[bold]Evaluating seed prompt on val set ({len(data.val)} examples)...[/bold]")
    seed_val_eval = evaluate_on_test(benchmark, seed_candidate, data.val, settings)
    console.print(f"  Seed val score: {seed_val_eval.score:.4f}")
    seed_prompt_test_score = seed_test_eval.score
    seed_prompt_val_score = seed_val_eval.score

    # Inject rollout=0 as the first trajectory point
    collector.record_val_score(iteration=0, score=seed_prompt_val_score,
                               prompt_length=len(seed_prompt))

    # =========================================================================
    # 5. Generate diverse candidates (19 + seed = 20 total)
    # =========================================================================
    console.print(f"\n[bold cyan]Generating diverse candidates...[/bold cyan]")
    _stage_start = time.time()
    _stage_rollouts_start = collector.rollout_count
    diverse = generate_diverse_prompts(
        reflection_lm=tracked_reflection,
        seed_prompt=seed_prompt,
        n=19,
        task_description=task_description,
        rng=rng,
    )
    candidates: list[str] = [seed_prompt] + diverse[:19]
    # Ensure exactly 20
    while len(candidates) < 20:
        candidates.append(seed_prompt)
    candidates = candidates[:20]

    console.print(f"  Generated {len(candidates)} candidates (including seed)")
    collector.method_specific.setdefault("stage_timings", []).append({
        "stage": "candidate_generation",
        "seconds": round(time.time() - _stage_start, 2),
        "rollouts_used": collector.rollout_count - _stage_rollouts_start,
    })

    # =========================================================================
    # 6. Progressive pruning (4 rounds)
    # =========================================================================
    method_specific: dict[str, Any] = {
        "round_survivors": [len(candidates)],
        "round_scores": [],
        "pruning_history": [],
    }

    _running_best_score = 0.0
    _running_best_prompt = seed_prompt

    for round_idx, (n_examples, keep_k) in enumerate(_PRUNING_SCHEDULE):
        round_num = round_idx + 1
        if collector.rollout_count >= max_metric_calls:
            console.print(
                f"\n[yellow]Budget exhausted before Round {round_num}; stopping early[/yellow]"
            )
            break

        console.print(
            f"\n[bold cyan]Round {round_num}/4[/bold cyan]: "
            f"{len(candidates)} candidates × {n_examples} examples "
            f"→ keep top {keep_k}"
        )

        _stage_start = time.time()
        _stage_rollouts_start = collector.rollout_count

        # Evaluate all candidates for this round
        sorted_candidates, sorted_scores, _ = run_pruning_round(
            adapter=adapter,
            candidates=candidates,
            trainset=trainset,
            n_examples=n_examples,
            collector=collector,
            rng=rng,
        )

        # Record per-candidate scores
        for rank, (score, cand_text) in enumerate(zip(sorted_scores, sorted_candidates)):
            method_specific["round_scores"].append({
                "round": round_num,
                "candidate_rank": rank,
                "score": score,
            })

        best_score = sorted_scores[0] if sorted_scores else 0.0
        worst_score = sorted_scores[-1] if sorted_scores else 0.0
        method_specific["pruning_history"].append({
            "round": round_num,
            "n_candidates": len(candidates),
            "n_examples": n_examples,
            "best_score": best_score,
            "worst_score": worst_score,
        })

        console.print(
            f"  Best score: {best_score:.4f}, Worst: {worst_score:.4f}"
        )

        if best_score > _running_best_score:
            _running_best_score = best_score
            _running_best_prompt = sorted_candidates[0] if sorted_candidates else seed_prompt

        collector.method_specific.setdefault("stage_timings", []).append({
            "stage": f"round_{round_num}",
            "seconds": round(time.time() - _stage_start, 2),
            "rollouts_used": collector.rollout_count - _stage_rollouts_start,
        })
        collector.record_val_score(iteration=round_num, score=_running_best_score,
                                   prompt_length=len(_running_best_prompt))

        # Keep top-k survivors
        survivors = sorted_candidates[:keep_k]
        survivor_scores = sorted_scores[:keep_k]
        method_specific["round_survivors"].append(len(survivors))

        # Between rounds: mutate survivors using failure info (skip after last round)
        if round_idx < len(_PRUNING_SCHEDULE) - 1:
            console.print(f"  Mutating {len(survivors)} survivors...")
            mutated: list[str] = []
            for prompt_text, score in zip(survivors, survivor_scores):
                # Gather some failure examples for the mutation call
                # We re-eval on a small sample to get failure info
                failure_sample_indices = rng.sample(
                    range(len(trainset)), k=min(5, len(trainset))
                )
                failures: list[dict[str, Any]] = []
                for idx in failure_sample_indices:
                    if collector.rollout_count >= max_metric_calls:
                        break
                    ex = trainset[idx]
                    try:
                        eval_out = adapter.evaluate(
                            [ex], {"system_prompt": prompt_text}, capture_traces=False
                        )
                        collector.record_rollouts(n=1)
                        example_score = eval_out.scores[0] if eval_out.scores else 0.0
                        if example_score < 0.5:
                            failures.append({
                                "input": getattr(ex, "input", str(ex)),
                                "expected": getattr(ex, "output", getattr(ex, "answer", "")),
                                "got": "",
                            })
                    except Exception:
                        pass

                improved = mutate_prompt(
                    reflection_lm=tracked_reflection,
                    prompt=prompt_text,
                    score=score,
                    failures=failures,
                )
                mutated.append(improved)
            candidates = mutated
        else:
            candidates = survivors

    # =========================================================================
    # 7. Champion: the single surviving candidate
    # =========================================================================
    champion = candidates[0] if candidates else seed_prompt
    console.print(f"\n[bold green]Champion prompt selected[/bold green]")
    console.print(f"  Rollouts so far: {collector.rollout_count}")

    # =========================================================================
    # 8. Evaluate champion on full val set
    # =========================================================================
    console.print(f"\n[bold]Evaluating champion on full val set ({len(valset)} examples)...[/bold]")
    _stage_start = time.time()
    _stage_rollouts_start = collector.rollout_count
    val_score, val_scores = evaluate_prompt(
        adapter, valset, {"system_prompt": champion}, collector
    )
    collector.method_specific.setdefault("stage_timings", []).append({
        "stage": "champion_eval",
        "seconds": round(time.time() - _stage_start, 2),
        "rollouts_used": collector.rollout_count - _stage_rollouts_start,
    })
    collector.record_val_score(iteration=5, score=val_score, prompt_length=len(champion))
    console.print(f"  Val score: {val_score:.4f} ({val_score * 100:.2f}%)")

    # =========================================================================
    # 9. Evaluate on test set
    # =========================================================================
    console.print(f"\n[bold]Evaluating on test set ({len(testset)} examples)...[/bold]")
    best_prompt = {"system_prompt": champion}
    test_eval = evaluate_on_test(benchmark, best_prompt, testset, settings)
    console.print(f"  Test score: {test_eval.score:.4f} ({test_eval.score * 100:.2f}%)")

    # Evaluate best prompt on train set
    console.print(f"\n[bold]Evaluating on train set ({len(trainset)} examples)...[/bold]")
    train_eval = evaluate_on_test(benchmark, best_prompt, trainset, settings)
    console.print(f"  Train score: {train_eval.score:.4f}")

    # Evaluate best prompt on val set (per-example scores)
    console.print(f"\n[bold]Evaluating on val set ({len(valset)} examples)...[/bold]")
    val_eval = evaluate_on_test(benchmark, best_prompt, valset, settings)

    wall_clock = time.time() - start_time

    # =========================================================================
    # 10. Build and save result
    # =========================================================================
    from gepa_mutations.config import model_id
    mtagval = get_model_tag(settings)

    config_snapshot = {
        "benchmark": benchmark,
        "seed": seed,
        "subset": subset,
        "method_name": METHOD_NAME,
        "model": model_id(settings),
        "max_metric_calls": max_metric_calls,
        "n_initial_candidates": 20,
        "pruning_schedule": _PRUNING_SCHEDULE,
    }

    # Finalize method-specific metrics into collector
    collector.method_specific.update(method_specific)
    metrics_data = collector.finalize(
        test_score=test_eval.score,
        best_prompt=best_prompt,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
        model=model_id(settings),
        model_tag=get_model_tag(settings),
        benchmark=benchmark,
        seed=seed,
        method=METHOD_NAME,
        seed_prompt=seed_prompt,
    )
    metrics_data["train_score"] = train_eval.score
    metrics_data["train_example_scores"] = train_eval.example_scores
    metrics_data["val_example_scores"] = val_eval.example_scores
    metrics_data["val_example_ids"] = val_eval.example_ids

    exp_result = ExperimentResult(
        benchmark=benchmark,
        seed=seed,
        test_score=test_eval.score,
        val_score=val_score,
        best_prompt=best_prompt,
        rollout_count=collector.rollout_count,
        config_snapshot=config_snapshot,
        wall_clock_seconds=wall_clock,
        method=METHOD_NAME,
        metrics=metrics_data,
        all_candidates=[{"system_prompt": c} for c in candidates],
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
        seed_prompt_test_score=seed_prompt_test_score,
        seed_prompt_val_score=seed_prompt_val_score,
        train_score=train_eval.score,
    )

    test_outputs = [
        {
            "example_id": test_eval.example_ids[i],
            "input": getattr(testset[i], 'input', str(testset[i])),
            "expected": getattr(testset[i], 'output', getattr(testset[i], 'answer', '')),
            "output": test_eval.example_outputs[i] if i < len(test_eval.example_outputs) else "",
            "score": test_eval.example_scores[i],
        }
        for i in range(len(testset))
    ]
    save_result(
        benchmark=benchmark,
        seed=seed,
        result_data=exp_result.to_dict(),
        config_data=config_snapshot,
        metrics_data=metrics_data,
        method=METHOD_NAME,
        model_tag=mtagval,
        test_outputs=test_outputs,
    )
    console.print(f"  Results saved to runs/{mtagval + '/' if mtagval else ''}{benchmark}/{METHOD_NAME}/{seed}/")

    console.print(f"\n[bold green]SMNO complete![/bold green]")
    console.print(f"  Val score:  {val_score:.4f}")
    console.print(f"  Test score: {test_eval.score:.4f}")
    console.print(f"  Rollouts:   {collector.rollout_count}")
    console.print(f"  LLM calls:  {collector.reflection_call_count}")
    console.print(f"  Wall clock: {wall_clock:.1f}s")

    return exp_result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SMNO (Slime Mold Network Optimization) prompt experiment"
    )
    parser.add_argument(
        "--benchmark", "-b", required=True,
        help="Benchmark name (hotpotqa, ifbench, hover, pupa, aime, livebench)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--subset", "-s", type=int, default=None,
        help="Limit train/val to this many examples (for quick testing)"
    )
    parser.add_argument(
        "--max-metric-calls", "-m", type=int, default=None,
        help="Rollout budget override (defaults to paper budget)"
    )
    args = parser.parse_args()

    run_slime_mold(
        benchmark=args.benchmark,
        seed=args.seed,
        subset=args.subset,
        max_metric_calls=args.max_metric_calls,
    )
