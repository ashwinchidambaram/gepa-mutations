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
from gepa_mutations.runner.experiment import (
    BENCHMARK_OUTPUT_SHAPES,
    BENCHMARK_SEED_PROMPTS,
    BENCHMARK_TASK_INSTRUCTIONS,
    SEED_PROMPT,
    ExperimentResult,
)
from gepa_mutations.storage.local import save_result

from slime_mold.colony import (
    PRESCRIBED_STRATEGIES,
    build_failure_matrix,
    collect_hard_examples,
    discover_refresh_strategies,
    discover_strategies,
    find_donor,
    generate_diverse_prompts,
    generate_specialized_prompt,
    mutate_prompt,
    mutate_prompt_with_context,
    run_pruning_round,
)
from slime_mold.naming import _derive_method_name

console = Console()


def _write_progress_json(
    benchmark: str,
    seed: int,
    method_name: str,
    model_tag: str,
    rollouts_used: int,
    best_val_score: float,
    iteration: int,
    wall_clock_seconds: float,
) -> None:
    """Write progress.json matching GEPA's schema for orchestrator stall detection."""
    import json
    import os
    from pathlib import Path

    base = Path(os.environ.get("RUNS_DIR", "runs"))
    if model_tag:
        run_dir = base / model_tag / benchmark / method_name / str(seed) / "gepa_state"
    else:
        run_dir = base / benchmark / method_name / str(seed) / "gepa_state"
    run_dir.mkdir(parents=True, exist_ok=True)

    progress_data = {
        "benchmark": benchmark,
        "seed": seed,
        "rollouts_used": rollouts_used,
        "best_val_score": best_val_score,
        "iteration": iteration,
        "wall_clock_seconds": round(wall_clock_seconds, 2),
    }

    target = run_dir / "progress.json"
    tmp = run_dir / "progress.json.tmp"
    tmp.write_text(json.dumps(progress_data, indent=2))
    os.replace(str(tmp), str(target))


def _evaluate_holdout(
    adapter,
    candidate_text: str,
    holdout_examples: list,
    collector: MetricsCollector,
) -> float:
    """Evaluate a candidate on the hold-out set. Returns mean score."""
    if not holdout_examples:
        return 0.0
    candidate = {"system_prompt": candidate_text}
    score, _ = evaluate_prompt(adapter, holdout_examples, candidate, collector)
    return score


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
    # New Phase 3 parameters
    strategy_mode: str = "personality",
    k: int | None = None,
    mutation_mode: str = "blind",
    refresh_mode: str = "none",
    hard_example_threshold: float = 0.7,
) -> ExperimentResult:
    """Run the SMNO (Slime Mold Network Optimization) experiment.

    Args:
        benchmark: Benchmark name (hotpotqa, ifbench, hover, pupa, etc.).
        seed: Random seed for reproducibility.
        subset: If set, limit train/val to this many examples.
        max_metric_calls: Rollout budget override (defaults to paper budget).
        settings: Environment settings (loaded from .env if not provided).
        strategy_mode: Strategy source: personality (baseline), prescribed8, or inductive.
        k: Number of skills for inductive discovery (3, 5, or None for adaptive).
        mutation_mode: Mutation approach: blind (current) or crosspollin.
        refresh_mode: Refresh pass after R1: none, expand, or replace.
        hard_example_threshold: Fraction of candidates that must have failed on an
            example for it to count as "hard" (used by refresh_mode='expand'). Default 0.7.

    Returns:
        ExperimentResult with test/val scores, best prompt, and diagnostics.
    """
    # Phase 4 note: strategy_mode dispatch is implemented below in candidate generation.
    if refresh_mode not in ("none", "expand"):
        raise NotImplementedError(f"refresh_mode={refresh_mode} not yet implemented")

    method_name = _derive_method_name(strategy_mode, k, mutation_mode, refresh_mode)
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

    # Sample 50 fixed hold-out examples (deterministic per benchmark, not per seed)
    _holdout_rng = random.Random(hash(benchmark) & 0xFFFFFFFF)
    _holdout_size = min(50, len(trainset))
    holdout_indices = sorted(_holdout_rng.sample(range(len(trainset)), k=_holdout_size))
    holdout_examples = [trainset[i] for i in holdout_indices]
    console.print(f"  Hold-out: {len(holdout_examples)} examples (deterministic per benchmark)")

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
    # Discovery-only signals — deliberately separate from `seed_prompt` so baselines
    # start from minimal seeds while discovery still has a useful task anchor.
    # Phrased without naming the benchmark to avoid memorization-based skill recall.
    task_description = BENCHMARK_TASK_INSTRUCTIONS.get(
        benchmark, "Solve the examples correctly."
    )
    output_shape = BENCHMARK_OUTPUT_SHAPES.get(
        benchmark, "a response in the format shown by the examples"
    )

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

    # Evaluate seed on hold-out (iteration 0)
    seed_holdout = _evaluate_holdout(adapter, seed_prompt, holdout_examples, collector)
    collector.record_trajectory_point(
        iteration=0,
        holdout_score=seed_holdout,
        best_so_far=seed_holdout,
        prompt_length=len(seed_prompt),
    )
    _running_best_holdout = seed_holdout
    _running_best_holdout_prompt = seed_prompt

    # =========================================================================
    # 5. Generate diverse candidates (dispatch on strategy_mode)
    # =========================================================================
    console.print(f"\n[bold cyan]Generating diverse candidates (mode={strategy_mode})...[/bold cyan]")
    _stage_start = time.time()
    _stage_rollouts_start = collector.rollout_count

    discovery_outputs: list[dict] = []
    strategies_used: list = []

    # candidate_strategies[i] tracks the strategy name for candidates[i]
    # Used by crosspollin mutation to identify donor cross-strategy preference
    candidate_strategies: list[str] = []

    if strategy_mode == "personality":
        # EXISTING CODE PATH — 4 personality strategies → 19 candidates + seed = 20 total
        diverse = generate_diverse_prompts(
            reflection_lm=tracked_reflection,
            seed_prompt=seed_prompt,
            n=19,
            task_description=task_description,
            rng=rng,
        )
        candidates: list[str] = [seed_prompt] + diverse[:19]
        while len(candidates) < 20:
            candidates.append(seed_prompt)
        candidates = candidates[:20]
        candidate_strategies = ["personality"] * len(candidates)

    elif strategy_mode == "prescribed8":
        # 8 prescribed strategies × 3 prompts each = 24 + seed = 25 total
        strategies_used = list(PRESCRIBED_STRATEGIES)
        candidates = [seed_prompt]
        candidate_strategies = ["seed"]
        for strategy in strategies_used:
            specialized = generate_specialized_prompt(
                reflection_lm=tracked_reflection,
                strategy=strategy,
                task_description=task_description,
                seed_prompt=seed_prompt,
                examples=trainset,
                n=3,
                rng=rng,
            )
            for p in specialized[:3]:
                candidates.append(p)
                candidate_strategies.append(strategy.name)
        # Pad if any generations failed
        while len(candidates) < 25:
            candidates.append(seed_prompt)
            candidate_strategies.append("seed")
        candidates = candidates[:25]
        candidate_strategies = candidate_strategies[:25]

    elif strategy_mode == "inductive":
        # Discover k strategies, generate 4 prompts each = 4k + seed
        strategies_used, raw_disc, fallback = discover_strategies(
            reflection_lm=tracked_reflection,
            benchmark=benchmark,
            task_description=task_description,
            output_shape=output_shape,
            examples=trainset,
            k=k,
            rng=rng,
        )
        discovery_outputs.append({
            "pass": "initial",
            "k_requested": k,
            "raw_llm_output": raw_disc,
            "fallback_used": fallback,
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "failure_pattern": s.failure_pattern,
                    "technique": s.technique,
                }
                for s in strategies_used
            ],
        })
        candidates = [seed_prompt]
        candidate_strategies = ["seed"]
        for strategy in strategies_used:
            specialized = generate_specialized_prompt(
                reflection_lm=tracked_reflection,
                strategy=strategy,
                task_description=task_description,
                seed_prompt=seed_prompt,
                examples=trainset,
                n=4,
                rng=rng,
            )
            for p in specialized[:4]:
                candidates.append(p)
                candidate_strategies.append(strategy.name)
        # Final pool size is 4*k + 1 (or 4 * len(strategies_used) + 1 for adaptive)
        target = 4 * len(strategies_used) + 1
        while len(candidates) < target:
            candidates.append(seed_prompt)
            candidate_strategies.append("seed")
        candidates = candidates[:target]
        candidate_strategies = candidate_strategies[:target]

    else:
        raise ValueError(f"Unknown strategy_mode: {strategy_mode}")

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
        "strategy_mode": strategy_mode,
        "k": k,
        "strategies_used": [
            {"name": s.name, "description": s.description, "source": s.source}
            for s in strategies_used
        ],
        "discovery_outputs": discovery_outputs,
        "hard_examples_found_count": 0,
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
        sorted_candidates, sorted_scores, sorted_orig_indices, per_example_scores = run_pruning_round(
            adapter=adapter,
            candidates=candidates,
            trainset=trainset,
            n_examples=n_examples,
            collector=collector,
            rng=rng,
        )

        # Build sorted_pos_to_strategy from sorted_orig_indices
        sorted_pos_to_strategy = {
            pos: (candidate_strategies[orig_idx] if orig_idx < len(candidate_strategies) else "unknown")
            for pos, orig_idx in enumerate(sorted_orig_indices)
        }

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

        # Evaluate round's best on hold-out + write progress.json (common to all strategy modes)
        _round_best_text = sorted_candidates[0] if sorted_candidates else seed_prompt
        _holdout_score = _evaluate_holdout(adapter, _round_best_text, holdout_examples, collector)
        if _holdout_score > _running_best_holdout:
            _running_best_holdout = _holdout_score
            _running_best_holdout_prompt = _round_best_text
        collector.record_trajectory_point(
            iteration=round_num,
            holdout_score=_holdout_score,
            best_so_far=_running_best_holdout,
            prompt_length=len(_round_best_text),
        )
        _write_progress_json(
            benchmark=benchmark,
            seed=seed,
            method_name=method_name,
            model_tag=get_model_tag(settings),
            rollouts_used=collector.rollout_count,
            best_val_score=_running_best_holdout,
            iteration=round_num,
            wall_clock_seconds=time.time() - start_time,
        )

        # Refresh pass (expand variant): after R1, discover new strategies on hard examples
        # and inject new candidates into the pool before R2.
        _refresh_new_candidates: list[str] = []
        _refresh_new_strategy_labels: list[str] = []
        if round_num == 1 and refresh_mode == "expand" and strategy_mode == "inductive":
            refresh_failure_matrix = build_failure_matrix(per_example_scores, threshold=0.5)
            hard_ex_ids = collect_hard_examples(
                failure_matrix=refresh_failure_matrix,
                n_candidates=len(sorted_candidates),
                threshold=hard_example_threshold,
            )
            hard_examples_count = len(hard_ex_ids)
            method_specific["hard_examples_found_count"] = hard_examples_count

            if hard_examples_count > 0:
                hard_examples_objects = [trainset[i] for i in hard_ex_ids if i < len(trainset)]
                new_strategies, refresh_raw = discover_refresh_strategies(
                    reflection_lm=tracked_reflection,
                    benchmark=benchmark,
                    task_description=task_description,
                    output_shape=output_shape,
                    hard_examples=hard_examples_objects,
                    existing_strategies=strategies_used,
                    k_new=2,
                )

                method_specific.setdefault("discovery_outputs", []).append({
                    "pass": "refresh",
                    "k_requested": 2,
                    "raw_llm_output": refresh_raw,
                    "fallback_used": False,
                    "skills": [
                        {
                            "name": s.name,
                            "description": s.description,
                            "failure_pattern": s.failure_pattern,
                            "technique": s.technique,
                        }
                        for s in new_strategies
                    ],
                })

                for new_strategy in new_strategies:
                    specialized = generate_specialized_prompt(
                        reflection_lm=tracked_reflection,
                        strategy=new_strategy,
                        task_description=task_description,
                        seed_prompt=seed_prompt,
                        examples=trainset,
                        n=4,
                        rng=rng,
                    )
                    for sp in specialized[:4]:
                        _refresh_new_candidates.append(sp)
                        _refresh_new_strategy_labels.append(new_strategy.name)

                console.print(
                    f"  Refresh: {hard_examples_count} hard examples found; "
                    f"added {len(_refresh_new_candidates)} new candidates from {len(new_strategies)} new skills"
                )
            else:
                console.print(f"  Refresh: 0 hard examples found; no new candidates added")

        # Between rounds: mutate survivors using failure info (skip after last round)
        if round_idx < len(_PRUNING_SCHEDULE) - 1:
            console.print(f"  Mutating {len(survivors)} survivors...")
            mutated: list[str] = []
            mutated_strategies: list[str] = []

            # Build failure matrix once per round for crosspollin mode
            round_failure_matrix = build_failure_matrix(per_example_scores, threshold=0.5) if mutation_mode == "crosspollin" else {}

            for survivor_pos, (prompt_text, score) in enumerate(zip(survivors, survivor_scores)):
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

                if mutation_mode == "crosspollin":
                    survivor_strategy = sorted_pos_to_strategy.get(survivor_pos, "unknown")

                    donor_result = find_donor(
                        survivor_idx=survivor_pos,
                        survivor_strategy=survivor_strategy,
                        failure_matrix=round_failure_matrix,
                        per_example_scores=per_example_scores,
                        strategies=sorted_pos_to_strategy,
                        threshold=0.5,
                    )

                    event: dict[str, Any] = {
                        "round": round_num,
                        "survivor_candidate_idx": survivor_pos,
                        "survivor_strategy": survivor_strategy,
                        "survivor_score": score,
                        "survivor_failed_examples": sorted(round_failure_matrix.get(survivor_pos, set())),
                        "donor_candidate_idx": None,
                        "donor_strategy": None,
                        "shared_failures_covered": 0,
                        "donor_score_on_failures": None,
                        "cross_strategy": False,
                        "no_donor_found": True,
                    }

                    if donor_result is not None:
                        donor_idx = donor_result["donor_candidate_idx"]
                        donor_prompt_text = sorted_candidates[donor_idx]
                        event.update(donor_result)
                        improved = mutate_prompt_with_context(
                            reflection_lm=tracked_reflection,
                            prompt=prompt_text,
                            score=score,
                            failures=failures,
                            survivor_strategy=survivor_strategy,
                            donor_strategy=donor_result["donor_strategy"],
                            donor_prompt=donor_prompt_text,
                        )
                    else:
                        improved = mutate_prompt_with_context(
                            reflection_lm=tracked_reflection,
                            prompt=prompt_text,
                            score=score,
                            failures=failures,
                            survivor_strategy=survivor_strategy,
                        )

                    method_specific.setdefault("cross_pollination_events", []).append(event)
                else:
                    improved = mutate_prompt(
                        reflection_lm=tracked_reflection,
                        prompt=prompt_text,
                        score=score,
                        failures=failures,
                    )

                mutated.append(improved)
                # Mutated survivors keep their strategy (prompt improved, not replaced)
                mutated_strategies.append(sorted_pos_to_strategy.get(survivor_pos, "unknown"))

            # Append refresh new candidates unchanged (unevaluated; R2 will eval them fresh)
            mutated.extend(_refresh_new_candidates)
            mutated_strategies.extend(_refresh_new_strategy_labels)

            candidates = mutated
            candidate_strategies = mutated_strategies
        else:
            candidates = survivors
            candidate_strategies = [
                sorted_pos_to_strategy.get(pos, "unknown") for pos in range(len(survivors))
            ]

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
        "method_name": method_name,
        "model": model_id(settings),
        "temperature": settings.gepa_temperature,
        "top_p": settings.gepa_top_p,
        "top_k": settings.gepa_top_k,
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
        method=method_name,
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
        method=method_name,
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
        method=method_name,
        model_tag=mtagval,
        test_outputs=test_outputs,
    )
    console.print(f"  Results saved to runs/{mtagval + '/' if mtagval else ''}{benchmark}/{method_name}/{seed}/")

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
    parser.add_argument(
        "--strategy-mode", choices=["personality", "prescribed8", "inductive"],
        default="personality",
        help="Strategy source: personality (baseline), prescribed8 (8 universal), or inductive (discovered)"
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Number of skills for inductive discovery (3, 5, or None for adaptive). Ignored for personality/prescribed8."
    )
    parser.add_argument(
        "--mutation-mode", choices=["blind", "crosspollin"], default="blind",
        help="Mutation approach: blind (current behavior) or crosspollin (strategy-aware with donor selection)"
    )
    parser.add_argument(
        "--refresh-mode", choices=["none", "expand", "replace"], default="none",
        help="Refresh pass after R1: none, expand (add new candidates), or replace (swap weakest)"
    )
    parser.add_argument(
        "--hard-example-threshold", type=float, default=0.7,
        help="Fraction of candidates that must fail on an example for it to be 'hard' (default: 0.7)"
    )
    args = parser.parse_args()

    run_slime_mold(
        benchmark=args.benchmark,
        seed=args.seed,
        subset=args.subset,
        max_metric_calls=args.max_metric_calls,
        strategy_mode=args.strategy_mode,
        k=args.k,
        mutation_mode=args.mutation_mode,
        refresh_mode=args.refresh_mode,
        hard_example_threshold=args.hard_example_threshold,
    )
