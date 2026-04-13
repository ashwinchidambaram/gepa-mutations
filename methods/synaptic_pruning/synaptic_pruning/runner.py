"""Runner for Synaptic Pruning Driven Optimization (SPDO).

Algorithm:
1. Generate 3 over-specified ~2000-word prompts via reflection LM.
2. Evaluate all 3 on a val subset (first 20 examples), pick best as baseline.
3. Parse best into sections.
4. Ablation loop: for each section, remove it and record score delta.
5. Mark sections where removal causes score drop < 0.01 as prunable.
6. Remove all prunable sections, re-evaluate to check for interaction effects.
7. If combined removal dropped score too much, add back prunable sections
   one at a time (sorted by least impact) until score recovers.
8. Strengthen load-bearing sections (score drop > 0.05) via reflection LM.
9. Evaluate best prompt on full valset.
10. evaluate_on_test(), save_result().
"""

from __future__ import annotations

import argparse
import random
import time
import traceback
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
from gepa_mutations.metrics.standalone_eval import evaluate_prompt
from gepa_mutations.metrics.tracked_lm import TrackedLM
from gepa_mutations.runner.experiment import (
    BENCHMARK_SEED_PROMPTS,
    SEED_PROMPT,
    ExperimentResult,
)
from gepa_mutations.storage.local import save_result

from synaptic_pruning.pruner import ablate_section, parse_sections, run_ablation

console = Console()

# Number of over-specified initial prompts to generate and evaluate
_N_INITIAL_PROMPTS = 3
# Number of val examples to use for section-level ablation evals
# Increased from 20 → 40: 20 examples gives ±5pp resolution (1 example = 5pp),
# making ablation decisions effectively noise on hard benchmarks like hover/pupa.
_ABLATION_VAL_SUBSET = 40
# Score drop threshold: sections above this are load-bearing
_LOAD_BEARING_THRESHOLD = 0.05
# Score drop threshold: sections below this are prunable
_PRUNABLE_THRESHOLD = 0.01
# If combined pruning drops score more than this, start adding back sections
_INTERACTION_THRESHOLD = 0.03
# Number of labeled (input, label) pairs to show in initial prompt generation.
# Without labels, the LLM cannot learn domain-specific scoring logic
# (e.g. hover's binary key-value encoding, pupa's PII redaction criteria).
_N_LABELED_EXAMPLES = 4


def _generate_initial_prompts(
    tracked_reflection: TrackedLM,
    seed_prompt: str,
    labeled_examples: list[tuple[str, str]],
    n: int = 3,
) -> list[str]:
    """Generate n over-specified initial prompts via the reflection LM.

    Uses labeled (input, expected_output) pairs so the LLM can learn the
    task's scoring logic from examples.  Without labels, benchmarks with
    non-obvious domain-specific logic (hover's binary key-value encoding,
    pupa's PII redaction criteria) produce systematically wrong prompts
    that no amount of ablation can recover from.
    """
    examples_str = "\n\n".join(
        f"Input:\n{inp}\n\nExpected output: {label}"
        for inp, label in labeled_examples
    )
    generation_prompt = (
        f"Write a detailed system prompt (~500 words) for this task.\n"
        f"Cover key reasoning strategies, format requirements, and common pitfalls.\n\n"
        f"Task description: {seed_prompt}\n\n"
        f"Labeled examples (study these carefully to understand correct behavior):\n"
        f"{examples_str}\n\n"
        f"Your system prompt must produce the correct output format and reasoning "
        f"strategy demonstrated by the examples above.\n\n"
        f"System prompt:"
    )

    prompts = []
    for i in range(n):
        try:
            response = tracked_reflection(generation_prompt)
            if response and response.strip():
                generated = response.strip()
                if len(generated) < 200:
                    console.print(
                        f"  [yellow]Warning: initial prompt {i+1} too short "
                        f"({len(generated)} chars < 200); discarding[/yellow]"
                    )
                else:
                    prompts.append(generated)
            else:
                console.print(f"  [yellow]Warning: empty response for initial prompt {i+1}[/yellow]")
        except Exception as e:
            console.print(
                f"  [yellow]Warning: failed to generate initial prompt {i+1}: {e}[/yellow]\n"
                f"  [yellow]Traceback:\n{traceback.format_exc()}[/yellow]"
            )

    if not prompts:
        raise RuntimeError(
            "All initial prompt generations failed — check logs above. "
            "Responses were either empty, too short (<200 chars), or raised exceptions."
        )

    return prompts


def _get_labeled_examples(dataset: list, n: int = 4) -> list[tuple[str, str]]:
    """Extract (input, label) pairs from a dataset for few-shot generation.

    Uses the first n examples so generation is deterministic across seeds.
    For ifbench (constraint-checking tasks), use constraints as the label.
    """
    pairs = []
    for ex in dataset[:n]:
        inp = getattr(ex, "input", None) or getattr(ex, "question", None) or str(ex)

        # For constraint-based tasks (ifbench), use constraints as the label
        constraints = getattr(ex, "constraints", None)
        if constraints:
            label = " | ".join(constraints)
        else:
            label = getattr(ex, "answer", None) or getattr(ex, "label", None) or getattr(ex, "output", None) or "?"

        # Truncate very long inputs/labels to avoid bloating the generation prompt
        inp_str = str(inp)[:400]
        label_str = str(label)[:200]
        pairs.append((inp_str, label_str))
    return pairs


def _strengthen_section(
    tracked_reflection: TrackedLM,
    section: str,
    task_description: str,
) -> str:
    """Ask the reflection LM to make a load-bearing section stronger."""
    strengthen_prompt = (
        f"The following section of a system prompt is critical for task performance "
        f"(removing it significantly degrades results). Please rewrite it to be clearer, "
        f"more specific, and more actionable. Keep the same purpose but make it stronger.\n\n"
        f"Task context: {task_description}\n\n"
        f"Section to strengthen:\n{section}\n\n"
        f"Strengthened version:"
    )
    try:
        response = tracked_reflection(strengthen_prompt)
        if response and response.strip():
            return response.strip()
    except Exception as e:
        console.print(f"  [yellow]Warning: failed to strengthen section: {e}[/yellow]")
    return section


def run_synaptic_pruning(
    benchmark: str,
    seed: int,
    subset: int | None = None,
    max_metric_calls: int | None = None,
    settings: Settings | None = None,
) -> ExperimentResult:
    """Run the Synaptic Pruning Driven Optimization (SPDO) experiment.

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
    mtagval = get_model_tag(settings)
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
    seed_candidate = {"system_prompt": seed_prompt}

    # =========================================================================
    # 4. Budget
    # =========================================================================
    budget = max_metric_calls or PAPER_ROLLOUTS["gepa"].get(benchmark, 5000)

    console.print(f"[bold]Running SPDO (Synaptic Pruning)[/bold]")
    console.print(f"  Benchmark: {benchmark}, Seed: {seed}")
    console.print(f"  Rollout budget: {budget}")

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
    collector.record_val_score(iteration=0, score=seed_prompt_val_score)

    # =========================================================================
    # 5. Generate 3 over-specified initial prompts (with labeled examples)
    # =========================================================================
    console.print("\n[bold]Step 1: Generating over-specified initial prompts...[/bold]")
    labeled_examples = _get_labeled_examples(trainset, n=_N_LABELED_EXAMPLES)
    console.print(f"  Using {len(labeled_examples)} labeled examples for generation")
    initial_prompts = _generate_initial_prompts(
        tracked_reflection, seed_prompt, labeled_examples, n=_N_INITIAL_PROMPTS
    )
    console.print(f"  Generated {len(initial_prompts)} initial prompts")

    # =========================================================================
    # 6. Evaluate all initial prompts on val subset (first 20 examples)
    # =========================================================================
    console.print("\n[bold]Step 2: Evaluating initial prompts on val subset...[/bold]")
    ablation_val_n = min(_ABLATION_VAL_SUBSET, len(valset), 40)
    ablation_val_indices = list(range(ablation_val_n))

    best_initial_score = -1.0
    best_initial_prompt = initial_prompts[0]

    for i, prompt_text in enumerate(initial_prompts):
        if collector.rollout_count >= budget:
            console.print("  [yellow]Budget exhausted during initial evaluation[/yellow]")
            break
        candidate = {"system_prompt": prompt_text}
        score, _ = evaluate_prompt(adapter, valset, candidate, collector, indices=ablation_val_indices)
        console.print(f"  Initial prompt {i+1}: val score = {score:.4f}")
        if score > best_initial_score:
            best_initial_score = score
            best_initial_prompt = prompt_text

    console.print(f"  Best initial prompt score: {best_initial_score:.4f}")
    collector.record_val_score(iteration=1, score=best_initial_score)
    original_prompt_length = len(best_initial_prompt)

    # =========================================================================
    # 7. Parse best initial prompt into sections
    # =========================================================================
    console.print("\n[bold]Step 3: Parsing prompt into sections...[/bold]")
    sections = parse_sections(best_initial_prompt)
    console.print(f"  Found {len(sections)} sections")

    # =========================================================================
    # 8. Ablation loop: evaluate each section removal
    # =========================================================================
    console.print("\n[bold]Step 4: Running section ablation...[/bold]")
    base_candidate = {"system_prompt": best_initial_prompt}

    load_bearing_indices, prunable_indices, ablation_scores = run_ablation(
        adapter=adapter,
        dataset=valset,
        sections=sections,
        candidate_template=base_candidate,
        collector=collector,
        eval_indices=ablation_val_indices,
        budget=budget,
        baseline_floor=0.4,  # skip ablation if initial prompt score is too low to trust
    )

    console.print(f"  Sections tested: {len(sections)}")
    console.print(f"  Load-bearing sections: {len(load_bearing_indices)}")
    console.print(f"  Prunable sections: {len(prunable_indices)}")
    collector.record_val_score(iteration=2, score=best_initial_score)

    # =========================================================================
    # 9. Remove all prunable sections at once and check interaction effects
    # =========================================================================
    console.print("\n[bold]Step 5: Checking interaction effects of combined pruning...[/bold]")

    # Build the pruned prompt (keep non-prunable sections)
    kept_indices = [i for i in range(len(sections)) if i not in prunable_indices]
    pruned_sections = [sections[i] for i in kept_indices]
    pruned_prompt = "\n\n".join(pruned_sections) if pruned_sections else best_initial_prompt

    if prunable_indices and pruned_prompt.strip():
        if collector.rollout_count >= budget:
            console.print("  [yellow]Budget exhausted; skipping interaction check[/yellow]")
        else:
            pruned_candidate = {"system_prompt": pruned_prompt}
            combined_score, _ = evaluate_prompt(
                adapter, valset, pruned_candidate, collector, indices=ablation_val_indices
            )
            console.print(f"  Combined pruning score: {combined_score:.4f} (baseline: {best_initial_score:.4f})")
            _combined_best = max(best_initial_score, combined_score)
            collector.record_val_score(iteration=3, score=_combined_best)
            combined_drop = best_initial_score - combined_score

            # If combined removal dropped score too much, add back sections one at a time
            # sorted by least impact (smallest score_delta, i.e., least important)
            if combined_drop > _INTERACTION_THRESHOLD and prunable_indices:
                console.print(f"  [yellow]Combined drop {combined_drop:.4f} exceeds threshold {_INTERACTION_THRESHOLD}; recovering...[/yellow]")

                # Sort prunable sections by ascending score_delta (add back least-impactful first)
                prunable_with_delta = [
                    (idx, next(s["score_delta"] for s in ablation_scores if s["section_idx"] == idx))
                    for idx in prunable_indices
                ]
                prunable_with_delta.sort(key=lambda x: x[1])  # ascending: least impact first

                current_sections = list(pruned_sections)
                current_score = combined_score

                for prunable_idx, delta in prunable_with_delta:
                    if current_score >= best_initial_score - _INTERACTION_THRESHOLD:
                        break  # Score has recovered enough
                    if collector.rollout_count >= budget:
                        console.print("    [yellow]Budget exhausted during recovery[/yellow]")
                        break

                    # Add this section back
                    current_sections.append(sections[prunable_idx])
                    recovery_prompt = "\n\n".join(current_sections)
                    recovery_candidate = {"system_prompt": recovery_prompt}
                    recovery_score, _ = evaluate_prompt(
                        adapter, valset, recovery_candidate, collector, indices=ablation_val_indices
                    )
                    console.print(f"    Added back section {prunable_idx}: score {recovery_score:.4f}")

                    if recovery_score > current_score:
                        current_score = recovery_score
                    else:
                        # Adding it back didn't help; remove it again
                        current_sections.pop()

                pruned_sections = current_sections
                pruned_prompt = "\n\n".join(pruned_sections)
                console.print(f"  Score after recovery: {current_score:.4f}")
                collector.record_val_score(iteration=4, score=max(best_initial_score, current_score))
    else:
        console.print("  No prunable sections or empty result; skipping combined pruning check.")

    # Track actual sections pruned/kept
    sections_pruned = len(sections) - len(pruned_sections)
    sections_kept = len(pruned_sections)

    # =========================================================================
    # 10. Strengthen load-bearing sections via reflection LM
    # =========================================================================
    console.print("\n[bold]Step 6: Strengthening load-bearing sections...[/bold]")
    strengthened_sections = list(pruned_sections)

    for lb_idx in load_bearing_indices:
        if collector.rollout_count >= budget:
            console.print("  [yellow]Budget exhausted; skipping remaining strengthening[/yellow]")
            break
        if lb_idx >= len(sections):
            continue
        original_section = sections[lb_idx]
        # Only strengthen if this section is still in our pruned set
        if original_section not in strengthened_sections:
            continue

        strengthened = _strengthen_section(tracked_reflection, original_section, seed_prompt)
        # Replace the section in our list
        pos = strengthened_sections.index(original_section)
        strengthened_sections[pos] = strengthened
        console.print(f"  Strengthened section {lb_idx} (was {len(original_section)} chars, now {len(strengthened)} chars)")

    best_prompt_text = "\n\n".join(strengthened_sections) if strengthened_sections else best_initial_prompt
    collector.record_val_score(iteration=5, score=best_initial_score)

    # =========================================================================
    # 11. Evaluate best prompt on full valset
    # =========================================================================
    console.print("\n[bold]Step 7: Evaluating final prompt on full valset...[/bold]")
    best_candidate = {"system_prompt": best_prompt_text}
    best_val_score, _ = evaluate_prompt(adapter, valset, best_candidate, collector)
    console.print(f"  Final val score: {best_val_score:.4f}")
    collector.record_val_score(iteration=6, score=best_val_score)

    best_prompt_dict = {"system_prompt": best_prompt_text}
    final_prompt_length = len(best_prompt_text)

    # =========================================================================
    # 12. Populate method-specific metrics
    # =========================================================================
    collector.method_specific.update({
        "sections_tested": len(sections),
        "sections_pruned": sections_pruned,
        "sections_kept": sections_kept,
        "per_section_ablation_scores": ablation_scores,
        "original_prompt_length": original_prompt_length,
        "final_prompt_length": final_prompt_length,
        "n_initial_prompts": len(initial_prompts),
        "best_initial_score": best_initial_score,
    })

    # =========================================================================
    # 13. Evaluate on test set
    # =========================================================================
    console.print(f"\n[bold]Evaluating on test set ({len(testset)} examples)...[/bold]")
    test_eval = evaluate_on_test(benchmark, best_prompt_dict, testset, settings)
    console.print(f"  Test score: {test_eval.score:.4f} ({test_eval.score * 100:.2f}%)")

    # =========================================================================
    # 14. Save results
    # =========================================================================
    metrics_data = collector.finalize(
        test_score=test_eval.score,
        best_prompt=best_prompt_dict,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
    )

    config_snap = {
        "benchmark": benchmark,
        "seed": seed,
        "subset": subset,
        "method": "synaptic_pruning",
        "model": model_id(settings),
        "temperature": settings.gepa_temperature,
        "top_p": settings.gepa_top_p,
        "top_k": settings.gepa_top_k,
        "rollout_budget": budget,
        "n_initial_prompts": _N_INITIAL_PROMPTS,
        "ablation_val_subset": _ABLATION_VAL_SUBSET,
        "load_bearing_threshold": _LOAD_BEARING_THRESHOLD,
        "prunable_threshold": _PRUNABLE_THRESHOLD,
        "interaction_threshold": _INTERACTION_THRESHOLD,
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
        method="synaptic_pruning",
        metrics=metrics_data,
        test_example_scores=test_eval.example_scores,
        test_example_ids=test_eval.example_ids,
        seed_prompt_test_score=seed_prompt_test_score,
        seed_prompt_val_score=seed_prompt_val_score,
    )

    save_result(
        benchmark=benchmark,
        seed=seed,
        result_data=exp_result.to_dict(),
        config_data=exp_result.config_snapshot,
        metrics_data=metrics_data,
        method="synaptic_pruning",
        model_tag=mtagval,
    )
    console.print(f"  Results saved to runs/{mtagval + '/' if mtagval else ''}{benchmark}/synaptic_pruning/{seed}/")

    return exp_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPDO (Synaptic Pruning) optimization")
    parser.add_argument("--benchmark", "-b", required=True, help="Benchmark name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subset", "-s", type=int, default=None, help="Use only N train/val examples")
    parser.add_argument("--max-metric-calls", "-m", type=int, default=None, help="Rollout budget override")
    args = parser.parse_args()
    run_synaptic_pruning(args.benchmark, args.seed, args.subset, args.max_metric_calls)
