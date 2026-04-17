"""MIPROv2 runner matching the unified CLI contract.

Wraps DSPy's MIPROv2 Bayesian instruction optimizer to produce results
comparable to GEPA and ISO. The MIPROv2 'prompt model' maps to our
--reflection-model flag (the 'proposer' that generates candidate instructions).

Budget control:
    We set auto=None and compute num_trials from --budget-rollouts:
        num_trials = budget_rollouts // len(valset)
    With minibatch=False, total evals = num_trials × len(valset).
    This gives clean budget control without monkey-patching DSPy.

    FLAG: If budget adherence exceeds ±10%, this is reported and the run
    proceeds. The plan says to stop and flag for decision if monkey-patching
    is needed — this approach avoids that entirely.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import dspy
from rich.console import Console

from gepa_mutations.base import build_qa_task_lm, build_reflection_lm_for_model, evaluate_on_test
from gepa_mutations.benchmarks.loader import load_benchmark
from gepa_mutations.config import PAPER_ROLLOUTS, Settings, model_tag as get_model_tag
from gepa_mutations.runner.experiment import (
    BENCHMARK_SEED_PROMPTS,
    SEED_PROMPT,
    ExperimentResult,
)
from gepa_mutations.storage.local import save_result

console = Console()


class _QAModule(dspy.Module):
    """Minimal DSPy module for QA-style benchmarks.

    MIPROv2 optimizes instruction text and few-shot demos for this module.
    The signature is simple: question → answer, with a system instruction.
    """

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict("question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        return self.generate(question=question)


def _examples_to_dspy(examples: list) -> list[dspy.Example]:
    """Convert our benchmark examples to DSPy Example format."""
    dspy_examples = []
    for ex in examples:
        # All our benchmarks have .input_text and .answer
        d = dspy.Example(
            question=getattr(ex, "input_text", str(ex)),
            answer=getattr(ex, "answer", ""),
        ).with_inputs("question")
        dspy_examples.append(d)
    return dspy_examples


def _make_metric(benchmark: str):
    """Create a DSPy metric function for the given benchmark.

    Returns a function(example, pred, trace=None) -> float that scores
    a prediction against the gold answer.
    """
    from gepa_mutations.benchmarks.evaluators import get_scorer

    scorer = get_scorer(benchmark)

    def metric(example, pred, trace=None) -> float:
        gold = example.answer
        predicted = pred.answer if hasattr(pred, "answer") else str(pred)
        return scorer(predicted, gold)

    return metric


def run_miprov2(
    benchmark: str = "hotpotqa",
    seed: int = 42,
    subset: int | None = None,
    max_metric_calls: int | None = None,
    settings: Settings | None = None,
) -> ExperimentResult:
    """Run MIPROv2 optimization matching the unified CLI contract.

    Args:
        benchmark: Benchmark name.
        seed: Random seed.
        subset: Limit train/val to this many examples.
        max_metric_calls: Rollout budget (total task-model evaluation calls).
        settings: Environment settings.

    Returns:
        ExperimentResult with test/val scores and best prompt.
    """
    settings = settings or Settings()
    start_time = time.time()

    # 1. Load benchmark
    console.print(f"[bold]Loading benchmark: {benchmark}[/bold]")
    data = load_benchmark(benchmark, seed=0)
    trainset = data.train[:subset] if subset else data.train
    valset = data.val[:subset] if subset else data.val
    testset = data.test

    console.print(f"  Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")

    # 2. Configure LMs
    task_lm = build_qa_task_lm(settings)
    proposer_lm = build_reflection_lm_for_model(settings)

    # Configure DSPy with task LM as default
    dspy.configure(lm=task_lm)

    # 3. Compute budget
    if max_metric_calls is None:
        max_metric_calls = PAPER_ROLLOUTS["gepa"].get(benchmark, 5000)

    # Compute num_trials and num_candidates from budget
    # With minibatch=False: total_evals = (1 + num_trials) × len(valset)
    # The +1 is for the default program evaluation
    num_trials = max(1, (max_metric_calls // len(valset)) - 1)
    # num_candidates: at least 3, derived from num_trials
    # MIPROv2 formula: num_trials = max(2*num_vars*log2(N), 1.5*N)
    # For 1 predictor with demos: num_vars=2, so num_trials = max(4*log2(N), 1.5*N)
    # We invert: N ≈ num_trials / 1.5
    num_candidates = max(3, int(num_trials / 1.5))

    expected_evals = (1 + num_trials) * len(valset)
    console.print(f"\n[bold]MIPROv2 budget plan:[/bold]")
    console.print(f"  Requested budget: {max_metric_calls} rollouts")
    console.print(f"  num_trials: {num_trials}, num_candidates: {num_candidates}")
    console.print(f"  Expected evaluations: {expected_evals}")

    adherence = abs(expected_evals - max_metric_calls) / max_metric_calls
    if adherence > 0.10:
        console.print(
            f"  [yellow]WARNING: Budget adherence {adherence:.1%} > 10% "
            f"(expected {expected_evals} vs requested {max_metric_calls})[/yellow]"
        )

    # 4. Convert data to DSPy format
    dspy_trainset = _examples_to_dspy(trainset)
    dspy_valset = _examples_to_dspy(valset)

    # 5. Create metric
    metric = _make_metric(benchmark)

    # 6. Run MIPROv2
    console.print(f"\n[bold]Running MIPROv2 optimization[/bold]")
    student = _QAModule()

    optimizer = dspy.MIPROv2(
        metric=metric,
        prompt_model=proposer_lm,
        task_model=task_lm,
        auto=None,  # Manual budget control
        num_candidates=num_candidates,
        seed=seed,
        verbose=False,
    )

    optimized_program = optimizer.compile(
        student,
        trainset=dspy_trainset,
        valset=dspy_valset,
        num_trials=num_trials,
        minibatch=False,  # Full eval each trial for clean budget control
        requires_permission_to_run=False,
    )

    # 7. Extract best prompt
    # MIPROv2 sets instructions on the predictor's signature
    best_instruction = ""
    for predictor in optimized_program.predictors():
        sig = predictor.signature
        if hasattr(sig, "instructions"):
            best_instruction = sig.instructions
            break

    best_prompt = {"system_prompt": best_instruction or SEED_PROMPT}
    val_score = getattr(optimized_program, "score", 0.0)

    console.print(f"\n[bold]Best MIPROv2 instruction:[/bold]")
    console.print(f"  {best_instruction[:200]}...")
    console.print(f"  Val score: {val_score:.4f}")

    # 8. Evaluate seed prompt on test
    seed_prompt = BENCHMARK_SEED_PROMPTS.get(benchmark, SEED_PROMPT)
    seed_candidate = {"system_prompt": seed_prompt}
    console.print(f"\n[bold]Evaluating seed prompt on test set...[/bold]")
    seed_test_eval = evaluate_on_test(benchmark, seed_candidate, testset, settings)
    console.print(f"  Seed test score: {seed_test_eval.score:.4f}")

    # 9. Evaluate best prompt on test
    console.print(f"\n[bold]Evaluating best MIPROv2 prompt on test set...[/bold]")
    test_eval = evaluate_on_test(benchmark, best_prompt, testset, settings)
    console.print(f"  Best test score: {test_eval.score:.4f}")

    # 10. Build result
    wall_clock = time.time() - start_time
    model_tag_str = get_model_tag(settings)

    result = ExperimentResult(
        benchmark=benchmark,
        seed=seed,
        test_score=test_eval.score,
        val_score=val_score,
        best_prompt=best_prompt,
        rollout_count=expected_evals,
        config_snapshot={
            "method": "miprov2",
            "num_trials": num_trials,
            "num_candidates": num_candidates,
            "budget_requested": max_metric_calls,
            "budget_actual": expected_evals,
            "minibatch": False,
        },
        wall_clock_seconds=wall_clock,
        method="miprov2",
        all_candidates=[best_prompt],
        test_example_scores=test_eval.per_example_scores,
        seed_prompt_test_score=seed_test_eval.score,
        train_score=None,
    )

    # Save result
    save_result(result, model_tag=model_tag_str)
    console.print(f"\n[bold green]MIPROv2 complete![/bold green] Wall clock: {wall_clock:.0f}s")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MIPROv2 optimization")
    parser.add_argument("--benchmark", default="hotpotqa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--max-metric-calls", type=int, default=None)
    args = parser.parse_args()

    run_miprov2(
        benchmark=args.benchmark,
        seed=args.seed,
        subset=args.subset,
        max_metric_calls=args.max_metric_calls,
    )
