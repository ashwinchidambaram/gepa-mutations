"""Run STAMP-V2 on real benchmarks (hover, livebench, etc.) on the raycluster.

Usage:
    uv run python scripts/raycluster/run_stamp_v2_benchmark.py --benchmark hover --seeds 42 123 456 789 1024
    uv run python scripts/raycluster/run_stamp_v2_benchmark.py --benchmark hover --seeds 42

Results saved to: runs/qwen3.5-27b/{benchmark}/stamp_v2/{seed}/result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import (  # noqa: E402
    BENCHMARKS,
    BENCHMARK_MAX_TOKENS,
    INFERENCE_BASE_URL,
    INFRA_TAG,
    MAX_TOKENS_QA,
    MAX_TOKENS_REFLECT,
    MODEL_FULL_NAME,
    MODEL_NAME,
    MODEL_TAG,
    SEEDS,
    TEMPERATURE,
)

from gepa_mutations.benchmarks.loader import load_benchmark  # noqa: E402
from gepa_mutations.benchmarks.evaluators import (  # noqa: E402
    _check_ifbench_by_id,
    _check_ifbench_constraint,
)

from stamp_v2.config import StampConfig  # noqa: E402
from stamp_v2.data.schemas import (  # noqa: E402
    BatchEvalResult,
    BenchmarkSplit,
    EvalResult,
    TaskExample,
)
from stamp_v2.models.openai_compatible_client import OpenAICompatibleClient  # noqa: E402
from stamp_v2.optimization.stamp_optimizer import STAMPV2Optimizer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Seed prompts (same as ISO/GEPA)
SEED_PROMPT = "You are a helpful assistant."
BENCHMARK_SEED_PROMPTS = {
    "aime": (
        "You are a helpful assistant. You are given a question and you need to answer "
        "it. The answer should be given at the end of your response in exactly the "
        "format '### <final answer>'"
    ),
}

# GEPA paper rollout budgets (for reference)
PAPER_ROLLOUTS = {
    "hover": 2426,
    "livebench": 1839,
    "ifbench": 3593,
    "pupa": 3936,
    "hotpotqa": 6871,
    "aime": 7051,
}


def get_seed_prompt(benchmark: str) -> str:
    return BENCHMARK_SEED_PROMPTS.get(benchmark, SEED_PROMPT)


class BenchmarkEvaluator:
    """Evaluator that wraps the benchmark-specific scoring logic.

    Adapts the ISO metric pattern to STAMP-V2's EvalResult interface.
    """

    def __init__(self, benchmark_name: str):
        self.benchmark_name = benchmark_name

    def evaluate(self, example: TaskExample, output: str) -> EvalResult:
        """Score a model output against the benchmark gold answer."""
        response = output
        answer = (example.expected_output or "").strip().lower()
        score = 0.0
        feedback = ""
        failure_modes = []

        if self.benchmark_name == "hotpotqa":
            if re.search(r'\b' + re.escape(answer) + r'\b', response.lower()):
                score = 1.0
                feedback = "Correct answer found."
            else:
                feedback = f"Expected '{answer}' not found."
                failure_modes = ["wrong_answer"]

        elif self.benchmark_name == "hover":
            resp_lower = response.lower()
            not_supported_indicators = [
                "not_supported", "not supported", "not enough", "insufficient",
                "does not support", "doesn't support", "cannot be verified",
                "no evidence", "contradicts", "refuted", "false", "incorrect",
            ]
            supported_indicators = ["supported", "verified", "confirmed", "true", "correct"]
            model_says_not_supported = any(ind in resp_lower for ind in not_supported_indicators)
            model_says_supported = (
                any(ind in resp_lower for ind in supported_indicators)
                and not model_says_not_supported
            )

            if answer in ("not_supported", "0"):
                score = 1.0 if model_says_not_supported else 0.0
            else:
                score = 1.0 if model_says_supported else 0.0

            if score == 0.0:
                failure_modes = ["wrong_classification"]
            feedback = f"Gold={answer}, verdict={'NOT_SUPPORTED' if model_says_not_supported else 'SUPPORTED' if model_says_supported else 'UNCLEAR'}"

        elif self.benchmark_name == "ifbench":
            instruction_ids = getattr(example, "_instruction_ids", [])
            kwargs_list = getattr(example, "_kwargs_list", [])
            constraints = getattr(example, "_constraints", [])

            if instruction_ids:
                satisfied = sum(
                    1 for i, iid in enumerate(instruction_ids)
                    if _check_ifbench_by_id(iid, kwargs_list[i] if i < len(kwargs_list) else None, response)
                )
                total = len(instruction_ids)
                score = satisfied / total if total > 0 else 0.0
                if score < 1.0:
                    failure_modes = ["constraint_violation"]
                feedback = f"Satisfied {satisfied}/{total} constraints."
            elif constraints:
                satisfied = sum(1 for c in constraints if _check_ifbench_constraint(c, response))
                total = len(constraints)
                score = satisfied / total if total > 0 else 0.0
                if score < 1.0:
                    failure_modes = ["constraint_violation"]
                feedback = f"Satisfied {satisfied}/{total} constraints."
            else:
                score = 0.0
                feedback = "No constraints found."

        elif self.benchmark_name == "pupa":
            pii_units = getattr(example, "_pii_units", [])
            if not pii_units:
                score = 1.0
                feedback = "No PII to protect."
            else:
                leaked = sum(1 for pii in pii_units if pii.lower() in response.lower())
                score = 1.0 - (leaked / len(pii_units))
                if leaked > 0:
                    failure_modes = ["pii_leaked"]
                feedback = f"Leaked {leaked}/{len(pii_units)} PII units."

        elif self.benchmark_name == "livebench":
            norm_answer = answer.strip().replace(",", "").replace(" ", "")
            norm_response = response.strip().lower().replace(",", "").replace(" ", "")
            if norm_answer == norm_response or norm_answer in norm_response:
                score = 1.0
            else:
                boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
                if boxed and boxed[-1].strip().lower().replace(",", "").replace(" ", "") == norm_answer:
                    score = 1.0
            if score == 0.0:
                failure_modes = ["wrong_answer"]
            feedback = f"Expected '{answer}', match={score > 0}."

        elif self.benchmark_name == "aime":
            nums = re.findall(r'\b(\d+)\b', response)
            if nums:
                try:
                    score = 1.0 if int(nums[-1]) == int(answer) else 0.0
                except ValueError:
                    score = 0.0
            if score == 0.0:
                failure_modes = ["wrong_answer"]
            feedback = f"Expected {answer}, extracted {nums[-1] if nums else 'nothing'}."

        else:
            score = 1.0 if answer in response.lower() else 0.0
            if score == 0.0:
                failure_modes = ["wrong_answer"]
            feedback = f"Generic check for '{answer}'."

        return EvalResult(
            example_id=example.id,
            score=score,
            passed=score >= 1.0,
            feedback=feedback,
            failure_modes=failure_modes,
        )


def _convert_dspy_examples(dspy_examples, benchmark_name: str) -> list[TaskExample]:
    """Convert dspy.Example objects to STAMP-V2 TaskExample format."""
    task_examples = []
    for i, ex in enumerate(dspy_examples):
        # Store benchmark-specific metadata as private attrs
        task_ex = TaskExample(
            id=f"{benchmark_name}_{i:04d}",
            input_text=str(ex.input),
            expected_output=str(getattr(ex, "answer", "")),
        )
        # Carry over IFBench-specific attributes
        if hasattr(ex, "instruction_ids"):
            task_ex._instruction_ids = ex.instruction_ids  # type: ignore
            task_ex._kwargs_list = getattr(ex, "kwargs_list", [])  # type: ignore
            task_ex._constraints = getattr(ex, "constraints", [])  # type: ignore
        # Carry over PUPA-specific attributes
        if hasattr(ex, "pii_units"):
            task_ex._pii_units = ex.pii_units  # type: ignore
        task_examples.append(task_ex)
    return task_examples


def run_single(
    benchmark_name: str, seed: int, runs_dir: Path, parallel_workers: int = 1
) -> None:
    """Run STAMP-V2 optimization for one benchmark/seed."""
    method_name = "stamp_v2" if parallel_workers <= 1 else "stamp_v2_par"
    result_path = runs_dir / benchmark_name / method_name / str(seed) / "result.json"

    if result_path.exists():
        logger.info(f"Skipping {benchmark_name}/stamp_v2/seed={seed} (already exists)")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"STAMP-V2: {benchmark_name} / seed={seed}")
    logger.info(f"{'='*60}")

    t0 = time.time()

    # Load benchmark data
    data = load_benchmark(benchmark_name, seed=0)
    trainset = _convert_dspy_examples(data.train, benchmark_name)
    valset = _convert_dspy_examples(data.val, benchmark_name)
    testset = _convert_dspy_examples(data.test, benchmark_name)
    logger.info(f"  Data: {len(trainset)} train, {len(valset)} val, {len(testset)} test")

    # Build split for STAMP-V2 (use first 16 train as diagnostic, rest of train as validation)
    diagnostic_size = min(16, len(trainset))
    validation_size = min(64, len(valset))

    split = BenchmarkSplit(
        diagnostic=trainset[:diagnostic_size],
        validation=valset[:validation_size],
        holdout=testset,
    )

    # Build config
    seed_prompt = get_seed_prompt(benchmark_name)
    max_tokens = BENCHMARK_MAX_TOKENS.get(benchmark_name, MAX_TOKENS_QA)

    config = StampConfig(
        experiment_name=f"{method_name}_{benchmark_name}_{seed}",
        random_seed=seed,
        rollout_budget_candidate_evals=200,
        max_candidate_evals_per_round=3,
        diagnostic_batch_size=diagnostic_size,
        validation_batch_size=validation_size,
        parallel_workers=parallel_workers,
        num_bundles_per_reflection=2,
        seed_prompt=[seed_prompt],
        # Screening thresholds
        min_delta_score=0.005,
        bundle_utility_threshold=0.0001,
        diagnostic_min_examples_improved=1,
        diagnostic_max_examples_regressed=4,
        final_delta=0.02,
    )

    # Build model clients
    task_model = OpenAICompatibleClient(
        model_name=MODEL_NAME,
        base_url=INFERENCE_BASE_URL,
        temperature=0.6,
        max_tokens=max_tokens,
        timeout=120.0,
    )
    reflector_model = OpenAICompatibleClient(
        model_name=MODEL_NAME,
        base_url=INFERENCE_BASE_URL,
        temperature=0.7,
        max_tokens=2048,
        timeout=120.0,
    )

    # Build evaluator
    evaluator = BenchmarkEvaluator(benchmark_name)

    # Run dir for JSONL logs
    run_dir = result_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run optimization
    optimizer = STAMPV2Optimizer(
        task_model=task_model,
        reflector_model=reflector_model,
        evaluator=evaluator,
        config=config,
        output_dir=run_dir,
    )

    try:
        result = optimizer.optimize(split)
        best_prompt = result.best_raw.render() if result.best_raw else seed_prompt
        wall_clock = time.time() - t0
        termination = "budget_exhausted"
    except Exception as e:
        logger.error(f"  Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        best_prompt = seed_prompt
        wall_clock = time.time() - t0
        termination = f"error: {str(e)[:200]}"
        result = None

    # Evaluate best prompt on test set
    logger.info(f"  Evaluating best prompt on test set ({len(testset)} examples)...")
    test_scores = []
    for example in testset:
        try:
            response = task_model.generate_chat([
                {"role": "system", "content": best_prompt},
                {"role": "user", "content": example.input_text},
            ])
            eval_result = evaluator.evaluate(example, response)
            test_scores.append(eval_result.score)
        except Exception:
            test_scores.append(0.0)

    test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
    logger.info(f"  Test score: {test_score:.4f}")
    logger.info(f"  Wall clock: {wall_clock:.1f}s ({wall_clock/60:.1f}m)")

    # Build result dict (same format as ISO/GEPA)
    git_sha = ""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        pass

    result_dict = {
        "metadata": {
            "model_tag": MODEL_TAG,
            "model_name": MODEL_FULL_NAME,
            "infra": INFRA_TAG,
            "inference_endpoint": INFERENCE_BASE_URL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha,
            "seed": seed,
            "benchmark": benchmark_name,
            "method": method_name,
            "seed_prompt": seed_prompt,
            "rollout_budget_candidate_evals": config.rollout_budget_candidate_evals,
        },
        "test_score": test_score,
        "best_prompt": best_prompt,
        "best_prompt_tokens": result.best_raw.token_count if result and result.best_raw else 0,
        "seed_score": result.all_candidates[0].score_mean if result and result.all_candidates else None,
        "frontier_size": len(result.frontier) if result else 0,
        "rounds_completed": result.rounds_completed if result else 0,
        "candidate_evals_used": result.total_candidate_evals if result else 0,
        "example_evals_used": result.total_example_evals if result else 0,
        "wall_clock_seconds": wall_clock,
        "termination": termination,
    }

    # Save result
    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    logger.info(f"  Saved: {result_path}")


def main():
    parser = argparse.ArgumentParser(description="Run STAMP-V2 on benchmarks")
    parser.add_argument(
        "--benchmark", nargs="+", default=list(BENCHMARKS),
        help="Benchmarks to run",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Random seeds",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Parallel workers for batch evaluation (1=sequential stamp_v2, >1=stamp_v2_par)",
    )
    args = parser.parse_args()

    runs_dir = Path("runs") / MODEL_TAG
    method = "stamp_v2" if args.parallel <= 1 else f"stamp_v2_par"
    logger.info(f"Method: {method} (workers={args.parallel})")

    for benchmark in args.benchmark:
        for seed in args.seeds:
            run_single(benchmark, seed, runs_dir, parallel_workers=args.parallel)

    logger.info("\nAll runs complete.")


if __name__ == "__main__":
    main()
