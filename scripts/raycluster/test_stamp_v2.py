"""Quick smoke test for STAMP-V2 on the raycluster.

Usage:
    uv run python scripts/raycluster/test_stamp_v2.py

Tests STAMP-V2 with Qwen3.5-27B via the vLLM endpoint.
Runs a minimal optimization (budget=10) to verify the pipeline works.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Cluster config
INFERENCE_BASE_URL = "http://10.0.10.66:8123/v1"
MODEL_NAME = "openai/gpt-oss-120b"


def main():
    from stamp_v2.config import StampConfig
    from stamp_v2.data.splits import split_benchmark
    from stamp_v2.data.synthetic_constraints import generate_synthetic_benchmark
    from stamp_v2.evaluators.constraint_eval import ConstraintEvaluator
    from stamp_v2.models.openai_compatible_client import OpenAICompatibleClient
    from stamp_v2.optimization.stamp_optimizer import STAMPV2Optimizer

    # Moderate budget — enough for ~6 rounds of optimization
    config = StampConfig(
        experiment_name="stamp_v2_cluster_smoke",
        rollout_budget_candidate_evals=20,
        max_candidate_evals_per_round=3,
        diagnostic_batch_size=8,
        validation_batch_size=16,
        random_seed=42,
        num_bundles_per_reflection=2,
        # Relaxed screening for smoke test (small batch = noisy)
        min_delta_score=0.0,
        diagnostic_min_examples_improved=1,
        diagnostic_max_examples_regressed=3,
        bundle_utility_threshold=0.0,
    )

    # Generate small benchmark
    logger.info("Generating synthetic benchmark...")
    examples = generate_synthetic_benchmark(n=50, seed=42)
    split = split_benchmark(
        examples,
        diagnostic_size=config.diagnostic_batch_size,
        validation_size=config.validation_batch_size,
        seed=42,
    )
    logger.info(
        f"Split: {len(split.diagnostic)} diagnostic, "
        f"{len(split.validation)} validation, "
        f"{len(split.holdout)} holdout"
    )

    # Build clients pointing to cluster
    logger.info(f"Connecting to {INFERENCE_BASE_URL} ({MODEL_NAME})...")
    task_model = OpenAICompatibleClient(
        model_name=MODEL_NAME,
        base_url=INFERENCE_BASE_URL,
        temperature=0.6,  # Higher temp so model responds to prompt changes
        max_tokens=256,
        timeout=60.0,
    )
    reflector_model = OpenAICompatibleClient(
        model_name=MODEL_NAME,
        base_url=INFERENCE_BASE_URL,
        temperature=0.7,
        max_tokens=1024,
        timeout=120.0,
    )

    # Quick connectivity test
    logger.info("Testing connectivity...")
    try:
        test_response = task_model.generate("Say hello in one word.")
        logger.info(f"Connectivity OK: '{test_response[:50]}'")
    except Exception as e:
        logger.error(f"Cannot reach inference endpoint: {e}")
        sys.exit(1)

    # Run optimizer
    evaluator = ConstraintEvaluator()
    output_dir = Path("runs/stamp_v2_smoke")

    optimizer = STAMPV2Optimizer(
        task_model=task_model,
        reflector_model=reflector_model,
        evaluator=evaluator,
        config=config,
        output_dir=output_dir,
    )

    logger.info("Starting STAMP-V2 optimization (budget=10)...")
    result = optimizer.optimize(split)

    # Report
    logger.info(f"\n{'='*60}")
    logger.info("STAMP-V2 Smoke Test Results")
    logger.info(f"{'='*60}")
    logger.info(f"Rounds completed: {result.rounds_completed}")
    logger.info(f"Candidate evals: {result.total_candidate_evals}")
    logger.info(f"Example evals: {result.total_example_evals}")
    logger.info(f"Frontier size: {len(result.frontier)}")

    if result.best_raw:
        logger.info(f"Best raw score: {result.best_raw.score_mean:.4f}")
        logger.info(f"Best raw tokens: {result.best_raw.token_count}")
        logger.info(f"Best raw prompt:\n{result.best_raw.render()}")

    seed_score = result.all_candidates[0].score_mean if result.all_candidates else None
    if seed_score and result.best_raw:
        gain = result.best_raw.score_mean - seed_score
        logger.info(f"Gain over seed: {gain:+.4f}")

    logger.info(f"\nOutput saved to: {output_dir}")
    logger.info("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
