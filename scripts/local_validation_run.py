#!/usr/bin/env python
"""Local validation run: ISO-Tide with real MLX model on small IFBench subset.

Measures wall-clock, token consumption, generates JSONL output for inspection.
NOT for meaningful scores — the 0.6B model is too small. This validates wiring.

Usage:
    export GEPA_MODEL="mlx-community/Qwen3-0.6B-4bit"
    export GEPA_BASE_URL="http://localhost:8132/v1"
    python scripts/local_validation_run.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("local_validation")


def main():
    import dspy

    from gepa_mutations.base import build_qa_task_lm, build_reflection_lm
    from gepa_mutations.benchmarks.evaluators import get_adapter
    from gepa_mutations.benchmarks.loader import load_benchmark
    from gepa_mutations.config import Settings
    from iso_harness.experiment.context import set_context
    from iso_harness.experiment.jsonl_writer import JSONLWriter
    from iso_harness.experiment.logging_lm import LoggingLM
    from iso_harness.optimizer.helpers import ensure_example_ids
    from iso_harness.optimizer.iso import ISO
    from iso_harness.optimizer.runtime import RolloutCounter

    # --- Config ---
    VARIANT = "tide"
    BENCHMARK = "hotpotqa"  # simpler than ifbench for a quick validation
    BUDGET = 50  # minimal budget for wiring validation
    SEED = 42
    SUBSET_SIZE = 10  # minimal subset
    RUN_ID = f"local-validation-{VARIANT}-{SEED}"

    # --- Output dir ---
    run_dir = Path(f"runs/local-validation/{RUN_ID}")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Local Validation Run ===")
    logger.info(f"Variant: {VARIANT}, Benchmark: {BENCHMARK}")
    logger.info(f"Budget: {BUDGET}, Seed: {SEED}, Subset: {SUBSET_SIZE}")
    logger.info(f"Output: {run_dir}")

    # --- Load benchmark ---
    logger.info("Loading benchmark...")
    data = load_benchmark(BENCHMARK, seed=SEED)
    import random
    rng = random.Random(SEED)

    trainset = rng.sample(list(data.train), min(SUBSET_SIZE, len(data.train)))
    valset = rng.sample(list(data.val), min(SUBSET_SIZE // 3, len(data.val)))
    ensure_example_ids(trainset, prefix="train")
    ensure_example_ids(valset, prefix="val")
    logger.info(f"Train: {len(trainset)}, Val: {len(valset)}")

    # --- Build LMs ---
    logger.info("Building LMs...")
    settings = Settings()

    # Build LMs directly via dspy.LM for BaseLM compatibility
    model_name = settings.gepa_model or os.environ.get("GEPA_MODEL", "")
    base_url = settings.gepa_base_url or os.environ.get("GEPA_BASE_URL", "")
    # Disable Qwen3 thinking mode — prevents <think> blocks that break DSPy parsing
    _no_think = {"chat_template_kwargs": {"enable_thinking": False}}

    task_lm = dspy.LM(
        model=f"openai/{model_name}",
        api_base=base_url,
        api_key="not-needed",
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
        extra_body=_no_think,
    )
    # Use same model for reflection (single-model local setup)
    reflection_lm_raw = task_lm

    # Wrap reflection LM with LoggingLM
    reflection_writer = JSONLWriter(run_dir / "reflections.jsonl")
    reflection_lm = LoggingLM(
        lm=reflection_lm_raw,
        writer=reflection_writer,
        role="reflection",
    )

    # Rollout writer
    rollout_writer = JSONLWriter(run_dir / "rollouts.jsonl")

    # Configure DSPy
    dspy.settings.configure(lm=task_lm)

    # --- Build metric ---
    adapter = get_adapter(BENCHMARK, task_lm=task_lm)

    def metric(gold, pred, trace=None, pred_name=None):
        if hasattr(pred, "answer"):
            pred_str = str(pred.answer)
        elif isinstance(pred, str):
            pred_str = pred
        else:
            pred_str = str(pred)
        score, feedback = adapter._score(gold, pred_str)
        return {"score": float(score), "feedback": str(feedback), "metadata": {}}

    # --- Build student ---
    class SimpleQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = dspy.Predict("question -> answer")

        def forward(self, **kwargs):
            question = kwargs.get("question") or kwargs.get("input", "")
            return self.qa(question=question)

    student = SimpleQA()

    # --- Smoke config overrides (smaller for speed) ---
    overrides = {
        "n_discovery_examples": 3,
        "target_skills_min": 1,
        "target_skills_max": 2,
        "mutations_per_seed": 1,  # need at least 2 candidates for tournament
        "minibatch_count": 1,
        "minibatch_size": 2,
        "pool_floor": 2,
        "max_rounds": 2,
        "merge_interval": 99,
        "plateau_rounds_threshold": 99,
    }

    # --- Run ISO ---
    logger.info("Starting ISO optimization...")
    rollout_counter = RolloutCounter()

    optimizer = ISO(
        variant=VARIANT,
        metric=metric,
        reflection_lm=reflection_lm,
        task_lm=task_lm,
        budget=BUDGET,
        seed=SEED,
        run_id=RUN_ID,
        rollout_counter=rollout_counter,
        rollout_writer=rollout_writer,
        run_dir=run_dir,
        **overrides,
    )

    start_time = time.time()
    result = optimizer.compile(student, trainset=trainset, valset=valset)
    elapsed = time.time() - start_time

    # --- Results ---
    rollouts_consumed = rollout_counter.value()
    logger.info(f"=== Run Complete ===")
    logger.info(f"Wall clock: {elapsed:.1f}s")
    logger.info(f"Rollouts consumed: {rollouts_consumed}")
    logger.info(f"Result type: {type(result).__name__}")

    # --- Read JSONL files ---
    rollout_records = rollout_writer.read_all()
    reflection_records = reflection_writer.read_all()

    logger.info(f"Rollout JSONL records: {len(rollout_records)}")
    logger.info(f"Reflection JSONL records: {len(reflection_records)}")

    # --- Validate context propagation ---
    if rollout_records:
        sample = rollout_records[0]
        logger.info(f"Sample rollout record keys: {list(sample.keys())}")
        logger.info(f"  run_id: {sample.get('run_id', 'MISSING')}")
        logger.info(f"  round_num: {sample.get('round_num', 'MISSING')}")
        logger.info(f"  candidate_id: {sample.get('candidate_id', 'MISSING')}")
        logger.info(f"  example_id: {sample.get('example_id', 'MISSING')}")
        logger.info(f"  score: {sample.get('score', 'MISSING')}")

        # Check all records have required fields
        missing_run_id = sum(1 for r in rollout_records if not r.get("run_id"))
        missing_candidate = sum(1 for r in rollout_records if not r.get("candidate_id"))
        missing_example = sum(1 for r in rollout_records if not r.get("example_id"))
        logger.info(f"Records missing run_id: {missing_run_id}")
        logger.info(f"Records missing candidate_id: {missing_candidate}")
        logger.info(f"Records missing example_id: {missing_example}")

    if reflection_records:
        sample = reflection_records[0]
        logger.info(f"Sample reflection record keys: {list(sample.keys())}")
        logger.info(f"  run_id: {sample.get('run_id', 'MISSING')}")
        logger.info(f"  round_num: {sample.get('round_num', 'MISSING')}")
        logger.info(f"  target_candidate_id: {sample.get('target_candidate_id', 'MISSING')}")

    # --- Checkpoint check ---
    checkpoint_path = run_dir / "checkpoint" / "iso_state.json"
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        logger.info(f"Checkpoint: round={ckpt['round_num']}, pool_size={len(ckpt['pool'])}, "
                     f"rollouts={ckpt['rollouts_consumed']}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}")

    # --- Per-candidate lifecycle trace ---
    if rollout_records:
        candidates_seen = {}
        for r in rollout_records:
            cid = r.get("candidate_id", "")
            if cid not in candidates_seen:
                candidates_seen[cid] = {"first_round": r.get("round_num"), "count": 0, "scores": []}
            candidates_seen[cid]["count"] += 1
            candidates_seen[cid]["scores"].append(r.get("score", 0))
            candidates_seen[cid]["last_round"] = r.get("round_num")

        logger.info(f"\n=== Candidate Lifecycle Summary ===")
        logger.info(f"Total unique candidates: {len(candidates_seen)}")
        for cid, info in sorted(candidates_seen.items(), key=lambda x: x[1]["first_round"]):
            avg = sum(info["scores"]) / len(info["scores"]) if info["scores"] else 0
            logger.info(f"  {cid[:12]}... rounds {info['first_round']}-{info['last_round']}, "
                         f"{info['count']} evals, avg_score={avg:.3f}")

    # --- Cost extrapolation ---
    logger.info(f"\n=== Cost Extrapolation (budget=3500) ===")
    if elapsed > 0 and rollouts_consumed > 0:
        time_per_rollout = elapsed / rollouts_consumed
        extrapolated_time = time_per_rollout * 3500
        logger.info(f"Time per rollout: {time_per_rollout:.2f}s")
        logger.info(f"Extrapolated wall-clock for budget=3500: {extrapolated_time:.0f}s "
                     f"({extrapolated_time/60:.1f}min, {extrapolated_time/3600:.1f}hr)")
        logger.info(f"Note: real hardware (L40S/H100) will be faster than MLX 0.6B")
    else:
        logger.warning("No rollouts consumed — cannot extrapolate")

    # --- Write summary ---
    summary = {
        "variant": VARIANT,
        "benchmark": BENCHMARK,
        "budget": BUDGET,
        "seed": SEED,
        "subset_size": SUBSET_SIZE,
        "wall_clock_seconds": elapsed,
        "rollouts_consumed": rollouts_consumed,
        "rollout_records": len(rollout_records),
        "reflection_records": len(reflection_records),
        "unique_candidates": len(candidates_seen) if rollout_records else 0,
    }
    summary_path = run_dir / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
