"""Factory function returning a run_fn callback for the orchestrator.

The orchestrator calls ``run_fn(spec, run_dir, budget)`` for each experiment
run.  This module provides ``make_run_fn(config)`` which closes over the
experiment config and returns a correctly-typed callback.

Supported optimizer strings
----------------------------
ISO variants:    "iso_sprint" / "sprint", "iso_grove" / "grove",
                 "iso_tide" / "tide", "iso_lens" / "lens", "iso_storm" / "storm"
Baselines:       "gepa", "mipro" / "miprov2"
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import dspy

from gepa_mutations.base import build_qa_task_lm, build_reflection_lm
from gepa_mutations.benchmarks.evaluators import get_adapter
from gepa_mutations.benchmarks.loader import load_benchmark
from gepa_mutations.config import Settings
from iso_harness.experiment.config import ISOExperimentConfig
from iso_harness.experiment.context import set_context
from iso_harness.experiment.jsonl_writer import JSONLWriter
from iso_harness.experiment.logging_lm import LoggingLM
from iso_harness.experiment.orchestrator import BudgetEnforcer, RunSpec
from iso_harness.experiment.reporter import generate_run_report

logger = logging.getLogger(__name__)

# Optimizer strings that route to ISO
_ISO_PREFIXES = {"iso_"}
_ISO_VARIANTS = {"sprint", "grove", "tide", "lens", "storm"}

# Optimizer strings that route to baselines
_MIPRO_NAMES = {"mipro", "miprov2"}


# ---------------------------------------------------------------------------
# Simple QA DSPy module
# ---------------------------------------------------------------------------


class _QAModule(dspy.Module):
    """Minimal DSPy QA module: question -> answer."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        return self.predict(question=question)


# ---------------------------------------------------------------------------
# Per-optimizer evaluation helpers
# ---------------------------------------------------------------------------


def _score_dataset(
    module: dspy.Module,
    dataset: list,
    metric: Callable,
) -> float:
    """Evaluate module on a dataset, returning mean score.

    Uses the ISO-style metric signature: (gold, pred, trace, pred_name) -> dict.
    Falls back gracefully on any single-example error.
    """
    if not dataset:
        return 0.0

    total = 0.0
    for example in dataset:
        try:
            pred = module(question=example.question if hasattr(example, "question") else example.input)
            result = metric(example, pred, trace=None, pred_name=None)
            if isinstance(result, dict):
                total += float(result.get("score", 0.0))
            else:
                total += float(result)
        except Exception as e:
            logger.debug("Score error on example: %s", e)

    return total / len(dataset)


# ---------------------------------------------------------------------------
# make_run_fn — public factory
# ---------------------------------------------------------------------------


def make_run_fn(config: ISOExperimentConfig) -> Callable:
    """Factory that returns a run_fn for the orchestrator's execute() method.

    The returned function closes over ``config`` and creates fresh Settings,
    LMs, and optimizers on every call so that parallel runs (if ever used)
    stay isolated.

    Args:
        config: Validated experiment configuration.

    Returns:
        run_fn(spec, run_dir, budget) -> dict
    """

    def run_fn(
        spec: RunSpec,
        run_dir: Path,
        budget: BudgetEnforcer,
    ) -> dict[str, Any]:
        """Execute a single experiment run.

        Args:
            spec: Run specification (optimizer, benchmark, seed, budget).
            run_dir: Directory for this run's artifacts.
            budget: Thread-safe rollout budget counter.

        Returns:
            Summary dict with status, scores, and metadata.
        """
        start_time = time.time()
        run_dir = Path(run_dir)

        # 1. Context
        set_context(run_id=spec.run_id, phase=config.phase)

        try:
            # 2. Load benchmark
            logger.info("[%s] Loading benchmark: %s (seed=%d)", spec.run_id, spec.benchmark, spec.seed)
            data = load_benchmark(spec.benchmark, seed=spec.seed)
            train = data.train
            val = data.val

            # 3. Build LMs
            settings = Settings()
            task_lm = build_qa_task_lm(settings)
            reflection_lm = build_reflection_lm(settings)

            # 3b. Wrap reflection LM with JSONL logging
            reflection_writer = JSONLWriter(run_dir / "reflections.jsonl")
            reflection_lm = LoggingLM(
                lm=reflection_lm,
                writer=reflection_writer,
                role="reflection",
            )

            # 4. Configure DSPy
            dspy.settings.configure(lm=task_lm)

            # 5. Create student module
            student = _QAModule()

            # 6. Build metric from benchmark evaluator
            adapter = get_adapter(spec.benchmark, task_lm=task_lm)

            def metric(gold: Any, pred: Any, trace: Any = None, pred_name: str | None = None) -> dict:  # noqa: ARG001
                """ISO-style metric wrapping the benchmark adapter."""
                # Extract string answer from DSPy prediction or string
                if hasattr(pred, "answer"):
                    pred_str = str(pred.answer)
                elif isinstance(pred, str):
                    pred_str = pred
                else:
                    pred_str = str(pred)

                score, feedback = adapter._score(gold, pred_str)
                return {
                    "score": float(score),
                    "feedback": str(feedback),
                    "metadata": {},
                }

            # 7. Branch on optimizer type
            optimizer_name = spec.optimizer.lower().strip()

            if optimizer_name in _ISO_VARIANTS or any(optimizer_name.startswith(p) for p in _ISO_PREFIXES):
                # ---- ISO optimizer ----
                from iso_harness.optimizer.iso import ISO
                from iso_harness.optimizer.runtime import RolloutCounter

                rollout_writer = JSONLWriter(run_dir / "rollouts.jsonl")
                rollout_counter = RolloutCounter(enforcer=budget)

                optimized = ISO(
                    variant=spec.optimizer,
                    metric=metric,
                    reflection_lm=reflection_lm,
                    task_lm=task_lm,
                    budget=spec.budget_rollouts,
                    seed=spec.seed,
                    run_id=spec.run_id,
                    rollout_counter=rollout_counter,
                    rollout_writer=rollout_writer,
                    run_dir=run_dir,
                ).compile(student, trainset=train, valset=val)

                val_score = _score_dataset(optimized, val, metric)
                optimizer_label = spec.optimizer

            elif optimizer_name == "gepa":
                # ---- GEPA baseline ----
                from gepa.api import optimize as gepa_optimize

                seed_candidate = {"system_prompt": "You are a helpful assistant."}

                gepa_result = gepa_optimize(
                    seed_candidate=seed_candidate,
                    trainset=train,
                    valset=val,
                    adapter=adapter,  # type: ignore[arg-type]  # runtime-compatible
                    reflection_lm=reflection_lm,
                    max_metric_calls=spec.budget_rollouts,
                    seed=spec.seed,
                    run_dir=str(run_dir / "gepa_state"),
                    raise_on_exception=True,
                )

                # Report rollouts consumed
                budget.record_rollouts(gepa_result.total_metric_calls or spec.budget_rollouts)

                val_score = float(
                    gepa_result.val_aggregate_scores[gepa_result.best_idx]
                    if gepa_result.val_aggregate_scores
                    else 0.0
                )
                optimized = student  # GEPA doesn't produce a DSPy module
                optimizer_label = "gepa"

            elif optimizer_name in _MIPRO_NAMES:
                # ---- MIPROv2 baseline ----
                try:
                    from dspy.teleprompt import MIPROv2

                    num_candidates = max(1, spec.budget_rollouts // max(len(val), 1))
                    optimized = MIPROv2(
                        metric=metric,
                        auto="medium",
                        num_candidates=num_candidates,
                        seed=spec.seed,
                    ).compile(student, trainset=train, valset=val)

                    budget.record_rollouts(spec.budget_rollouts)
                    val_score = _score_dataset(optimized, val, metric)

                except ImportError as e:
                    raise RuntimeError(
                        f"MIPROv2 not available in installed DSPy version: {e}"
                    ) from e

                optimizer_label = "miprov2"

            else:
                raise ValueError(
                    f"Unknown optimizer: {spec.optimizer!r}. "
                    f"Expected one of: iso_sprint, iso_grove, iso_tide, iso_lens, iso_storm, gepa, mipro."
                )

            # 8. Final score on val set (already computed above for each branch)
            elapsed = time.time() - start_time

            # 9. Build summary
            summary = {
                "run_id": spec.run_id,
                "optimizer": optimizer_label,
                "benchmark": spec.benchmark,
                "seed": spec.seed,
                "status": "completed",
                "final_score_val": val_score,
                "final_score_test": 0.0,  # test eval deferred (expensive, add later)
                "rollouts_consumed": budget.consumed,
                "rollouts_consumed_total": budget.consumed,
                "tokens_consumed_total": 0,
                "duration_seconds": elapsed,
                "wall_clock_seconds": elapsed,
                "cost_estimate_usd": 0.0,
                "model_task": getattr(task_lm, "model", str(task_lm)),
                "model_reflection": getattr(reflection_lm, "model", str(reflection_lm)),
                "final_candidate_prompts": {},
                "final_candidate_id": spec.run_id,
            }

            # 10. Write reports
            generate_run_report(run_dir, summary)

            logger.info(
                "[%s] Done. val_score=%.4f, rollouts=%d, elapsed=%.1fs",
                spec.run_id, val_score, budget.consumed, elapsed,
            )

            return {
                "status": "completed",
                "final_score_val": val_score,
                "final_score_test": 0.0,
                "optimizer": optimizer_label,
                "benchmark": spec.benchmark,
                "seed": spec.seed,
                "rollouts_consumed": budget.consumed,
                "elapsed_seconds": elapsed,
            }

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("[%s] Run failed: %s: %s", spec.run_id, type(e).__name__, e)

            error_summary = {
                "run_id": spec.run_id,
                "optimizer": spec.optimizer,
                "benchmark": spec.benchmark,
                "seed": spec.seed,
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "final_score_val": 0.0,
                "final_score_test": 0.0,
                "rollouts_consumed": budget.consumed,
                "rollouts_consumed_total": budget.consumed,
                "tokens_consumed_total": 0,
                "duration_seconds": elapsed,
                "wall_clock_seconds": elapsed,
                "cost_estimate_usd": 0.0,
                "model_task": "",
                "model_reflection": "",
                "final_candidate_prompts": {},
                "final_candidate_id": spec.run_id,
            }

            try:
                generate_run_report(run_dir, error_summary)
            except Exception as report_err:
                logger.debug("Failed to write error report: %s", report_err)

            return {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "optimizer": spec.optimizer,
                "benchmark": spec.benchmark,
                "seed": spec.seed,
                "rollouts_consumed": budget.consumed,
                "elapsed_seconds": elapsed,
            }

    return run_fn
