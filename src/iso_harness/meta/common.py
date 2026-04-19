"""Shared data structures and helpers for Track 2 meta-optimizers.

MetaEpisode records one episode of the meta-optimizer outer loop.
MetaAction defines the hyperparameter space the meta-LLM searches over.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

logger = logging.getLogger("iso.meta")


@dataclass
class MetaEpisode:
    """One episode of the meta-optimizer outer loop.

    Attributes:
        episode_id: Unique identifier for this episode.
        meta_run_id: ID of the meta-optimizer run this episode belongs to.
        episode_num: Sequential episode number (0-based).
        proposed_config: Subset of ISOConfig fields proposed by the meta-LLM.
        meta_llm_reasoning: Natural-language reasoning from the meta-LLM.
        episode_outcome: ISO run summary dict (final_score, rollouts, tokens).
        reward: Scalar (constrained) or vector (Pareto) reward.
        playbook_state: Current playbook text (for Cartographer and Atlas).
    """

    episode_id: str = field(default_factory=lambda: str(uuid4()))
    meta_run_id: str = ""
    episode_num: int = 0
    proposed_config: dict = field(default_factory=dict)
    meta_llm_reasoning: str = ""
    episode_outcome: dict = field(default_factory=dict)
    reward: float | list[float] = 0.0
    playbook_state: str = ""


@dataclass
class MetaAction:
    """The action space the meta-LLM picks from.

    Each field has a valid range; the meta-LLM proposes values within these bounds.
    """

    pool_size_seed: int = 5  # 3-8
    mutations_per_seed: int = 2  # 1-3
    minibatch_count: int = 5  # 3-7
    minibatch_size: int = 4  # 2-8
    prune_aggressiveness: float = 0.25  # 0.1-0.6
    max_rounds: int = 10  # 3-20

    def to_dict(self) -> dict:
        return {
            "pool_size_seed": self.pool_size_seed,
            "mutations_per_seed": self.mutations_per_seed,
            "minibatch_count": self.minibatch_count,
            "minibatch_size": self.minibatch_size,
            "prune_aggressiveness": self.prune_aggressiveness,
            "max_rounds": self.max_rounds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MetaAction:
        """Create from a parsed config dict, clamping to valid ranges."""
        return cls(
            pool_size_seed=max(3, min(8, int(d.get("pool_size_seed", 5)))),
            mutations_per_seed=max(1, min(3, int(d.get("mutations_per_seed", 2)))),
            minibatch_count=max(3, min(7, int(d.get("minibatch_count", 5)))),
            minibatch_size=max(2, min(8, int(d.get("minibatch_size", 4)))),
            prune_aggressiveness=max(0.1, min(0.6, float(d.get("prune_aggressiveness", 0.25)))),
            max_rounds=max(3, min(20, int(d.get("max_rounds", 10)))),
        )


def compute_constrained_reward(outcome: dict) -> float:
    """Compute scalar reward with budget penalty.

    If rollouts exceed budget, subtract 1.0 from the score as penalty.
    """
    score = outcome.get("final_score", 0.0)
    rollouts = outcome.get("rollouts_consumed", 0)
    budget = outcome.get("budget", rollouts)  # no penalty if budget not specified

    if rollouts > budget:
        return score - 1.0
    return score


def _compute_budget(config_overrides: dict) -> int:
    """Compute rollout budget from config overrides.

    Mirrors the MetaAction defaults: budget = max_rounds * minibatch_count *
    minibatch_size * pool_size_seed.
    """
    return (
        config_overrides.get("max_rounds", 10)
        * config_overrides.get("minibatch_count", 5)
        * config_overrides.get("minibatch_size", 4)
        * config_overrides.get("pool_size_seed", 5)
    )


def run_iso_with_config(
    variant: str,
    benchmark: str,
    subset_size: int,
    config_overrides: dict,
    meta_lm: Any = None,
    seed: int = 0,
    mock: bool = False,
) -> dict:
    """Run one ISO episode with overridden config.

    This is the inner loop call used by all meta-optimizers.
    Returns an outcome dict with: final_score, rollouts_consumed, tokens_consumed, budget.

    Args:
        variant: ISO variant string (e.g. "iso_tide", "tide").
        benchmark: Benchmark name (e.g. "ifbench", "hotpotqa").
        subset_size: Number of training examples to use as surrogate subset.
        config_overrides: Dict of ISOConfig field overrides (from MetaAction).
        meta_lm: Unused — reserved for future meta-LM integration.
        seed: Random seed for dataset sampling and ISO run.
        mock: When True (or when ISO_META_MOCK=1 env var is set), return stub
              outcome without running ISO. Useful for unit tests.

    Returns:
        Dict with keys: final_score, rollouts_consumed, tokens_consumed, budget.
        On failure (mock=False only), also includes an "error" key.
    """
    import os

    # Allow tests to enable mock mode globally via environment variable
    _mock = mock or os.environ.get("ISO_META_MOCK", "").strip() in ("1", "true", "yes")

    budget = _compute_budget(config_overrides)

    logger.info(
        "run_iso_with_config: variant=%s, benchmark=%s, subset=%s, overrides=%s, mock=%s",
        variant,
        benchmark,
        subset_size,
        config_overrides,
        _mock,
    )

    if _mock:
        return {
            "final_score": 0.0,
            "rollouts_consumed": 0,
            "tokens_consumed": 0,
            "budget": budget,
        }

    # -----------------------------------------------------------------------
    # Live path — run ISO on a surrogate subset
    # -----------------------------------------------------------------------
    try:
        import random

        import dspy

        from gepa_mutations.base import build_qa_task_lm, build_reflection_lm
        from gepa_mutations.benchmarks.evaluators import get_adapter
        from gepa_mutations.benchmarks.loader import load_benchmark
        from gepa_mutations.config import Settings
        from iso_harness.optimizer.feedback_adapter import adapt_evaluator_to_feedback_fn
        from iso_harness.optimizer.helpers import ensure_example_ids
        from iso_harness.optimizer.iso import ISO
        from iso_harness.optimizer.runtime import RolloutCounter

        # 1. Load benchmark and subset the trainset
        data = load_benchmark(benchmark, seed=seed)
        train_full = data.train
        val = data.val

        rng = random.Random(seed)
        if subset_size < len(train_full):
            train = rng.sample(train_full, subset_size)
        else:
            train = list(train_full)

        # Ensure examples have stable IDs
        ensure_example_ids(train, prefix="train")
        ensure_example_ids(val, prefix="val")

        # 2. Build LMs from environment settings
        settings = Settings()
        task_lm = build_qa_task_lm(settings)
        reflection_lm = build_reflection_lm(settings)

        # 3. Configure DSPy task LM
        dspy.settings.configure(lm=task_lm)

        # 4. Build metric from benchmark evaluator
        adapter = get_adapter(benchmark, task_lm=task_lm)

        def _evaluator(gold: Any, pred: Any) -> tuple[float, str]:
            """Extract string prediction and score via adapter."""
            if hasattr(pred, "answer"):
                pred_str = str(pred.answer)
            elif isinstance(pred, str):
                pred_str = pred
            else:
                pred_str = str(pred)
            return adapter._score(gold, pred_str)

        metric = adapt_evaluator_to_feedback_fn(_evaluator)

        # 5. Build student DSPy module
        class _QAStudent(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predict = dspy.Predict("question -> answer")

            def forward(self, **kwargs):
                question = kwargs.get("question") or kwargs.get("input", "")
                return self.predict(question=question)

        student = _QAStudent()

        # 6. Build rollout counter for tracking
        rollout_counter = RolloutCounter()

        # 7. Map config_overrides to ISO constructor kwargs
        # ISOConfig fields directly supported as kwargs in ISO.__init__ via _build_config
        iso_kwargs: dict[str, Any] = {}
        _iso_config_fields = {
            "max_rounds", "mutations_per_seed", "minibatch_count",
            "minibatch_size", "pool_floor", "plateau_rounds_threshold",
            "plateau_tolerance", "n_discovery_examples",
            "target_skills_min", "target_skills_max", "merge_interval",
        }
        for k, v in config_overrides.items():
            if k in _iso_config_fields:
                iso_kwargs[k] = v

        # 8. Run ISO optimization
        optimizer = ISO(
            variant=variant,
            metric=metric,
            reflection_lm=reflection_lm,
            task_lm=task_lm,
            budget=budget,
            seed=seed,
            rollout_counter=rollout_counter,
            **iso_kwargs,
        )
        optimized = optimizer.compile(student, trainset=train, valset=val)

        # 9. Score optimized module on val set
        val_scores = []
        for example in val:
            try:
                question = getattr(example, "question", None) or getattr(example, "input", "")
                pred = optimized(question=question)
                result = metric(example, pred, trace=None, pred_name=None)
                if isinstance(result, dict):
                    val_scores.append(float(result.get("score", 0.0)))
                else:
                    val_scores.append(float(result))
            except Exception as ex:
                logger.debug("Val scoring error: %s", ex)
                val_scores.append(0.0)

        final_score = sum(val_scores) / len(val_scores) if val_scores else 0.0
        rollouts_consumed = rollout_counter.value()

        logger.info(
            "run_iso_with_config complete: variant=%s benchmark=%s "
            "final_score=%.4f rollouts=%d budget=%d",
            variant, benchmark, final_score, rollouts_consumed, budget,
        )

        return {
            "final_score": final_score,
            "rollouts_consumed": rollouts_consumed,
            "tokens_consumed": 0,  # token tracking not wired at meta level yet
            "budget": budget,
        }

    except Exception as e:
        logger.error(
            "run_iso_with_config failed: variant=%s benchmark=%s error=%s: %s",
            variant, benchmark, type(e).__name__, e,
        )
        return {
            "final_score": 0.0,
            "rollouts_consumed": 0,
            "tokens_consumed": 0,
            "budget": budget,
            "error": str(e),
        }


def update_pareto_frontier(
    frontier: list[dict],
    new_reward: list[float],
    outcome: dict,
) -> list[dict]:
    """Add new point to frontier if non-dominated; remove dominated points.

    Each point is a dict with 'reward' (list[float]) and 'outcome' (dict).
    """
    point = {"reward": new_reward, "outcome": outcome}

    # Check if dominated by existing frontier
    for existing in frontier:
        if all(e >= n for e, n in zip(existing["reward"], new_reward)) and any(
            e > n for e, n in zip(existing["reward"], new_reward)
        ):
            return frontier  # new point is dominated

    # Remove points dominated by new point
    frontier = [
        existing
        for existing in frontier
        if not (
            all(n >= e for n, e in zip(new_reward, existing["reward"]))
            and any(n > e for n, e in zip(new_reward, existing["reward"]))
        )
    ]

    frontier.append(point)
    return frontier
