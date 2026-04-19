"""Baseline optimizer wrappers (GEPA, MIPROv2) for comparison.

These wrap existing optimizers with the same interface
used by the ISO runner for apples-to-apples comparison.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("iso")


def run_gepa_baseline(
    student,
    trainset: list,
    valset: list,
    metric: Callable,
    reflection_lm: Any,
    budget: int,
    seed: int,
):
    """Run GEPA via the local gepa submodule.

    Uses gepa.api.optimize() with a DSPy adapter.

    NOTE: gepa.api.optimize() does NOT accept a 'student' DSPy module directly.
    It requires a seed_candidate (dict[str, str]) and an adapter.
    The caller is responsible for building a GEPAAdapter wrapping `student`
    and extracting seed prompts.  This wrapper provides a best-effort
    integration; TODO: wire up a proper DSPy adapter once the interface
    is confirmed in gepa.adapters.dspy_adapter.

    Args:
        student: DSPy module to optimize (used to extract seed prompts).
        trainset: Training examples.
        valset: Validation examples.
        metric: Callable metric — (gold, pred, trace, pred_name) -> dict.
        reflection_lm: LM for reflection/mutation proposals.
        budget: Maximum metric calls.
        seed: Random seed.

    Returns:
        GEPAResult from gepa.core.result.
    """
    try:
        from gepa.api import optimize
    except ImportError:
        logger.error("gepa submodule not available. Install with: pip install -e gepa/")
        raise

    # Extract seed prompts from DSPy module named_predictors
    seed_candidate: dict[str, str] = {}
    try:
        for name, predictor in student.named_predictors():
            seed_candidate[name] = predictor.signature.instructions
    except Exception:
        # Fallback: use a generic seed prompt
        seed_candidate = {"qa": "You are a helpful assistant."}

    if not seed_candidate:
        seed_candidate = {"qa": "You are a helpful assistant."}

    # TODO: Build a proper GEPAAdapter that wraps the DSPy student module
    # and routes evaluation through the ISO metric.  For now we pass
    # task_lm=None and evaluator=None, which requires the caller to supply
    # a full adapter.  This will raise at runtime until the adapter is wired.
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        reflection_lm=reflection_lm,
        max_metric_calls=budget,
        seed=seed,
        # adapter must be provided by a higher-level caller — see TODO above
    )
    return result


def run_mipro_baseline(
    student,
    trainset: list,
    valset: list,
    metric: Callable,
    budget: int,
    seed: int,
):
    """Run MIPROv2 via DSPy 2.6.27.

    Args:
        student: DSPy module to optimize.
        trainset: Training examples.
        valset: Validation examples.
        metric: Callable metric compatible with DSPy's signature.
        budget: Approximate budget expressed as num_trials * len(valset).
        seed: Random seed.

    Returns:
        Optimized DSPy module.
    """
    try:
        from dspy.teleprompt import MIPROv2
    except ImportError:
        logger.error("MIPROv2 not available in installed DSPy version.")
        raise

    num_trials = max(1, budget // max(len(valset), 1))
    optimizer = MIPROv2(
        metric=metric,
        auto="medium",
        num_trials=num_trials,
        seed=seed,
    )
    return optimizer.compile(student, trainset=trainset, valset=valset)
