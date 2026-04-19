"""ISO Teleprompter: DSPy-compatible optimizer wrapping the core ISO loop."""

from __future__ import annotations

import random
import logging
from typing import Any, Callable

from dspy.teleprompt import Teleprompter

from iso_harness.optimizer.config import ISOConfig
from iso_harness.optimizer.core import iso_compile
from iso_harness.optimizer.runtime import ISORuntime, RolloutCounter, TraceStore
from iso_harness.optimizer.variants import (
    iso_sprint_config,
    iso_grove_config,
    iso_tide_config,
    iso_lens_config,
    iso_storm_config,
)

logger = logging.getLogger("iso")

_VARIANT_FACTORIES = {
    "sprint": iso_sprint_config,
    "grove": iso_grove_config,
    "tide": iso_tide_config,
    "lens": iso_lens_config,
    "storm": iso_storm_config,
}


class ISO(Teleprompter):
    """ISO optimizer implementing the dspy.Teleprompter interface.

    Variant strings accepted (either form):
      - "sprint" / "iso_sprint"
      - "grove"  / "iso_grove"
      - "tide"   / "iso_tide"
      - "lens"   / "iso_lens"
      - "storm"  / "iso_storm"

    Example:
        optimizer = ISO(
            variant="tide",
            metric=my_metric,
            reflection_lm=reflection_lm,
            task_lm=task_lm,
            budget=3500,
            seed=42,
        )
        optimized = optimizer.compile(student, trainset=train, valset=val)
    """

    def __init__(
        self,
        variant: str,
        metric: Callable,
        reflection_lm: Any,
        task_lm: Any,
        budget: int,
        seed: int = 0,
        run_id: str | None = None,
        rollout_counter: RolloutCounter | None = None,
        rollout_writer: Any = None,
        run_dir: Any = None,
        **kwargs,
    ):
        super().__init__()
        self.variant = self._canonical_variant(variant)
        self.metric = metric
        self.reflection_lm = reflection_lm
        self.task_lm = task_lm
        self.budget = budget
        self.seed = seed
        self.run_id = run_id or f"iso-{self.variant}-{seed}"
        self._rollout_counter = rollout_counter
        self._rollout_writer = rollout_writer
        self._run_dir = run_dir
        self._extra_config = kwargs
        self.config = self._build_config(self.variant, budget, seed, kwargs)

    @staticmethod
    def _canonical_variant(variant: str) -> str:
        """Accept both 'sprint' and 'iso_sprint'; return canonical 'sprint'."""
        v = variant.lower().strip()
        if v.startswith("iso_"):
            v = v[4:]
        if v not in _VARIANT_FACTORIES:
            raise ValueError(
                f"Unknown ISO variant: {variant!r}. "
                f"Valid: {', '.join(_VARIANT_FACTORIES)}"
            )
        return v

    @staticmethod
    def _build_config(variant: str, budget: int, seed: int, kwargs: dict) -> ISOConfig:
        base = {"budget": budget, "seed": seed, **kwargs}
        return _VARIANT_FACTORIES[variant](base)

    def compile(
        self,
        student,
        trainset=None,
        valset=None,
        **kwargs,
    ):
        """Run ISO optimization on the student module.

        Args:
            student: DSPy module to optimize.
            trainset: Training examples (list of dspy.Example).
            valset: Validation examples. If None, splits trainset 80/20.

        Returns:
            Optimized dspy.Module.
        """
        if trainset is None:
            raise ValueError("trainset is required for ISO.compile()")

        if valset is None:
            # Deterministic split
            rng = random.Random(self.seed)
            shuffled = list(trainset)
            rng.shuffle(shuffled)
            split = int(len(shuffled) * 0.8)
            trainset, valset = shuffled[:split], shuffled[split:]

        # Construct runtime
        runtime = ISORuntime(
            reflection_lm=self.reflection_lm,
            task_lm=self.task_lm,
            metric=self.metric,
            run_id=self.run_id,
            seed=self.seed,
            rng=random.Random(self.seed),
            trace_store=TraceStore(),
            rollout_counter=self._rollout_counter or RolloutCounter(),
            rollout_writer=self._rollout_writer,
            run_dir=self._run_dir,
        )

        return iso_compile(student, trainset, valset, self.config, runtime)
