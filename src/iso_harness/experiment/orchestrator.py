"""Experiment orchestrator: run matrix, budget enforcement, sequential execution.

Manages the full experiment lifecycle -- iterates through (optimizer, benchmark, seed)
triples, enforces matched-GEPA rollout budgets, handles failures, and coordinates
with checkpoint/MLflow systems.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from iso_harness.experiment.config import ISOExperimentConfig

logger = logging.getLogger(__name__)


# -- Budget Enforcement -------------------------------------------------------


class BudgetExhaustedError(Exception):
    """Raised when a run's rollout budget is exhausted."""
    pass


class BudgetEnforcer:
    """Thread-safe rollout budget counter with hard-stop.

    Args:
        max_rollouts: Maximum number of rollouts allowed for this run.
    """

    def __init__(self, max_rollouts: int) -> None:
        if max_rollouts < 0:
            raise ValueError(f"max_rollouts must be non-negative, got {max_rollouts}")
        self._max = max_rollouts
        self._count = 0
        self._lock = threading.Lock()

    def record_rollouts(self, n: int = 1) -> None:
        """Record n rollouts consumed."""
        with self._lock:
            self._count += n

    @property
    def remaining(self) -> int:
        """Rollouts remaining before budget exhaustion."""
        with self._lock:
            return max(0, self._max - self._count)

    @property
    def consumed(self) -> int:
        """Total rollouts consumed so far."""
        with self._lock:
            return self._count

    @property
    def is_exhausted(self) -> bool:
        """True if budget has been fully consumed."""
        with self._lock:
            return self._count >= self._max

    def check(self) -> None:
        """Raise BudgetExhaustedError if budget is exhausted."""
        if self.is_exhausted:
            raise BudgetExhaustedError(
                f"Budget exhausted: {self._count}/{self._max} rollouts consumed"
            )


# -- Run Specification --------------------------------------------------------


class RunSpec(BaseModel):
    """Specification for a single experiment run."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    optimizer: str
    variant_config: dict[str, Any] = Field(default_factory=dict)
    benchmark: str
    seed: int
    budget_rollouts: int = Field(gt=0)


# -- Orchestrator -------------------------------------------------------------


def _get_git_sha() -> str:
    """Get current git HEAD SHA, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _is_tree_dirty() -> bool:
    """Check if git working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        return bool(result.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


class Orchestrator:
    """Sequential experiment runner with budget enforcement and failure handling.

    Args:
        config: Validated experiment configuration.
        runs_dir: Base directory for run outputs (default: current dir / "runs").
    """

    def __init__(
        self,
        config: ISOExperimentConfig,
        runs_dir: str | Path = "runs",
    ) -> None:
        self.config = config
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._consecutive_failures = 0

    def build_matrix(self) -> list[RunSpec]:
        """Generate all (optimizer, benchmark, seed) run specs from config.

        Returns:
            List of RunSpec objects, one per experiment run.
        """
        specs: list[RunSpec] = []

        if self.config.run_matrix is None:
            # Pilot mode: use smoke_test config if available
            if self.config.smoke_test is not None:
                st = self.config.smoke_test
                for seed in [0]:  # Single seed for smoke test
                    specs.append(RunSpec(
                        optimizer=st.optimizer,
                        benchmark=st.benchmark,
                        seed=seed,
                        budget_rollouts=st.budget_rollouts,
                    ))
            return specs

        matrix = self.config.run_matrix
        for benchmark_name, benchmark_config in matrix.benchmarks.items():
            for optimizer_name, optimizer_config in matrix.optimizers.items():
                for seed in matrix.seeds:
                    specs.append(RunSpec(
                        optimizer=optimizer_name,
                        variant_config=optimizer_config if isinstance(optimizer_config, dict) else {},
                        benchmark=benchmark_name,
                        seed=seed,
                        budget_rollouts=benchmark_config.budget_rollouts,
                    ))

        logger.info("Built run matrix: %d runs", len(specs))
        return specs

    def execute(
        self,
        matrix: list[RunSpec],
        run_fn: Any = None,
        strict_git: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute all runs in the matrix sequentially.

        Args:
            matrix: List of RunSpec objects to execute.
            run_fn: Callable(run_spec, run_dir, budget_enforcer) -> dict.
                If None, runs are logged but not executed (useful for testing).
            strict_git: If True, refuse to start if git tree is dirty.

        Returns:
            List of result dicts, one per run (with status, scores, errors).
        """
        # Git SHA enforcement
        git_sha = _get_git_sha()
        if strict_git and _is_tree_dirty():
            raise RuntimeError(
                "Strict git mode: working tree is dirty. Commit all changes before "
                "starting experiment runs."
            )
        if not strict_git and _is_tree_dirty():
            git_sha = f"{git_sha}-dirty"
            logger.warning("Git tree is dirty -- recording SHA as %s", git_sha)

        results: list[dict[str, Any]] = []
        total = len(matrix)

        for i, spec in enumerate(matrix):
            run_dir = self.runs_dir / spec.run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save run config
            config_data = {
                "run_id": spec.run_id,
                "optimizer": spec.optimizer,
                "variant_config": spec.variant_config,
                "benchmark": spec.benchmark,
                "seed": spec.seed,
                "budget_rollouts": spec.budget_rollouts,
                "git_sha": git_sha,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            config_path = run_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2, default=str)

            logger.info(
                "[%d/%d] Starting: %s/%s/seed=%d (budget=%d)",
                i + 1, total, spec.optimizer, spec.benchmark, spec.seed,
                spec.budget_rollouts,
            )

            budget = BudgetEnforcer(spec.budget_rollouts)
            start_time = time.time()

            try:
                result: dict[str, Any] = {}
                if run_fn is not None:
                    result = run_fn(spec, run_dir, budget)
                else:
                    result = {"status": "skipped", "reason": "no run_fn provided"}

                elapsed = time.time() - start_time
                result["run_id"] = spec.run_id
                result.setdefault("status", "completed")
                result["elapsed_seconds"] = elapsed
                result["rollouts_consumed"] = budget.consumed

                # Write COMPLETE marker
                from iso_harness.experiment.jsonl_writer import write_complete_marker
                write_complete_marker(run_dir, run_id=spec.run_id)

                self._consecutive_failures = 0
                results.append(result)

                logger.info(
                    "[%d/%d] Completed: %s/%s/seed=%d in %.1fs (%d rollouts)",
                    i + 1, total, spec.optimizer, spec.benchmark, spec.seed,
                    elapsed, budget.consumed,
                )

            except BudgetExhaustedError:
                elapsed = time.time() - start_time
                result = {
                    "run_id": spec.run_id,
                    "status": "budget_exhausted",
                    "elapsed_seconds": elapsed,
                    "rollouts_consumed": budget.consumed,
                }
                results.append(result)
                self._consecutive_failures = 0
                logger.info(
                    "Run %s hit budget limit at %d rollouts",
                    spec.run_id, budget.consumed,
                )

            except Exception as e:
                elapsed = time.time() - start_time
                self._consecutive_failures += 1

                error_info = {
                    "run_id": spec.run_id,
                    "status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "elapsed_seconds": elapsed,
                    "rollouts_consumed": budget.consumed,
                }
                results.append(error_info)

                # Log error to file
                error_log = run_dir / "errors.log"
                with open(error_log, "a") as f:
                    f.write(
                        f"[{datetime.now(timezone.utc).isoformat()}] "
                        f"{type(e).__name__}: {e}\n"
                    )

                logger.error(
                    "[%d/%d] Failed: %s/%s/seed=%d -- %s: %s",
                    i + 1, total, spec.optimizer, spec.benchmark, spec.seed,
                    type(e).__name__, e,
                )

                # Halt on 3+ consecutive failures
                if self._consecutive_failures >= 3:
                    logger.critical(
                        "3+ consecutive failures -- halting orchestrator. "
                        "Last error: %s: %s", type(e).__name__, e,
                    )
                    break

        return results

    def dry_run(self, matrix: list[RunSpec]) -> None:
        """Validate config and print the run matrix without executing."""
        print(f"Dry run -- {len(matrix)} runs planned:")
        print(f"  Phase: {self.config.phase}")
        print(f"  Runs dir: {self.runs_dir}")
        print()
        for i, spec in enumerate(matrix):
            print(
                f"  [{i+1:3d}] {spec.optimizer}/{spec.benchmark}/seed={spec.seed} "
                f"(budget={spec.budget_rollouts})"
            )
        print(f"\nTotal rollout budget: {sum(s.budget_rollouts for s in matrix)}")
