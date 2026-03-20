"""Experiment runner for GEPA benchmarks.

Uses gepa.api.optimize() (not optimize_anything()) with custom GEPAAdapter
implementations per benchmark, matching the paper's exact configuration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import dspy
import litellm
from gepa.api import optimize
from rich.console import Console

from gepa_mutations.benchmarks.evaluators import get_adapter
from gepa_mutations.benchmarks.loader import load_benchmark
from gepa_mutations.config import PAPER_ROLLOUTS, Settings
from gepa_mutations.runner.callbacks import MetricsCallback
from gepa_mutations.storage.local import save_result


class LM:
    """Lightweight LM wrapper over LiteLLM matching GEPA's LanguageModel protocol.

    Conforms to: (str) -> str
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_retries: int = 3,
        **kwargs: Any,
    ):
        self.model = model
        self.num_retries = num_retries
        self.completion_kwargs: dict[str, Any] = {
            **({"temperature": temperature} if temperature is not None else {}),
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            "timeout": kwargs.pop("timeout", 120),
            **kwargs,
        }

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        completion = litellm.completion(
            model=self.model,
            messages=messages,
            num_retries=self.num_retries,
            drop_params=True,
            **self.completion_kwargs,
        )
        return completion.choices[0].message.content

    def __repr__(self) -> str:
        return f"LM(model={self.model!r})"

console = Console()

# Default seeds for multi-seed experiments (5 independent runs)
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

# Initial seed prompt for optimization
SEED_PROMPT = (
    "Solve the math problem carefully. Break down the steps and provide "
    "the final answer as a single number."
)

# Benchmark-specific seed prompts
BENCHMARK_SEED_PROMPTS = {
    "aime": SEED_PROMPT,
    "livebench": (
        "Solve the math problem step by step. Provide the final answer in the exact "
        "format requested (number, expression, comma-separated list, etc.)."
    ),
    "hotpotqa": (
        "Answer the question by reasoning step by step through the provided context. "
        "Give a concise, factual answer."
    ),
    "ifbench": (
        "Follow the instructions carefully, ensuring you satisfy all the given constraints. "
        "Provide a complete response that meets every requirement."
    ),
    "hover": (
        "Determine whether the given claim is SUPPORTED or NOT_SUPPORTED based on the "
        "provided evidence. Reason through the evidence step by step."
    ),
    "pupa": (
        "Rewrite the user's query to remove all personally identifiable information (PII) "
        "such as names, addresses, phone numbers, and emails. Replace PII with generic "
        "placeholders while preserving the query's meaning and intent."
    ),
}


@dataclass
class ExperimentResult:
    """Result of a single GEPA experiment run."""

    benchmark: str
    seed: int
    test_score: float
    val_score: float
    best_prompt: dict[str, str]
    rollout_count: int
    config_snapshot: dict[str, Any]
    wall_clock_seconds: float
    metrics: dict[str, Any] | None = None
    all_candidates: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "seed": self.seed,
            "test_score": self.test_score,
            "val_score": self.val_score,
            "best_prompt": self.best_prompt,
            "rollout_count": self.rollout_count,
            "wall_clock_seconds": self.wall_clock_seconds,
            "num_candidates": len(self.all_candidates),
        }


class ExperimentRunner:
    """Orchestrates GEPA experiments on benchmarks using the optimize() API."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

    def _build_task_lm(self) -> dspy.LM:
        """Build the task LM (dspy.LM for math benchmarks)."""
        return dspy.LM(
            f"openrouter/{self.settings.gepa_model}",
            temperature=self.settings.gepa_temperature,
            top_p=self.settings.gepa_top_p,
            top_k=self.settings.gepa_top_k,
            max_tokens=self.settings.gepa_max_context,
            timeout=120,
        )

    def _build_reflection_lm(self) -> LM:
        """Build the reflection LM (gepa.lm.LM with explicit params)."""
        return LM(
            f"openrouter/{self.settings.gepa_model}",
            temperature=self.settings.gepa_temperature,
            max_tokens=self.settings.gepa_max_context,
            top_p=self.settings.gepa_top_p,
            top_k=self.settings.gepa_top_k,
        )

    def _build_qa_task_lm(self) -> LM:
        """Build the task LM for QA benchmarks (gepa.lm.LM for direct calls)."""
        return LM(
            f"openrouter/{self.settings.gepa_model}",
            temperature=self.settings.gepa_temperature,
            max_tokens=self.settings.gepa_max_context,
            top_p=self.settings.gepa_top_p,
            top_k=self.settings.gepa_top_k,
        )

    def _get_seed_prompt(self, benchmark: str) -> str:
        return BENCHMARK_SEED_PROMPTS.get(benchmark, SEED_PROMPT)

    def _uses_dspy(self, benchmark: str) -> bool:
        """Check if benchmark uses dspy (AIME only) vs direct LM calls."""
        return benchmark == "aime"

    def run(
        self,
        benchmark: str,
        seed: int = 42,
        subset: int | None = None,
        use_merge: bool = True,
        max_metric_calls: int | None = None,
        dry_run: bool = False,
    ) -> ExperimentResult:
        """Run a single GEPA experiment.

        Args:
            benchmark: Benchmark name (aime, hotpotqa, etc.)
            seed: Random seed for reproducibility.
            subset: If set, use only this many examples (for quick testing).
            use_merge: Whether to use merge strategy (default True, matching paper).
            max_metric_calls: Override rollout budget (defaults to paper budget).
            dry_run: If True, validate config and data loading without running.

        Returns:
            ExperimentResult with scores, best prompt, and diagnostics.
        """
        start_time = time.time()

        # Load benchmark data
        console.print(f"[bold]Loading benchmark: {benchmark}[/bold]")
        data = load_benchmark(benchmark, seed=0)  # Data loading always uses seed 0
        console.print(
            f"  Train: {len(data.train)}, Val: {len(data.val)}, Test: {len(data.test)}"
        )

        # Apply subset if requested
        trainset = data.train[:subset] if subset else data.train
        valset = data.val[:subset] if subset else data.val
        testset = data.test

        if dry_run:
            console.print("[green]Dry run: config validated, data loaded successfully.[/green]")
            return ExperimentResult(
                benchmark=benchmark,
                seed=seed,
                test_score=0.0,
                val_score=0.0,
                best_prompt={"system_prompt": self._get_seed_prompt(benchmark)},
                rollout_count=0,
                config_snapshot=self._config_snapshot(benchmark, seed, subset, use_merge),
                wall_clock_seconds=0.0,
            )

        # Configure task LM
        if self._uses_dspy(benchmark):
            task_lm = self._build_task_lm()
            dspy.configure(lm=task_lm)
            adapter = get_adapter(benchmark)
        else:
            qa_lm = self._build_qa_task_lm()
            adapter = get_adapter(benchmark, task_lm=qa_lm)

        # Build reflection LM (explicit params, not bare string)
        reflection_lm = self._build_reflection_lm()

        # Rollout budget
        if max_metric_calls is None:
            max_metric_calls = PAPER_ROLLOUTS["gepa"].get(benchmark, 5000)

        # Metrics callback
        metrics_cb = MetricsCallback(benchmark=benchmark, seed=seed)

        # Seed candidate
        seed_prompt = self._get_seed_prompt(benchmark)
        seed_candidate = {"system_prompt": seed_prompt}

        # Run directory for GEPA state persistence
        run_dir = f"runs/{benchmark}/gepa/{seed}/gepa_state"

        console.print(f"[bold]Running GEPA optimization[/bold]")
        console.print(f"  Benchmark: {benchmark}, Seed: {seed}")
        console.print(f"  Rollout budget: {max_metric_calls}")
        console.print(f"  Merge: {use_merge}")

        # Run optimization via optimize() API
        result = optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            candidate_selection_strategy="pareto",
            frontier_type="instance",
            skip_perfect_score=True,
            perfect_score=1.0,
            module_selector="round_robin",
            use_merge=use_merge,
            max_merge_invocations=5,
            max_metric_calls=max_metric_calls,
            cache_evaluation=True,
            seed=seed,
            run_dir=run_dir,
            callbacks=[metrics_cb],
            display_progress_bar=True,
            raise_on_exception=True,
        )

        # Extract results
        best_prompt = result.best_candidate
        if isinstance(best_prompt, str):
            best_prompt = {"system_prompt": best_prompt}
        val_score = result.val_aggregate_scores[result.best_idx]

        console.print(f"\n[bold green]Optimization complete![/bold green]")
        console.print(f"  Best val score: {val_score:.4f}")
        console.print(f"  Candidates explored: {result.num_candidates}")

        # Evaluate best prompt on test set
        console.print(f"\n[bold]Evaluating on test set ({len(testset)} examples)...[/bold]")
        test_score = self._evaluate_on_test(benchmark, best_prompt, testset)
        console.print(f"  Test score: {test_score:.4f} ({test_score * 100:.2f}%)")

        wall_clock = time.time() - start_time

        # Build experiment result
        exp_result = ExperimentResult(
            benchmark=benchmark,
            seed=seed,
            test_score=test_score,
            val_score=val_score,
            best_prompt=best_prompt,
            rollout_count=result.total_metric_calls or 0,
            config_snapshot=self._config_snapshot(benchmark, seed, subset, use_merge),
            wall_clock_seconds=wall_clock,
            metrics=metrics_cb.metrics.to_dict(),
            all_candidates=result.candidates,
        )

        # Save results
        save_result(
            benchmark=benchmark,
            seed=seed,
            result_data=exp_result.to_dict(),
            config_data=exp_result.config_snapshot,
            metrics_data=exp_result.metrics,
        )
        console.print(f"  Results saved to runs/{benchmark}/gepa/{seed}/")

        return exp_result

    def run_multi_seed(
        self,
        benchmark: str,
        seeds: list[int] | None = None,
        subset: int | None = None,
        use_merge: bool = True,
    ) -> list[ExperimentResult]:
        """Run experiments with multiple seeds for statistical analysis."""
        seeds = seeds or DEFAULT_SEEDS
        results = []

        for i, seed in enumerate(seeds):
            console.print(f"\n{'='*60}")
            console.print(f"[bold]Seed {i+1}/{len(seeds)}: {seed}[/bold]")
            console.print(f"{'='*60}")
            result = self.run(
                benchmark=benchmark,
                seed=seed,
                subset=subset,
                use_merge=use_merge,
            )
            results.append(result)

        # Print summary
        test_scores = [r.test_score for r in results]
        mean_score = sum(test_scores) / len(test_scores)
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Multi-seed summary for {benchmark}[/bold]")
        console.print(f"  Seeds: {seeds}")
        console.print(f"  Scores: {[f'{s:.4f}' for s in test_scores]}")
        console.print(f"  Mean: {mean_score:.4f} ({mean_score * 100:.2f}%)")
        console.print(f"{'='*60}")

        return results

    def _evaluate_on_test(
        self, benchmark: str, best_prompt: dict[str, str], testset: list
    ) -> float:
        """Evaluate the best prompt on the test set."""
        if self._uses_dspy(benchmark):
            return self._evaluate_dspy(best_prompt, testset)
        else:
            qa_lm = self._build_qa_task_lm()
            return self._evaluate_qa(best_prompt, testset, qa_lm)

    def _evaluate_dspy(self, prompt: dict[str, str], testset: list) -> float:
        """Evaluate using dspy (for math benchmarks)."""
        from gepa_mutations.benchmarks.evaluators import _math_metric, _run_llm
        from gepa_mutations.benchmarks.signatures import MathSolverSignature

        predictor = dspy.ChainOfThought(MathSolverSignature)
        prompt_text = prompt["system_prompt"]
        total = len(testset)

        correct = 0
        errors = 0
        for i, example in enumerate(testset):
            try:
                prediction = _run_llm(example, prompt_text, predictor)
                score, _ = _math_metric(example, prediction)
                correct += score
            except Exception as e:
                errors += 1
                if errors <= 3:
                    console.print(f"  [dim]Test eval error on example {i}: {e}[/dim]")

            if (i + 1) % 10 == 0 or (i + 1) == total:
                pct = (i + 1) / total * 100
                acc = correct / (i + 1) * 100
                console.print(
                    f"  Test eval: {i+1}/{total} ({pct:.0f}%) | "
                    f"correct: {correct}/{i+1} ({acc:.1f}%) | errors: {errors}"
                )

        return correct / total if total else 0.0

    def _evaluate_qa(self, prompt: dict[str, str], testset: list, lm: LM) -> float:
        """Evaluate using direct LM calls (for QA benchmarks)."""
        prompt_text = prompt["system_prompt"]
        total = len(testset)

        correct = 0
        errors = 0
        for i, example in enumerate(testset):
            try:
                messages = [
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": example.input},
                ]
                response = lm(messages)
                answer = str(example.answer).lower()
                if answer in response.lower():
                    correct += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    console.print(f"  [dim]Test eval error on example {i}: {e}[/dim]")

            if (i + 1) % 10 == 0 or (i + 1) == total:
                pct = (i + 1) / total * 100
                acc = correct / (i + 1) * 100
                console.print(
                    f"  Test eval: {i+1}/{total} ({pct:.0f}%) | "
                    f"correct: {correct}/{i+1} ({acc:.1f}%) | errors: {errors}"
                )

        return correct / total if total else 0.0

    def _config_snapshot(
        self, benchmark: str, seed: int, subset: int | None, use_merge: bool
    ) -> dict[str, Any]:
        return {
            "benchmark": benchmark,
            "seed": seed,
            "subset": subset,
            "use_merge": use_merge,
            "model": f"openrouter/{self.settings.gepa_model}",
            "temperature": self.settings.gepa_temperature,
            "top_p": self.settings.gepa_top_p,
            "top_k": self.settings.gepa_top_k,
            "max_context": self.settings.gepa_max_context,
            "rollout_budget": PAPER_ROLLOUTS["gepa"].get(benchmark),
            "candidate_selection": "pareto",
            "frontier_type": "instance",
            "module_selector": "round_robin",
            "skip_perfect_score": True,
            "perfect_score": 1.0,
            "max_merge_invocations": 5,
            "cache_evaluation": True,
        }
