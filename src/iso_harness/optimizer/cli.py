"""CLI entry point for running ISO optimizer experiments.

Usage:
    python -m iso_harness.optimizer.cli --variant iso_tide --benchmark ifbench --budget 3500 --seed 42
    python -m iso_harness.optimizer.cli --variant iso_tide --smoke-test --mock-lm
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time

logger = logging.getLogger("iso")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ISO Optimizer CLI")
    parser.add_argument("--variant", required=True,
                       choices=["iso_sprint", "iso_grove", "iso_tide", "iso_lens", "iso_storm",
                                "sprint", "grove", "tide", "lens", "storm"],
                       help="ISO variant to run")
    parser.add_argument("--benchmark", default="ifbench",
                       help="Benchmark name (default: ifbench)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--budget", type=int, default=3500, help="Rollout budget")
    parser.add_argument("--subset-size", type=int, default=None,
                       help="Limit dataset size (for smoke testing)")
    parser.add_argument("--smoke-test", action="store_true",
                       help="Run with reduced config (budget=100, subset=20)")
    parser.add_argument("--mock-lm", action="store_true",
                       help="Use mock LMs instead of real vLLM servers")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory (default: runs/<run_id>)")
    return parser


def main(args=None):
    parser = build_parser()
    opts = parser.parse_args(args)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    if opts.smoke_test:
        opts.budget = 100
        if opts.subset_size is None:
            opts.subset_size = 20

    logger.info(f"ISO CLI: variant={opts.variant}, benchmark={opts.benchmark}, "
                f"seed={opts.seed}, budget={opts.budget}")

    if opts.mock_lm:
        _run_with_mock_lm(opts)
    else:
        _run_with_real_lm(opts)


def _run_with_mock_lm(opts):
    """Run with mock LMs for testing."""
    import dspy
    from unittest.mock import MagicMock
    from dspy.clients.base_lm import BaseLM
    from iso_harness.optimizer.iso import ISO
    from iso_harness.optimizer.helpers import ensure_example_ids
    from tests.mocks.mock_lm import MockReflectionLM, MockMetric

    # Create a DSPy-compatible mock LM that returns field markers DSPy can parse.
    class _DSPyMockLM(BaseLM):
        """BaseLM subclass that returns canned [[ ## field ## ]] markers."""

        def __init__(self):
            super().__init__(model="mock-task-lm", cache=False)

        def forward(self, prompt=None, messages=None, **kwargs):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "[[ ## answer ## ]]\nmock answer"
            response.usage = {"prompt_tokens": 10, "completion_tokens": 5}
            response.model = "mock-task-lm"
            return response

    mock_lm = _DSPyMockLM()
    dspy.settings.configure(lm=mock_lm)

    # Simple student
    class SimpleQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = dspy.Predict("question -> answer")

        def forward(self, question):
            return self.qa(question=question)

    # Synthetic data
    n = opts.subset_size or 20
    trainset = [
        dspy.Example(question=f"Q{i}?", answer=f"A{i}").with_inputs("question")
        for i in range(n)
    ]
    ensure_example_ids(trainset)
    valset = trainset[int(n * 0.8):]
    trainset = trainset[:int(n * 0.8)]
    ensure_example_ids(valset, prefix="val")

    # Build smoke overrides
    extra = {}
    if opts.smoke_test:
        extra = {
            "n_discovery_examples": 5,
            "target_skills_min": 2,
            "target_skills_max": 3,
            "mutations_per_seed": 0,
            "minibatch_count": 2,
            "minibatch_size": 2,
            "pool_floor": 2,
            "max_rounds": 3,
            "merge_interval": 2,
            "plateau_rounds_threshold": 99,
        }

    optimizer = ISO(
        variant=opts.variant,
        metric=MockMetric(base_score=0.4),
        reflection_lm=MockReflectionLM(),
        task_lm=mock_lm,
        budget=opts.budget,
        seed=opts.seed,
        **extra,
    )

    start = time.time()
    result = optimizer.compile(SimpleQA(), trainset=trainset, valset=valset)
    elapsed = time.time() - start

    logger.info(f"Completed in {elapsed:.1f}s")
    logger.info(f"Result type: {type(result).__name__}")


def _run_with_real_lm(opts):
    """Run with real vLLM servers."""
    logger.info("Real LM mode — loading benchmark and configuring LMs...")

    try:
        import dspy
        # TODO: Verify the exact module paths for load_benchmark and build_*_lm
        # if the gepa_mutations package layout changes.
        from gepa_mutations.benchmarks.loader import load_benchmark
        from gepa_mutations.base import build_qa_task_lm, build_reflection_lm
        from gepa_mutations.config import Settings
        from iso_harness.optimizer.iso import ISO
        from iso_harness.optimizer.helpers import ensure_example_ids
        from iso_harness.optimizer.feedback_adapter import adapt_evaluator_to_feedback_fn
        from gepa_mutations.benchmarks.evaluators import get_adapter
    except ImportError as e:
        logger.error(f"Missing dependency for real LM mode: {e}")
        logger.error("Install with: uv sync")
        sys.exit(1)

    # Load benchmark
    settings = Settings()
    data = load_benchmark(opts.benchmark, seed=opts.seed)
    trainset = list(data.train)
    valset = list(data.val)

    if opts.subset_size:
        rng = random.Random(opts.seed)
        trainset = rng.sample(trainset, min(opts.subset_size, len(trainset)))
        valset = rng.sample(valset, min(opts.subset_size, len(valset)))

    ensure_example_ids(trainset)
    ensure_example_ids(valset, prefix="val")

    # Build LMs via dspy.LM for BaseLM compatibility
    import os
    model_name = settings.gepa_model or os.environ.get("GEPA_MODEL", "")
    base_url = settings.gepa_base_url or os.environ.get("GEPA_BASE_URL", "")
    refl_model = getattr(settings, "reflection_model", None) or os.environ.get("REFLECTION_MODEL", model_name)
    refl_url = getattr(settings, "reflection_base_url", None) or os.environ.get("REFLECTION_BASE_URL", base_url)

    # Disable Qwen3 thinking mode via extra_body — prevents <think> blocks
    # that waste tokens and break DSPy's structured output parsing
    _no_think = {"chat_template_kwargs": {"enable_thinking": False}}

    task_lm = dspy.LM(
        model=f"openai/{model_name}",
        api_base=base_url,
        api_key="not-needed",
        temperature=0.6,
        max_tokens=2048,
        extra_body=_no_think,
    )
    reflection_lm = dspy.LM(
        model=f"openai/{refl_model}",
        api_base=refl_url,
        api_key="not-needed",
        temperature=0.6,
        max_tokens=1024,
        extra_body=_no_think,
    )
    dspy.settings.configure(lm=task_lm)

    # Build student (simple QA for now)
    class BenchmarkQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = dspy.Predict("question -> answer")

        def forward(self, **kwargs):
            question = kwargs.get("question") or kwargs.get("input", "")
            return self.qa(question=question)

    # Adapt evaluator — bridge adapter._score to ISO metric contract
    adapter = get_adapter(opts.benchmark, task_lm=task_lm)

    def _evaluator(gold, pred):
        if hasattr(pred, "answer"):
            pred_str = str(pred.answer)
        elif isinstance(pred, str):
            pred_str = pred
        else:
            pred_str = str(pred)
        return adapter._score(gold, pred_str)

    metric = adapt_evaluator_to_feedback_fn(_evaluator)

    # Extra config for smoke test
    extra = {}
    if opts.smoke_test:
        extra = {
            "n_discovery_examples": 5,
            "target_skills_min": 2,
            "target_skills_max": 3,
            "mutations_per_seed": 0,
            "minibatch_count": 2,
            "minibatch_size": 2,
            "pool_floor": 2,
            "max_rounds": 3,
            "merge_interval": 2,
            "plateau_rounds_threshold": 99,
        }

    # Set up output directory, JSONL writers, and LoggingLM
    from pathlib import Path
    from iso_harness.experiment.jsonl_writer import JSONLWriter
    from iso_harness.experiment.logging_lm import LoggingLM
    from iso_harness.optimizer.runtime import RolloutCounter

    run_dir = Path(opts.output_dir) if opts.output_dir else Path(f"runs/iso-{opts.variant}-{opts.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    rollout_writer = JSONLWriter(run_dir / "rollouts.jsonl")
    reflection_writer = JSONLWriter(run_dir / "reflections.jsonl")
    reflection_lm_logged = LoggingLM(lm=reflection_lm, writer=reflection_writer, role="reflection")

    optimizer = ISO(
        variant=opts.variant,
        metric=metric,
        reflection_lm=reflection_lm_logged,
        task_lm=task_lm,
        budget=opts.budget,
        seed=opts.seed,
        rollout_writer=rollout_writer,
        run_dir=run_dir,
        **extra,
    )

    result = optimizer.compile(BenchmarkQA(), trainset=trainset, valset=valset)
    logger.info(f"Optimization complete. Result: {type(result).__name__}")


if __name__ == "__main__":
    main()
