"""Run ISO optimization variants on the raycluster.

Runs all 5 ISO variants (Sprint, Grove, Tide, Lens, Storm) using Qwen3.5-27B
as both task LM and reflection LM.

Usage (on gho-vm-2):
    uv run python scripts/raycluster/run_iso.py
    uv run python scripts/raycluster/run_iso.py --variant sprint grove
    uv run python scripts/raycluster/run_iso.py --benchmark hotpotqa --seeds 42 --variant sprint
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests
from rich.console import Console

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (  # noqa: E402
    BENCHMARKS,
    BENCHMARK_MAX_TOKENS,
    BENCHMARK_PARALLEL_WORKERS,
    DISABLE_THINKING,
    INFERENCE_BASE_URL,
    INFRA_TAG,
    MAX_TOKENS_QA,
    MAX_TOKENS_REFLECT,
    MODEL_FULL_NAME,
    MODEL_NAME,
    MODEL_TAG,
    PARALLEL_WORKERS,
    SEEDS,
    TEMPERATURE,
    TOP_P,
)

from iso_harness.optimizer.core import iso_compile  # noqa: E402
from iso_harness.optimizer.config import ISOConfig  # noqa: E402
from iso_harness.optimizer.runtime import ISORuntime, RolloutCounter, TraceStore  # noqa: E402
from iso_harness.optimizer.variants import (  # noqa: E402
    iso_sprint_config,
    iso_grove_config,
    iso_tide_config,
    iso_lens_config,
    iso_storm_config,
)
from gepa_mutations.benchmarks.loader import load_benchmark  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()

# Constants
SEED_PROMPT = "You are a helpful assistant."
BENCHMARK_SEED_PROMPTS = {
    "aime": (
        "You are a helpful assistant. You are given a question and you need to answer "
        "it. The answer should be given at the end of your response in exactly the "
        "format '### <final answer>'"
    ),
}


def get_seed_prompt(benchmark: str) -> str:
    return BENCHMARK_SEED_PROMPTS.get(benchmark, SEED_PROMPT)
ALL_VARIANTS = ["sprint", "grove", "tide", "lens", "storm"]

VARIANT_FACTORIES = {
    "sprint": iso_sprint_config,
    "grove": iso_grove_config,
    "tide": iso_tide_config,
    "lens": iso_lens_config,
    "storm": iso_storm_config,
}

# Rollout budgets (matching GEPA paper for fair comparison)
PAPER_ROLLOUTS = {
    "hotpotqa": 6871,
    "hover": 2426,
    "pupa": 3936,
    "ifbench": 3593,
    "livebench": 1839,
    "aime": 7051,
}


class ClusterLM:
    """LM wrapper for the cluster inference API.

    Implements callable interface: __call__(messages) -> str
    Used for reflection LM (called directly by ISO's reflection strategies).
    """

    def __init__(self, max_tokens: int = MAX_TOKENS_QA, role: str = "task"):
        self.max_tokens = max_tokens
        self.role = role
        self.total_tokens = 0
        self.call_count = 0
        self.errors = 0

    def __call__(self, messages, **kwargs) -> str:
        """Call the inference API. Accepts messages list or plain string prompt."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        return self._raw_call(messages, max_tokens)

    def _raw_call(self, messages: list, max_tokens: int) -> str:
        """Low-level API call."""
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }
        if DISABLE_THINKING:
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        try:
            timeout = 120 if max_tokens <= 512 else 300
            resp = requests.post(
                f"{INFERENCE_BASE_URL}/chat/completions",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            msg = data["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            usage = data.get("usage", {})
            self.total_tokens += usage.get("total_tokens", 0)
            self.call_count += 1
            return text

        except Exception as e:
            self.errors += 1
            logger.warning(f"LM call error ({self.role}): {e}")
            raise


def _make_cluster_dspy_lm(max_tokens: int = MAX_TOKENS_QA):
    """Create a dspy.BaseLM subclass instance for cluster inference.

    Must be created after dspy is imported. Uses the openai Python client
    directly (returns proper response objects with attribute access).
    """
    import dspy
    from openai import OpenAI

    client = OpenAI(base_url=INFERENCE_BASE_URL, api_key="EMPTY", timeout=120.0)

    class _ClusterDSPyLM(dspy.BaseLM):
        def __init__(self, max_tokens_inner):
            super().__init__(
                model=MODEL_NAME,
                model_type="chat",
                temperature=TEMPERATURE,
                max_tokens=max_tokens_inner,
            )
            self._total_tokens = 0
            self._call_count = 0

        def forward(self, prompt=None, messages=None, **kwargs):
            """Return OpenAI chat completion response object."""
            if messages is None:
                messages = [{"role": "user", "content": prompt}]
            mt = kwargs.get("max_tokens", self.kwargs.get("max_tokens", MAX_TOKENS_QA))

            extra_body = None
            if DISABLE_THINKING:
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=mt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                extra_body=extra_body,
            )

            # Fix content if thinking mode leaked (content=None, reasoning in other field)
            msg = response.choices[0].message
            if not msg.content:
                # Try reasoning_content (Qwen3 thinking mode fallback)
                raw_content = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
                raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
                msg.content = raw_content

            self._total_tokens += getattr(response.usage, "total_tokens", 0) if response.usage else 0
            self._call_count += 1

            return response

    return _ClusterDSPyLM(max_tokens)


def build_iso_metric(benchmark_name: str):
    """Build a metric function compatible with ISO's evaluation system.

    Returns a function: (example, prediction, trace, pred_name) -> dict
    The dict must have keys: "score" (float), "feedback" (str), optionally "metadata".
    Uses identical scoring logic to run_baseline.py and run_gepa.py.
    """
    from gepa_mutations.benchmarks.evaluators import _check_ifbench_by_id, _check_ifbench_constraint

    def metric(example, prediction, trace=None, pred_name=None):
        """Score a prediction against gold. Returns {"score": float, "feedback": str}."""
        response = str(prediction) if not hasattr(prediction, "answer") else str(prediction.answer)
        answer = str(example.answer).strip().lower() if hasattr(example, "answer") else ""
        score = 0.0
        feedback = ""

        if benchmark_name == "hotpotqa":
            if re.search(r'\b' + re.escape(answer) + r'\b', response.lower()):
                score = 1.0
                feedback = "Correct answer found."
            else:
                feedback = f"Expected '{answer}' not found in response."

        elif benchmark_name == "hover":
            resp_lower = response.lower()
            not_supported_indicators = [
                "not_supported", "not supported", "not enough", "insufficient",
                "does not support", "doesn't support", "cannot be verified",
                "no evidence", "contradicts", "refuted", "false", "incorrect",
            ]
            supported_indicators = ["supported", "verified", "confirmed", "true", "correct"]
            model_says_not_supported = any(ind in resp_lower for ind in not_supported_indicators)
            model_says_supported = any(ind in resp_lower for ind in supported_indicators) and not model_says_not_supported

            if answer in ("not_supported", "0"):
                score = 1.0 if model_says_not_supported else 0.0
            else:
                score = 1.0 if model_says_supported else 0.0
            feedback = f"Gold={answer}, model_supported={model_says_supported}, model_not_supported={model_says_not_supported}"

        elif benchmark_name == "ifbench":
            instruction_ids = getattr(example, "instruction_ids", [])
            kwargs_list = getattr(example, "kwargs_list", [])
            constraints = getattr(example, "constraints", [])

            if instruction_ids:
                satisfied = 0
                total = len(instruction_ids)
                for i, iid in enumerate(instruction_ids):
                    kw = kwargs_list[i] if i < len(kwargs_list) else None
                    if _check_ifbench_by_id(iid, kw, response):
                        satisfied += 1
                score = satisfied / total if total > 0 else 0.0
                feedback = f"Satisfied {satisfied}/{total} constraints (structured)."
            elif constraints:
                satisfied = sum(1 for c in constraints if _check_ifbench_constraint(c, response))
                total = len(constraints)
                score = satisfied / total if total > 0 else 0.0
                feedback = f"Satisfied {satisfied}/{total} constraints (text-based)."
            else:
                feedback = "No constraints found."

        elif benchmark_name == "pupa":
            pii_units = getattr(example, "pii_units", [])
            if not pii_units:
                score = 1.0
                feedback = "No PII to protect."
            else:
                leaked = sum(1 for pii in pii_units if pii.lower() in response.lower())
                score = 1.0 - (leaked / len(pii_units))
                feedback = f"Leaked {leaked}/{len(pii_units)} PII units."

        elif benchmark_name == "livebench":
            norm_answer = answer.strip().replace(",", "").replace(" ", "")
            norm_response = response.strip().lower().replace(",", "").replace(" ", "")
            if norm_answer == norm_response or norm_answer in norm_response:
                score = 1.0
            else:
                boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
                if boxed and boxed[-1].strip().lower().replace(",", "").replace(" ", "") == norm_answer:
                    score = 1.0
            feedback = f"Expected '{answer}', got response (match={score > 0})."

        elif benchmark_name == "aime":
            nums = re.findall(r'\b(\d+)\b', response)
            if nums:
                predicted = int(nums[-1])
                try:
                    score = 1.0 if predicted == int(answer) else 0.0
                except ValueError:
                    score = 0.0
            feedback = f"Expected {answer}, extracted {nums[-1] if nums else 'nothing'}."

        else:
            score = 1.0 if answer in response.lower() else 0.0
            feedback = f"Generic containment check for '{answer}'."

        return {"score": score, "feedback": feedback}

    return metric


def run_iso_single(
    variant_name: str, benchmark_name: str, seed: int, runs_dir: Path
) -> dict:
    """Run a single ISO variant on one benchmark + seed."""
    method_name = f"iso_{variant_name}"

    # Skip if result already exists
    result_path = runs_dir / benchmark_name / method_name / str(seed) / "result.json"
    if result_path.exists():
        console.print(f"\n[dim]ISO-{variant_name}: {benchmark_name} / seed={seed} — result exists, skipping[/dim]")
        with open(result_path) as f:
            return json.load(f)

    console.print(f"\n[bold]ISO-{variant_name}: {benchmark_name} / seed={seed}[/bold]")

    # Load data
    data = load_benchmark(benchmark_name, seed=0)
    trainset = data.train
    valset = data.val
    testset = data.test
    console.print(f"  Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")

    # Build fresh LMs (isolation: new instances per run)
    max_tokens = BENCHMARK_MAX_TOKENS.get(benchmark_name, MAX_TOKENS_QA)
    task_lm = ClusterLM(max_tokens=max_tokens, role="task")
    reflection_lm = ClusterLM(max_tokens=MAX_TOKENS_REFLECT, role="reflection")

    # Build metric
    metric = build_iso_metric(benchmark_name)

    # Build run directory
    run_dir = runs_dir / benchmark_name / method_name / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Budget
    budget = PAPER_ROLLOUTS.get(benchmark_name, 5000)

    # Build ISO config for this variant
    base_config = {
        "budget": budget,
        "seed": seed,
        "n_discovery_examples": min(20, len(trainset)),
        "mutations_per_seed": 2,
        "minibatch_count": 5,
        "minibatch_size": 3,
        "max_rounds": 20,
        "plateau_rounds_threshold": 3,
        "plateau_tolerance": 0.005,
        "merge_interval": 3,
    }
    config = VARIANT_FACTORIES[variant_name](base_config)

    # Build fresh runtime (isolation: new runtime per run)
    run_id = str(uuid4())
    runtime = ISORuntime(
        reflection_lm=reflection_lm,
        task_lm=task_lm,
        metric=metric,
        run_id=run_id,
        seed=seed,
        rng=random.Random(seed),
        trace_store=TraceStore(),
        rollout_counter=RolloutCounter(),
        run_dir=str(run_dir),
    )

    # Track per-round scores
    round_scores = []
    original_round_num = 0

    console.print(f"  Budget: {budget}, Variant: {variant_name}")
    console.print(f"  Starting ISO optimization...")

    t0 = time.time()

    try:
        # Build a DSPy module for ISO to optimize.
        # We bypass DSPy's adapter/parsing layer entirely — the model gives free text
        # and our metric scores it directly. ISO modifies the signature instructions
        # (system prompt) via apply_candidate_prompts.
        import dspy

        task_dspy_lm = _make_cluster_dspy_lm(max_tokens=max_tokens)

        class QAModule(dspy.Module):
            def __init__(self):
                super().__init__()
                # Predictor exists so ISO can modify its signature.instructions
                self.generate = dspy.Predict("input -> answer")

            def forward(self, input: str):
                # Get current instructions (modified by ISO during optimization)
                instructions = self.generate.signature.instructions or get_seed_prompt(benchmark_name)

                # Call LM.forward() directly to get raw OpenAI response object
                messages = [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": input},
                ]
                response = task_dspy_lm.forward(messages=messages)

                # Extract text from response object
                text = response.choices[0].message.content or ""
                return dspy.Prediction(answer=text)

        student = QAModule()
        dspy.settings.configure(lm=task_dspy_lm)

        # Run the ISO optimization loop
        optimized_student = iso_compile(
            student=student,
            trainset=trainset,
            valset=valset,
            config=config,
            runtime=runtime,
        )

        # Extract best prompt from the optimized student
        best_prompt = {"system_prompt": get_seed_prompt(benchmark_name)}  # Default
        # Try to extract from optimized module
        for name, pred in optimized_student.named_predictors():
            if hasattr(pred, "demos") or hasattr(pred, "instructions"):
                instr = getattr(pred, "instructions", None)
                if instr:
                    best_prompt = {"system_prompt": instr}
                    break

        wall_clock = time.time() - t0
        termination = "completed"

    except Exception as e:
        console.print(f"  [bold red]ISO optimization failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        best_prompt = {"system_prompt": get_seed_prompt(benchmark_name)}
        wall_clock = time.time() - t0
        termination = f"error: {str(e)[:200]}"

    # Evaluate best prompt on test set
    console.print(f"  Evaluating on test set ({len(testset)} examples)...")
    test_lm = ClusterLM(max_tokens=max_tokens, role="test_eval")
    test_scores = []
    for example in testset:
        try:
            response = test_lm([
                {"role": "system", "content": best_prompt["system_prompt"]},
                {"role": "user", "content": str(example.input)},
            ])
            result = metric(example, type("P", (), {"answer": response})())
            test_scores.append(result["score"] if isinstance(result, dict) else result)
        except Exception:
            test_scores.append(0.0)

    test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0

    console.print(f"  [bold green]Test score: {test_score:.4f}[/bold green]")
    console.print(f"  Wall clock: {wall_clock:.1f}s ({wall_clock/60:.1f}m)")
    console.print(f"  Task LM: {task_lm.call_count} calls, {task_lm.total_tokens} tokens")
    console.print(f"  Reflection LM: {reflection_lm.call_count} calls, {reflection_lm.total_tokens} tokens")
    console.print(f"  Rounds completed: {runtime.round_num}")
    console.print(f"  Rollouts used: {runtime.rollout_counter.value()}")

    # Build result
    git_sha = ""
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
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
            "variant": variant_name,
            "seed_prompt": get_seed_prompt(benchmark_name),
            "rollout_budget": budget,
            "termination": termination,
        },
        "test_score": test_score,
        "val_score": None,
        "best_prompt": best_prompt,
        "test_example_scores": test_scores,
        "rollout_count": runtime.rollout_counter.value(),
        "wall_clock_seconds": wall_clock,
        "rounds_completed": runtime.round_num,
    }

    metrics = {
        "total_tokens": task_lm.total_tokens + reflection_lm.total_tokens + test_lm.total_tokens,
        "task_tokens": task_lm.total_tokens,
        "reflection_tokens": reflection_lm.total_tokens,
        "test_eval_tokens": test_lm.total_tokens,
        "task_calls": task_lm.call_count,
        "reflection_calls": reflection_lm.call_count,
        "task_error_count": task_lm.errors,
        "reflection_error_count": reflection_lm.errors,
        "rounds_completed": runtime.round_num,
        "rollouts_consumed": runtime.rollout_counter.value(),
        "termination_reason": termination,
        "val_score_trajectory": [],  # TODO: hook into core.py's per-round logging
    }

    # Try to load checkpoint for per-round trajectory
    from iso_harness.optimizer.checkpoint import load_checkpoint
    ckpt = load_checkpoint(run_id, run_dir=str(run_dir))
    if ckpt:
        metrics["final_pool_size"] = len(ckpt.get("pool", []))
        metrics["final_top3_mean"] = ckpt.get("prev_top3_mean", 0.0)

    # Save results
    with open(run_dir / "result.json", "w") as f:
        json.dump(result_dict, f, indent=2)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    console.print(f"  Saved to {run_dir.relative_to(PROJECT_ROOT)}")
    return result_dict


def main():
    parser = argparse.ArgumentParser(description="Run ISO optimization variants on raycluster")
    parser.add_argument("--variant", nargs="+", default=ALL_VARIANTS, help="ISO variants to run")
    parser.add_argument("--benchmark", nargs="+", default=BENCHMARKS, help="Benchmarks to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Random seeds")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Override runs directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    runs_dir = args.runs_dir or (PROJECT_ROOT / "runs" / MODEL_TAG)
    runs_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]=" * 60)
    console.print("[bold]Raycluster ISO Optimization")
    console.print(f"[bold]Model: {MODEL_FULL_NAME} ({MODEL_TAG})")
    console.print(f"[bold]Endpoint: {INFERENCE_BASE_URL}")
    console.print(f"[bold]Variants: {args.variant}")
    console.print(f"[bold]Benchmarks: {args.benchmark}")
    console.print(f"[bold]Seeds: {args.seeds}")
    total_runs = len(args.variant) * len(args.benchmark) * len(args.seeds)
    console.print(f"[bold]Total runs: {total_runs}")
    console.print("[bold]=" * 60)

    # Connectivity check
    try:
        resp = requests.get(f"{INFERENCE_BASE_URL}/models", timeout=10)
        resp.raise_for_status()
        console.print("[green]API reachable[/green]\n")
    except Exception as e:
        console.print(f"[bold red]API not reachable: {e}[/bold red]")
        sys.exit(1)

    # Run all variant × benchmark × seed combinations
    all_results = []
    total_t0 = time.time()

    for variant in args.variant:
        if variant not in VARIANT_FACTORIES:
            console.print(f"[red]Unknown variant: {variant}. Options: {ALL_VARIANTS}[/red]")
            continue
        for benchmark in args.benchmark:
            for seed in args.seeds:
                result = run_iso_single(variant, benchmark, seed, runs_dir)
                all_results.append(result)

    # Final summary
    total_time = time.time() - total_t0
    console.print(f"\n[bold]{'=' * 60}")
    console.print(f"[bold]ISO COMPLETE — {len(all_results)} runs in {total_time:.0f}s ({total_time/3600:.1f}h)")
    console.print(f"[bold]{'=' * 60}\n")

    console.print("[bold]Results Summary:[/bold]")
    for r in all_results:
        bm = r["metadata"]["benchmark"]
        seed = r["metadata"]["seed"]
        variant = r["metadata"]["variant"]
        score = r["test_score"]
        rounds = r["rounds_completed"]
        console.print(f"  {variant:8s} {bm:12s} seed={seed:4d}  score={score:.4f}  rounds={rounds}")


if __name__ == "__main__":
    main()
