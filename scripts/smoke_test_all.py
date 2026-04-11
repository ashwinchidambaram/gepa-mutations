"""Smoke test all methods × all benchmarks on RayCluster (gpt-oss-120b).

Runs each method on each benchmark with subset=5 and max_metric_calls=50
for quick validation. Logs all output and produces a summary table.
"""

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "methods" / "best_of_k"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "methods" / "contrastive_reflection"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "methods" / "failure_stratified_k"))

BENCHMARKS = ["hotpotqa", "ifbench", "hover", "pupa", "aime", "livebench"]
SEED = 42
SUBSET = 5
MAX_METRIC_CALLS = 50

LOG_DIR = Path("logs/smoke_tests")
LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SmokeResult:
    method: str
    benchmark: str
    status: str  # "PASS", "FAIL", "ERROR"
    val_score: float | None = None
    test_score: float | None = None
    wall_clock: float = 0.0
    error: str = ""
    candidates: int = 0


def run_gepa_baseline(benchmark: str) -> SmokeResult:
    """Run baseline GEPA (K=1, paper defaults)."""
    from gepa_mutations.runner.experiment import ExperimentRunner

    runner = ExperimentRunner()
    result = runner.run(
        benchmark=benchmark,
        seed=SEED,
        subset=SUBSET,
        max_metric_calls=MAX_METRIC_CALLS,
    )
    return SmokeResult(
        method="gepa_baseline",
        benchmark=benchmark,
        status="PASS",
        val_score=result.val_score,
        test_score=result.test_score,
        wall_clock=result.wall_clock_seconds,
        candidates=len(result.all_candidates),
    )


def run_best_of_k(benchmark: str, k: int) -> SmokeResult:
    """Run best-of-K mutation."""
    from best_of_k.runner import run_best_of_k as _run
    from gepa_mutations.base import MutationConfig

    config = MutationConfig(
        mutation_name=f"best_of_k_K{k}",
        benchmark=benchmark,
        seed=SEED,
        subset=SUBSET,
        mutation_candidates=k,
        max_metric_calls=MAX_METRIC_CALLS,
    )
    result = _run(config, k=k)
    return SmokeResult(
        method=f"best_of_k_K{k}",
        benchmark=benchmark,
        status="PASS",
        val_score=result.val_score,
        test_score=result.test_score,
        wall_clock=result.wall_clock_seconds,
        candidates=len(result.all_candidates),
    )


def run_contrastive(benchmark: str) -> SmokeResult:
    """Run contrastive reflection mutation."""
    from contrastive_reflection.runner import run_contrastive_reflection as _run

    result = _run(
        benchmark=benchmark,
        seed=SEED,
        subset=SUBSET,
        max_metric_calls=MAX_METRIC_CALLS,
    )
    return SmokeResult(
        method="contrastive_reflection",
        benchmark=benchmark,
        status="PASS",
        val_score=result.val_score,
        test_score=result.test_score,
        wall_clock=result.wall_clock_seconds,
        candidates=len(result.all_candidates),
    )


def run_failure_stratified(benchmark: str, k: int) -> SmokeResult:
    """Run failure-stratified-K mutation."""
    from failure_stratified_k.runner import run_failure_stratified_k as _run
    from gepa_mutations.base import MutationConfig

    config = MutationConfig(
        mutation_name=f"failure_stratified_k_K{k}",
        benchmark=benchmark,
        seed=SEED,
        subset=SUBSET,
        mutation_candidates=k,
        use_failure_stratified_k=True,
        max_metric_calls=MAX_METRIC_CALLS,
    )
    result = _run(config, k=k, use_failure_stratified_k=True)
    return SmokeResult(
        method=f"failure_stratified_k_K{k}",
        benchmark=benchmark,
        status="PASS",
        val_score=result.val_score,
        test_score=result.test_score,
        wall_clock=result.wall_clock_seconds,
        candidates=len(result.all_candidates),
    )


# All methods to test
METHODS = [
    ("gepa_baseline", lambda bm: run_gepa_baseline(bm)),
    ("best_of_k_K3", lambda bm: run_best_of_k(bm, k=3)),
    ("best_of_k_K5", lambda bm: run_best_of_k(bm, k=5)),
    ("contrastive_reflection", lambda bm: run_contrastive(bm)),
    ("failure_stratified_k_K3", lambda bm: run_failure_stratified(bm, k=3)),
    ("failure_stratified_k_K5", lambda bm: run_failure_stratified(bm, k=5)),
]


def run_single_test(method_name: str, runner_fn, benchmark: str) -> SmokeResult:
    """Run a single smoke test with error handling and logging."""
    log_file = LOG_DIR / f"{method_name}_{benchmark}.log"

    # Redirect stdout/stderr to log file AND console
    import io

    old_stdout, old_stderr = sys.stdout, sys.stderr

    class Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
            return len(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    with open(log_file, "w") as f:
        tee_out = Tee(old_stdout, f)
        tee_err = Tee(old_stderr, f)
        sys.stdout = tee_out
        sys.stderr = tee_err

        try:
            print(f"\n{'='*60}")
            print(f"SMOKE TEST: {method_name} / {benchmark}")
            print(f"Time: {datetime.now().isoformat()}")
            print(f"Params: subset={SUBSET}, seed={SEED}, max_metric_calls={MAX_METRIC_CALLS}")
            print(f"{'='*60}\n")

            result = runner_fn(benchmark)
            print(f"\n--- Result: {result.status} ---")
            print(f"Val score: {result.val_score:.4f}" if result.val_score is not None else "Val score: N/A")
            print(f"Test score: {result.test_score:.4f}" if result.test_score is not None else "Test score: N/A")
            print(f"Wall clock: {result.wall_clock:.1f}s")
            print(f"Candidates: {result.candidates}")
            return result

        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n--- ERROR ---\n{tb}")
            return SmokeResult(
                method=method_name,
                benchmark=benchmark,
                status="ERROR",
                error=str(e),
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def print_summary(results: list[SmokeResult]):
    """Print a summary table of all results."""
    print(f"\n{'='*100}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*100}")
    print(f"{'Method':<30} {'Benchmark':<12} {'Status':<8} {'Val':>8} {'Test':>8} {'Time':>8} {'Cands':>6}")
    print(f"{'-'*30} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for r in results:
        val = f"{r.val_score:.4f}" if r.val_score is not None else "N/A"
        test = f"{r.test_score:.4f}" if r.test_score is not None else "N/A"
        t = f"{r.wall_clock:.1f}s" if r.wall_clock else "N/A"
        print(f"{r.method:<30} {r.benchmark:<12} {r.status:<8} {val:>8} {test:>8} {t:>8} {r.candidates:>6}")

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status in ("FAIL", "ERROR"))
    print(f"\n{'='*100}")
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    if failed:
        print("\nFailed tests:")
        for r in results:
            if r.status != "PASS":
                print(f"  {r.method}/{r.benchmark}: {r.error[:100]}")
    print(f"{'='*100}")


def main():
    start = time.time()
    results: list[SmokeResult] = []

    total = len(METHODS) * len(BENCHMARKS)
    idx = 0

    for method_name, runner_fn in METHODS:
        for benchmark in BENCHMARKS:
            idx += 1
            print(f"\n[{idx}/{total}] Running {method_name} / {benchmark} ...")
            result = run_single_test(method_name, runner_fn, benchmark)
            results.append(result)
            # Print running tally
            p = sum(1 for r in results if r.status == "PASS")
            f = sum(1 for r in results if r.status != "PASS")
            print(f"  -> {result.status} (running: {p} passed, {f} failed)")

    elapsed = time.time() - start
    print_summary(results)
    print(f"\nTotal wall clock: {elapsed / 60:.1f} minutes")

    # Save results as JSON
    results_file = LOG_DIR / "smoke_test_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "total_seconds": elapsed,
                "params": {"subset": SUBSET, "seed": SEED, "max_metric_calls": MAX_METRIC_CALLS},
                "results": [
                    {
                        "method": r.method,
                        "benchmark": r.benchmark,
                        "status": r.status,
                        "val_score": r.val_score,
                        "test_score": r.test_score,
                        "wall_clock": r.wall_clock,
                        "candidates": r.candidates,
                        "error": r.error,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
