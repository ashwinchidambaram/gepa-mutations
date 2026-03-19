"""Statistical analysis for comparing experiment results against paper baselines.

Provides bootstrap CIs, Cohen's d effect sizes, and reproduction verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from gepa_mutations.config import PAPER_BASELINES


# Per-benchmark tolerance bands (in percentage points)
TOLERANCE = {
    "hotpotqa": 3.0,
    "ifbench": 2.0,
    "hover": 3.0,
    "pupa": 3.0,
    "aime": 4.0,
    "livebench": 3.0,
}

ReproductionVerdict = Literal["STRONG_MATCH", "ACCEPTABLE", "FAILED"]


@dataclass
class BenchmarkStats:
    """Statistical summary for a single benchmark."""

    benchmark: str
    scores: list[float]  # Raw test scores (0-1 scale)
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    paper_score: float
    diff_pp: float  # Difference in percentage points
    tolerance: float
    within_tolerance: bool
    cohens_d: float | None


@dataclass
class ReproductionReport:
    """Overall reproduction verdict with per-benchmark details."""

    verdict: ReproductionVerdict
    aggregate_mean: float
    aggregate_paper: float
    aggregate_diff_pp: float
    benchmarks: list[BenchmarkStats]
    num_within_tolerance: int
    total_benchmarks: int


def bootstrap_ci(
    scores: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns (mean, ci_lower, ci_upper) in the same scale as input scores.
    """
    rng = np.random.RandomState(seed)
    arr = np.array(scores)
    n = len(arr)

    if n <= 1:
        m = float(arr.mean())
        return m, m, m

    bootstrap_means = np.array([
        arr[rng.randint(0, n, size=n)].mean() for _ in range(n_bootstrap)
    ])

    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(bootstrap_means, alpha * 100))
    ci_upper = float(np.percentile(bootstrap_means, (1 - alpha) * 100))
    mean = float(arr.mean())

    return mean, ci_lower, ci_upper


def cohens_d(scores: list[float], paper_score: float) -> float | None:
    """Compute Cohen's d effect size vs paper score.

    Returns None if insufficient data.
    """
    if len(scores) < 2:
        return None

    arr = np.array(scores)
    std = float(arr.std(ddof=1))
    if std == 0:
        return None

    return float((arr.mean() - paper_score) / std)


def analyze_benchmark(
    benchmark: str,
    scores: list[float],
    model: str = "qwen3-8b",
) -> BenchmarkStats:
    """Analyze results for a single benchmark against paper baselines.

    Args:
        benchmark: Benchmark name.
        scores: List of test scores (0-1 scale) from multiple seeds.
        model: Model key in PAPER_BASELINES.

    Returns:
        BenchmarkStats with statistical analysis.
    """
    paper = PAPER_BASELINES.get(model, {}).get("gepa", {})
    paper_score = paper.get(benchmark, 0.0)

    mean, ci_lower, ci_upper = bootstrap_ci(scores)
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

    # Convert to percentage points for comparison
    mean_pct = mean * 100
    diff_pp = mean_pct - paper_score
    tol = TOLERANCE.get(benchmark, 3.0)

    return BenchmarkStats(
        benchmark=benchmark,
        scores=scores,
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        paper_score=paper_score,
        diff_pp=diff_pp,
        tolerance=tol,
        within_tolerance=abs(diff_pp) <= tol,
        cohens_d=cohens_d(scores, paper_score / 100),
    )


def reproduction_verdict(
    benchmark_stats: list[BenchmarkStats],
) -> ReproductionReport:
    """Compute overall reproduction verdict.

    Criteria:
    - STRONG_MATCH: All benchmarks within tolerance, aggregate within 1pp
    - ACCEPTABLE: 5/6+ within tolerance, aggregate within 2pp
    - FAILED: 3+ outside tolerance OR aggregate differs by 3+pp
    """
    num_within = sum(1 for s in benchmark_stats if s.within_tolerance)
    total = len(benchmark_stats)

    # Compute aggregate
    if benchmark_stats:
        agg_mean = sum(s.mean * 100 for s in benchmark_stats) / total
    else:
        agg_mean = 0.0

    paper = PAPER_BASELINES.get("qwen3-8b", {}).get("gepa", {})
    agg_paper = paper.get("aggregate", 0.0)
    agg_diff = agg_mean - agg_paper

    # Determine verdict
    if num_within == total and abs(agg_diff) <= 1.0:
        verdict: ReproductionVerdict = "STRONG_MATCH"
    elif num_within >= total - 1 and abs(agg_diff) <= 2.0:
        verdict = "ACCEPTABLE"
    else:
        verdict = "FAILED"

    return ReproductionReport(
        verdict=verdict,
        aggregate_mean=agg_mean,
        aggregate_paper=agg_paper,
        aggregate_diff_pp=agg_diff,
        benchmarks=benchmark_stats,
        num_within_tolerance=num_within,
        total_benchmarks=total,
    )
