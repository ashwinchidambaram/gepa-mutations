"""Regression test for remove_dominated_programs performance.

The original O(n²)–O(n³) while-loop timed out at iter 15-23 with many
candidates, causing GEPA to stall. The replacement O(n·d) essential-set
algorithm must handle large inputs efficiently.
"""

import random
import time
import sys
import pathlib

# Import directly from gepa submodule
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "gepa" / "src"))
from gepa.gepa_utils import remove_dominated_programs


def test_pareto_scales_with_1000_programs():
    """O(n²) on n=1000 takes >10s; O(n·d) completes in <2s."""
    rng = random.Random(42)
    n_programs, n_examples = 1000, 200
    fronts = {}
    for ex in range(n_examples):
        k = rng.randint(3, 8)
        fronts[ex] = set(rng.sample(range(n_programs), k=min(k, n_programs)))
    scores = {p: rng.random() for p in range(n_programs)}

    t0 = time.monotonic()
    result = remove_dominated_programs(fronts, scores=scores)
    elapsed = time.monotonic() - t0

    assert elapsed < 2.0, f"Took {elapsed:.1f}s (expected <2s)"
    # Every front must still have at least one program
    for ex, front in result.items():
        if fronts[ex]:  # only check non-empty input fronts
            assert len(front) > 0, f"Front {ex} became empty after dominance removal"


def test_pareto_correctness_small():
    """Verify correctness on a small, hand-checkable example."""
    # 3 programs, 3 examples
    # Program 0: appears in fronts {0, 1}
    # Program 1: appears in fronts {1, 2}
    # Program 2: appears in fronts {0, 2}
    fronts = {
        0: {0, 2},
        1: {0, 1},
        2: {1, 2},
    }
    scores = {0: 0.9, 1: 0.5, 2: 0.7}
    result = remove_dominated_programs(fronts, scores=scores)

    # All three programs are needed (no single program covers all fronts)
    # At minimum, each front must remain non-empty
    for ex in fronts:
        assert len(result[ex]) > 0

    # All returned programs must be a subset of the original front
    for ex in fronts:
        assert result[ex].issubset(fronts[ex])


def test_pareto_single_program_per_front():
    """When each front has exactly one program, nothing should be removed."""
    fronts = {0: {0}, 1: {1}, 2: {2}}
    scores = {0: 0.3, 1: 0.9, 2: 0.5}
    result = remove_dominated_programs(fronts, scores=scores)
    assert result == fronts


def test_pareto_one_program_dominates_all():
    """A program that appears in every front should survive; others may be removed."""
    fronts = {0: {0, 1}, 1: {0, 2}, 2: {0, 3}}
    scores = {0: 0.9, 1: 0.1, 2: 0.2, 3: 0.3}
    result = remove_dominated_programs(fronts, scores=scores)

    # Program 0 covers all fronts, so it must survive
    for ex in fronts:
        assert 0 in result[ex]
