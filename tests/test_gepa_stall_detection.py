"""Regression tests for the GEPA Pareto dominance stall mitigation.

Bug: ``remove_dominated_programs()`` in ``gepa/src/gepa/gepa_utils.py`` has
O(n^2)-O(n^3) complexity.  By iteration 15-23, it stalls for minutes with zero
log output — invisible to the orchestrator's stall detector until 30 min later.

Fix: a self-limiting ``_DOMINANCE_TIMEOUT`` that breaks out of the while-loop
early, returning a conservatively over-inclusive Pareto front.
"""

from __future__ import annotations

import logging

import pytest

from gepa import gepa_utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pareto_front(n_programs: int, n_fronts: int) -> dict[str, set[int]]:
    """Construct a synthetic Pareto front mapping.

    Each front (keyed by "val_0", "val_1", ...) contains a random-ish subset
    of program indices, ensuring every program appears in at least one front.
    """
    import random

    rng = random.Random(42)
    programs = list(range(n_programs))
    fronts: dict[str, set[int]] = {}

    # Ensure every program appears in at least one front
    for p in programs:
        key = f"val_{rng.randint(0, n_fronts - 1)}"
        fronts.setdefault(key, set()).add(p)

    # Fill remaining fronts with random subsets
    for i in range(n_fronts):
        key = f"val_{i}"
        subset_size = rng.randint(1, max(1, n_programs // 3))
        fronts.setdefault(key, set()).update(rng.sample(programs, min(subset_size, len(programs))))

    return fronts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_small_front_completes_fast():
    """A small front should complete well under the timeout."""
    import time

    front = _make_pareto_front(n_programs=5, n_fronts=10)
    t0 = time.monotonic()
    result = gepa_utils.remove_dominated_programs(front)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0, f"Small front took {elapsed:.2f}s — should be <1s"
    # Result should be a valid subset
    for val_id, progs in result.items():
        assert progs.issubset(front[val_id])


def test_timeout_breaks_out_of_long_computation(monkeypatch, caplog):
    """With a tiny timeout, the function should break early and log a warning."""
    # Force immediate timeout
    monkeypatch.setattr(gepa_utils, "_DOMINANCE_TIMEOUT", 0.0)

    # Create a front large enough that the while-loop would iterate multiple times
    front = _make_pareto_front(n_programs=30, n_fronts=100)

    with caplog.at_level(logging.WARNING, logger="gepa.gepa_utils"):
        result = gepa_utils.remove_dominated_programs(front)

    # Should have logged a timeout warning
    assert any("timed out" in rec.message for rec in caplog.records), (
        f"Expected 'timed out' warning, got: {[r.message for r in caplog.records]}"
    )

    # Result must still be a valid mapping (subset of original per front)
    for val_id in front:
        assert val_id in result
        assert result[val_id].issubset(front[val_id])


def test_result_valid_after_timeout(monkeypatch):
    """After a forced timeout, every front must still have >= 1 program."""
    monkeypatch.setattr(gepa_utils, "_DOMINANCE_TIMEOUT", 0.0)

    front = _make_pareto_front(n_programs=20, n_fronts=50)
    result = gepa_utils.remove_dominated_programs(front)

    # The key safety invariant: every non-empty original front has at least
    # one surviving program in the result.
    for val_id, original_progs in front.items():
        if original_progs:
            assert len(result[val_id]) >= 1, (
                f"Front {val_id} lost all programs after timeout"
            )


@pytest.mark.parametrize("n_programs,n_fronts", [
    (3, 5),
    (10, 20),
    (5, 50),
])
def test_various_sizes_produce_valid_output(n_programs, n_fronts):
    """Verify output validity across different front sizes."""
    front = _make_pareto_front(n_programs=n_programs, n_fronts=n_fronts)
    result = gepa_utils.remove_dominated_programs(front)

    # Every result front is a subset of the original
    for val_id in front:
        assert result[val_id].issubset(front[val_id])

    # Every non-empty original front has at least one surviving program
    for val_id, original_progs in front.items():
        if original_progs:
            assert len(result[val_id]) >= 1
