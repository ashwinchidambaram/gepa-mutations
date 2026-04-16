"""Test donor selection for cross-pollination with cross-strategy preference."""
import importlib.util
import pathlib
import sys
import types
import pytest

# Stub gepa_mutations dependencies so colony.py can be imported standalone.
# We only add stubs for modules that don't already have the real implementation loaded.
# This avoids contaminating test_trajectory_point.py which imports the real MetricsCollector.
import sys as _sys
import types as _types

def _needs_stub(mod_name: str, attr: str) -> bool:
    """Return True if mod_name is absent or its attr is missing/is bare object."""
    mod = _sys.modules.get(mod_name)
    if mod is None:
        return True
    val = getattr(mod, attr, None)
    return val is None or val is object

if _needs_stub("gepa_mutations.metrics.collector", "MetricsCollector"):
    for _mod_name in [
        "gepa_mutations",
        "gepa_mutations.metrics",
        "gepa_mutations.metrics.collector",
        "gepa_mutations.metrics.standalone_eval",
    ]:
        if _mod_name not in _sys.modules:
            _sys.modules[_mod_name] = _types.ModuleType(_mod_name)
    _sys.modules["gepa_mutations.metrics.collector"].MetricsCollector = object  # type: ignore[attr-defined]
    _sys.modules["gepa_mutations.metrics.standalone_eval"].evaluate_prompt = lambda *_a, **_kw: None  # type: ignore[attr-defined]

_colony_path = (
    pathlib.Path(__file__).parent.parent
    / "methods" / "slime_mold" / "slime_mold" / "colony.py"
)
_spec = importlib.util.spec_from_file_location("slime_mold.colony", _colony_path)
assert _spec is not None and _spec.loader is not None
_colony_mod = importlib.util.module_from_spec(_spec)
sys.modules["slime_mold.colony"] = _colony_mod
_spec.loader.exec_module(_colony_mod)
find_donor = _colony_mod.find_donor  # type: ignore[attr-defined]


def test_no_donor_available():
    """No candidate covers any survivor failure => returns None."""
    survivor_idx = 0
    survivor_strategy = "decomposition"
    failure_matrix = {0: {100, 101}, 1: {100, 101}, 2: {100, 101}}  # all fail same
    per_example_scores = {
        0: {100: 0.0, 101: 0.0},
        1: {100: 0.0, 101: 0.0},
        2: {100: 0.0, 101: 0.0},
    }
    strategies = {0: "decomposition", 1: "analogy", 2: "abstraction"}
    result = find_donor(survivor_idx, survivor_strategy, failure_matrix, per_example_scores, strategies, threshold=0.5)
    assert result is None


def test_single_donor_covers_all():
    """One candidate covers all failures => that one is selected."""
    survivor_idx = 0
    survivor_strategy = "decomposition"
    failure_matrix = {0: {100, 101}, 1: set()}  # candidate 1 passed on both
    per_example_scores = {
        0: {100: 0.0, 101: 0.0},
        1: {100: 0.8, 101: 0.9},
    }
    strategies = {0: "decomposition", 1: "analogy"}
    result = find_donor(survivor_idx, survivor_strategy, failure_matrix, per_example_scores, strategies, threshold=0.5)
    assert result is not None
    assert result["donor_candidate_idx"] == 1
    assert result["donor_strategy"] == "analogy"
    assert result["shared_failures_covered"] == 2
    assert result["cross_strategy"] is True


def test_cross_strategy_wins_over_same_strategy():
    """If two candidates tie on coverage, cross-strategy wins."""
    survivor_idx = 0
    survivor_strategy = "decomposition"
    failure_matrix = {0: {100, 101}, 1: set(), 2: set()}
    # Candidate 1 is same strategy (decomposition). Candidate 2 is different (analogy).
    per_example_scores = {
        0: {100: 0.0, 101: 0.0},
        1: {100: 0.9, 101: 0.9},  # higher score
        2: {100: 0.8, 101: 0.8},  # lower score but cross-strategy
    }
    strategies = {0: "decomposition", 1: "decomposition", 2: "analogy"}
    result = find_donor(survivor_idx, survivor_strategy, failure_matrix, per_example_scores, strategies, threshold=0.5)
    assert result is not None
    assert result["donor_candidate_idx"] == 2, "Cross-strategy should win even with lower score"


def test_tie_on_cross_strategy_higher_score_wins():
    """Two cross-strategy candidates: higher score wins."""
    survivor_idx = 0
    survivor_strategy = "decomposition"
    failure_matrix = {0: {100, 101}, 1: set(), 2: set()}
    per_example_scores = {
        0: {100: 0.0, 101: 0.0},
        1: {100: 0.7, 101: 0.7},
        2: {100: 0.9, 101: 0.9},  # higher score
    }
    strategies = {0: "decomposition", 1: "analogy", 2: "analogy"}
    result = find_donor(survivor_idx, survivor_strategy, failure_matrix, per_example_scores, strategies, threshold=0.5)
    assert result is not None
    assert result["donor_candidate_idx"] == 2


def test_tie_on_score_lower_idx_wins():
    """Two cross-strategy candidates with same score: lower idx wins (deterministic)."""
    survivor_idx = 0
    survivor_strategy = "decomposition"
    failure_matrix = {0: {100, 101}, 1: set(), 2: set()}
    per_example_scores = {
        0: {100: 0.0, 101: 0.0},
        1: {100: 0.9, 101: 0.9},
        2: {100: 0.9, 101: 0.9},  # tied with 1
    }
    strategies = {0: "decomposition", 1: "analogy", 2: "analogy"}
    result = find_donor(survivor_idx, survivor_strategy, failure_matrix, per_example_scores, strategies, threshold=0.5)
    assert result is not None
    assert result["donor_candidate_idx"] == 1, "Lower idx should win tiebreaker"


def test_same_strategy_fallback_when_no_cross():
    """If no cross-strategy donor, fall back to best same-strategy."""
    survivor_idx = 0
    survivor_strategy = "decomposition"
    failure_matrix = {0: {100, 101}, 1: set()}
    per_example_scores = {
        0: {100: 0.0, 101: 0.0},
        1: {100: 0.9, 101: 0.9},
    }
    strategies = {0: "decomposition", 1: "decomposition"}  # both same strategy
    result = find_donor(survivor_idx, survivor_strategy, failure_matrix, per_example_scores, strategies, threshold=0.5)
    assert result is not None
    assert result["donor_candidate_idx"] == 1
    assert result["cross_strategy"] is False


def test_partial_coverage():
    """Donor that covers some (not all) survivor failures still selectable."""
    survivor_idx = 0
    survivor_strategy = "decomposition"
    failure_matrix = {0: {100, 101, 102}, 1: {102}}  # 1 covers 100, 101 (fails only on 102)
    per_example_scores = {
        0: {100: 0.0, 101: 0.0, 102: 0.0},
        1: {100: 0.9, 101: 0.9, 102: 0.0},
    }
    strategies = {0: "decomposition", 1: "analogy"}
    result = find_donor(survivor_idx, survivor_strategy, failure_matrix, per_example_scores, strategies, threshold=0.5)
    assert result is not None
    assert result["donor_candidate_idx"] == 1
    assert result["shared_failures_covered"] == 2  # survivor failed on {100,101,102}, donor passed on {100,101}
