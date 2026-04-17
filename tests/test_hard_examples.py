"""Test hard-example collection for refresh pass."""
import importlib.util
import pathlib
import sys
import types
import pytest

# Stub gepa_mutations dependencies
def _needs_stub(mod_name: str, attr: str) -> bool:
    mod = sys.modules.get(mod_name)
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
        if _mod_name not in sys.modules:
            sys.modules[_mod_name] = types.ModuleType(_mod_name)
    sys.modules["gepa_mutations.metrics.collector"].MetricsCollector = object  # type: ignore[attr-defined]
    sys.modules["gepa_mutations.metrics.standalone_eval"].evaluate_prompt = lambda *_a, **_kw: None  # type: ignore[attr-defined]

_colony_path = (
    pathlib.Path(__file__).parent.parent
    / "methods" / "iso" / "iso" / "colony.py"
)
_spec = importlib.util.spec_from_file_location("iso.colony", _colony_path)
assert _spec is not None and _spec.loader is not None
_colony_mod = importlib.util.module_from_spec(_spec)
sys.modules["iso.colony"] = _colony_mod
_spec.loader.exec_module(_colony_mod)
collect_hard_examples = _colony_mod.collect_hard_examples  # type: ignore[attr-defined]


def test_empty_matrix_returns_empty():
    """Empty failure matrix => empty list."""
    assert collect_hard_examples({}, n_candidates=0, threshold=0.7) == []


def test_all_fail_all_hard():
    """If all candidates fail on example X, it's hard."""
    # 10 candidates, all fail on example 42
    failure_matrix = {i: {42} for i in range(10)}
    result = collect_hard_examples(failure_matrix, n_candidates=10, threshold=0.7)
    assert 42 in result


def test_threshold_edge_exactly_at_boundary():
    """Example failed by exactly 70% of candidates is included (>= threshold)."""
    failure_matrix = {0: {1}, 1: {1}, 2: {1}, 3: {1}, 4: {1}, 5: {1}, 6: {1}, 7: set(), 8: set(), 9: set()}
    # 7/10 = 0.7 = 70%. Should be included at threshold=0.7.
    result = collect_hard_examples(failure_matrix, n_candidates=10, threshold=0.7)
    assert 1 in result


def test_threshold_below_not_hard():
    """Example failed by < threshold is NOT hard."""
    failure_matrix = {0: {1}, 1: {1}, 2: {1}, 3: {1}, 4: {1}, 5: {1}, 6: set(), 7: set(), 8: set(), 9: set()}
    # 6/10 = 60% < 70%
    result = collect_hard_examples(failure_matrix, n_candidates=10, threshold=0.7)
    assert 1 not in result


def test_threshold_1_0_never_selected():
    """Threshold 1.0 requires 100% failure => rare to select anything."""
    failure_matrix = {0: {1, 2}, 1: {1}, 2: {1, 2}}
    # example 1: 3/3 = 100% fail => selected
    # example 2: 2/3 = 66% fail => not selected
    result = collect_hard_examples(failure_matrix, n_candidates=3, threshold=1.0)
    assert 1 in result
    assert 2 not in result


def test_threshold_0_5_default():
    """Threshold 0.5 = half or more failed."""
    failure_matrix = {0: {1}, 1: {1}, 2: set(), 3: set()}
    # example 1: 2/4 = 50% fail => included at threshold=0.5
    result = collect_hard_examples(failure_matrix, n_candidates=4, threshold=0.5)
    assert 1 in result
