"""Test failure matrix construction for cross-pollination."""
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
    # If the attr is just builtins.object (our own stub) or None, needs stub
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
    / "methods" / "iso" / "iso" / "colony.py"
)
_spec = importlib.util.spec_from_file_location("iso.colony", _colony_path)
assert _spec is not None and _spec.loader is not None
_colony_mod = importlib.util.module_from_spec(_spec)
sys.modules["iso.colony"] = _colony_mod
_spec.loader.exec_module(_colony_mod)
build_failure_matrix = _colony_mod.build_failure_matrix  # type: ignore[attr-defined]


def test_empty_matrix():
    """Empty scores => empty matrix."""
    result = build_failure_matrix({}, threshold=0.5)
    assert result == {}


def test_all_pass():
    """All scores >= threshold => empty failure sets."""
    scores = {
        0: {100: 1.0, 101: 0.8, 102: 0.6},
        1: {100: 0.7, 101: 0.9, 102: 0.5},
    }
    result = build_failure_matrix(scores, threshold=0.5)
    assert result == {0: set(), 1: set()}


def test_all_fail():
    """All scores < threshold => all examples in failure set."""
    scores = {
        0: {100: 0.1, 101: 0.0, 102: 0.3},
    }
    result = build_failure_matrix(scores, threshold=0.5)
    assert result == {0: {100, 101, 102}}


def test_threshold_edge_0_5_passes():
    """Score exactly 0.5 must pass (>= threshold)."""
    scores = {0: {100: 0.5, 101: 0.499}}
    result = build_failure_matrix(scores, threshold=0.5)
    assert result == {0: {101}}


def test_per_candidate_independent():
    """Each candidate has independent failure set."""
    scores = {
        0: {100: 1.0, 101: 0.0},
        1: {100: 0.0, 101: 1.0},
    }
    result = build_failure_matrix(scores, threshold=0.5)
    assert result == {0: {101}, 1: {100}}


def test_custom_threshold():
    """Threshold parameter should work."""
    scores = {0: {100: 0.7, 101: 0.6}}
    # Threshold 0.65 => 101 fails
    result = build_failure_matrix(scores, threshold=0.65)
    assert result == {0: {101}}
