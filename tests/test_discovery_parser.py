"""Test that discovery parser handles good/malformed/short outputs correctly."""
import importlib.util
import pathlib
import sys
import types
import pytest

# Stub gepa_mutations dependencies that colony.py imports transitively.
# Only install stubs if the real implementation isn't already loaded (avoids
# clobbering MetricsCollector for test_trajectory_point.py when run together).
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

# Load colony.py directly (bypasses slime_mold/__init__.py → runner.py chain)
_colony_path = (
    pathlib.Path(__file__).parent.parent
    / "methods" / "slime_mold" / "slime_mold" / "colony.py"
)
_spec = importlib.util.spec_from_file_location("slime_mold.colony", _colony_path)
assert _spec is not None and _spec.loader is not None
_colony_mod = importlib.util.module_from_spec(_spec)
sys.modules["slime_mold.colony"] = _colony_mod
_spec.loader.exec_module(_colony_mod)
_parse_discovered_skills = _colony_mod._parse_discovered_skills  # type: ignore[attr-defined]
PRESCRIBED_STRATEGIES = _colony_mod.PRESCRIBED_STRATEGIES  # type: ignore[attr-defined]


GOOD_OUTPUT = """1. Multi-hop entity tracking: The ability to chain facts across multiple paragraphs to arrive at an answer. Failure: stops at first hop, answering from a single paragraph.
2. Distractor resistance: The ability to ignore irrelevant information and focus on the paragraph containing the actual answer. Failure: answers from the wrong paragraph.
3. Implicit relationship inference: The ability to connect entities when the relationship is not explicitly stated. Failure: refuses to answer or says 'not enough information'.
4. Numeric comparison: The ability to compare quantities across entities. Failure: gets confused when units differ or quantities are implicit.
5. Temporal reasoning: The ability to reason about sequences of events. Failure: confuses cause and effect."""


MALFORMED_OUTPUT = """Here are some skills:
- Multi-hop reasoning
- Distractor handling
Skills require practice to develop."""


SHORT_OUTPUT = """1. Multi-hop entity tracking: Chains facts across paragraphs. Failure: stops at first hop.
2. Distractor resistance: Ignores irrelevant paragraphs. Failure: wrong paragraph."""


def test_parse_good_output():
    skills = _parse_discovered_skills(GOOD_OUTPUT)
    assert len(skills) == 5
    assert skills[0].name.lower() == "multi-hop entity tracking" or "multi-hop" in skills[0].name.lower()
    # Each skill should have a description and failure pattern
    assert all(s.description for s in skills)
    assert all(s.failure_pattern for s in skills)


def test_parse_short_output():
    """Parser returns what it finds, even if less than k."""
    skills = _parse_discovered_skills(SHORT_OUTPUT)
    assert len(skills) == 2  # only 2 skills in output


def test_parse_malformed_returns_empty_or_few():
    """Parser returns empty list (or <k) on malformed output."""
    skills = _parse_discovered_skills(MALFORMED_OUTPUT)
    assert len(skills) <= 2  # bullet list may parse as 2 or fewer skills


def test_prescribed_strategies_count():
    """PRESCRIBED_STRATEGIES should have exactly 8 entries."""
    assert len(PRESCRIBED_STRATEGIES) == 8


def test_prescribed_strategies_structure():
    """Each prescribed strategy should have name and description."""
    for s in PRESCRIBED_STRATEGIES:
        assert s.name
        assert s.description
        assert s.source == "prescribed"
