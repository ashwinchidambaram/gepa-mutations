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


# --- Legacy format (used as backward-compat fallback in the parser) ---

LEGACY_GOOD_OUTPUT = """1. Multi-hop entity tracking: The ability to chain facts across multiple paragraphs to arrive at an answer. Failure: stops at first hop, answering from a single paragraph.
2. Distractor resistance: The ability to ignore irrelevant information and focus on the paragraph containing the actual answer. Failure: answers from the wrong paragraph.
3. Implicit relationship inference: The ability to connect entities when the relationship is not explicitly stated. Failure: refuses to answer or says 'not enough information'.
4. Numeric comparison: The ability to compare quantities across entities. Failure: gets confused when units differ or quantities are implicit.
5. Temporal reasoning: The ability to reason about sequences of events. Failure: confuses cause and effect."""


MALFORMED_OUTPUT = """Here are some skills:
- Multi-hop reasoning
- Distractor handling
Skills require practice to develop."""


LEGACY_SHORT_OUTPUT = """1. Multi-hop entity tracking: Chains facts across paragraphs. Failure: stops at first hop.
2. Distractor resistance: Ignores irrelevant paragraphs. Failure: wrong paragraph."""


# --- New format (primary output of the rewritten discovery prompt) ---

NEW_FORMAT_GOOD_OUTPUT = """### Failure modes
- F1: Answers from the first paragraph that mentions the query entity, even when the target relation is in a later paragraph (Examples 2, 5).
- F2: Treats every named entity in the context as equally relevant, ignoring whether it appears as subject vs object of the target relation (Example 3).
- F3: Stops searching after finding one supporting sentence, missing a contradicting one elsewhere (Examples 1, 4).
- F4: Confuses similarly-named entities (e.g., two people sharing a surname) (Example 7).

### Skills
1. **Bridge-entity scratchpad** — Addresses: F1. Before answering, the model writes the (bridge entity, source paragraph) pair, then looks up the target relation in a different paragraph. Technique: Scratchpad.
2. **Relation-role decomposition** — Addresses: F2. The model labels each candidate entity with its grammatical role (subject, object) relative to the target relation before selecting one. Technique: Decomposition.
3. **Contradicting-evidence sweep** — Addresses: F3. After drafting an answer, the model scans for any paragraph that contradicts the draft and revises if found. Technique: Verification step.
4. **Disambiguating qualifier rule** — Addresses: F4. When two entities share a name, the model requires a disambiguating qualifier (date, role, location) before treating them as equivalent. Technique: Negative check.
5. **Source-span citation** — Addresses: F1, F3. Every fact in the answer cites the paragraph number it came from. Technique: Grounding citation.
"""


NEW_FORMAT_SHORT_OUTPUT = """### Failure modes
- F1: Answers using the first paragraph that mentions the query entity (Examples 2, 5).
- F2: Stops after finding one supporting sentence (Example 1).

### Skills
1. **Bridge-entity scratchpad** — Addresses: F1. Writes (entity, paragraph) pairs before answering. Technique: Scratchpad.
2. **Contradicting-evidence sweep** — Addresses: F2. Scans for contradicting paragraphs after drafting. Technique: Verification step.
"""


# --- Tests ---


def test_parse_new_format_good():
    """New-format parser extracts name, description, technique, and failure_pattern."""
    skills = _parse_discovered_skills(NEW_FORMAT_GOOD_OUTPUT)
    assert len(skills) == 5

    # First skill should have a clean name (no bold markers)
    first = skills[0]
    assert "*" not in first.name
    assert "bridge" in first.name.lower()

    # All skills should have technique populated
    assert all(s.technique for s in skills), [s.technique for s in skills]

    # Technique values should match the canonical list (case-insensitive)
    techniques = {s.technique.lower() for s in skills}
    assert "scratchpad" in techniques
    assert "decomposition" in techniques
    assert "verification step" in techniques

    # failure_pattern should be populated by lookup of F<n> references
    assert all(s.failure_pattern for s in skills), [s.failure_pattern for s in skills]
    assert "first paragraph" in skills[0].failure_pattern.lower()


def test_parse_new_format_multi_address():
    """Skills addressing multiple F<n>s should join their failure descriptions."""
    skills = _parse_discovered_skills(NEW_FORMAT_GOOD_OUTPUT)
    # Skill 5 addresses F1 AND F3
    citation_skill = next(s for s in skills if "citation" in s.name.lower())
    assert citation_skill.technique.lower() == "grounding citation"
    # failure_pattern should contain text from both F1 and F3
    assert "first paragraph" in citation_skill.failure_pattern.lower()
    assert "supporting sentence" in citation_skill.failure_pattern.lower()


def test_parse_new_format_short():
    """Parser returns what it finds, even if fewer than expected."""
    skills = _parse_discovered_skills(NEW_FORMAT_SHORT_OUTPUT)
    assert len(skills) == 2
    assert all(s.technique for s in skills)


def test_parse_legacy_format_still_works():
    """Backward-compat: old `1. name: desc. Failure: ...` format still parses."""
    skills = _parse_discovered_skills(LEGACY_GOOD_OUTPUT)
    assert len(skills) == 5
    assert "multi-hop" in skills[0].name.lower()
    # Each skill should have a description and failure pattern
    assert all(s.description for s in skills)
    assert all(s.failure_pattern for s in skills)
    # Legacy format has no technique field
    assert all(s.technique == "" for s in skills)


def test_parse_legacy_short_output():
    """Backward-compat: legacy parser returns what it finds."""
    skills = _parse_discovered_skills(LEGACY_SHORT_OUTPUT)
    assert len(skills) == 2


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
