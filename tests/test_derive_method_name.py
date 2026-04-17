"""Test that _derive_method_name produces unique names for all method variants."""

import importlib.util
import pathlib
import pytest

# Load naming.py directly to avoid triggering iso/__init__.py which
# imports runner.py which requires the full gepa_mutations environment.
_naming_path = (
    pathlib.Path(__file__).parent.parent
    / "methods" / "iso" / "iso" / "naming.py"
)
_spec = importlib.util.spec_from_file_location("iso.naming", _naming_path)
assert _spec is not None and _spec.loader is not None
_naming_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_naming_mod)
_derive_method_name = _naming_mod._derive_method_name  # type: ignore[attr-defined]


# All combinations and their expected names
TEST_CASES = [
    # (strategy_mode, k, mutation_mode, refresh_mode, expected_name)
    # Baselines
    ("personality", None, "blind", "none", "iso_personality"),
    ("personality", None, "crosspollin", "none", "iso_personality_crosspollin"),
    ("prescribed8", None, "blind", "none", "iso_prescribed8"),
    # K=3
    ("inductive", 3, "blind", "none", "iso_inductive_k3"),
    ("inductive", 3, "crosspollin", "none", "iso_inductive_k3_crosspollin"),
    ("inductive", 3, "crosspollin", "expand", "iso_inductive_k3_refresh_expand"),
    ("inductive", 3, "crosspollin", "replace", "iso_inductive_k3_refresh_replace"),
    # K=5
    ("inductive", 5, "blind", "none", "iso_inductive_k5"),
    ("inductive", 5, "crosspollin", "none", "iso_inductive_k5_crosspollin"),
    ("inductive", 5, "crosspollin", "expand", "iso_inductive_k5_refresh_expand"),
    ("inductive", 5, "crosspollin", "replace", "iso_inductive_k5_refresh_replace"),
    # K=adaptive
    ("inductive", None, "blind", "none", "iso_inductive_kadaptive"),
    ("inductive", None, "crosspollin", "none", "iso_inductive_kadaptive_crosspollin"),
    ("inductive", None, "crosspollin", "expand", "iso_inductive_kadaptive_refresh_expand"),
    ("inductive", None, "crosspollin", "replace", "iso_inductive_kadaptive_refresh_replace"),
]


@pytest.mark.parametrize("strategy_mode,k,mutation_mode,refresh_mode,expected", TEST_CASES)
def test_derive_method_name(strategy_mode, k, mutation_mode, refresh_mode, expected):
    result = _derive_method_name(strategy_mode, k, mutation_mode, refresh_mode)
    assert result == expected, (
        f"Expected '{expected}' for "
        f"(strategy_mode={strategy_mode}, k={k}, "
        f"mutation_mode={mutation_mode}, refresh_mode={refresh_mode}), got '{result}'"
    )


def test_all_names_unique():
    """Ensure all 14 combinations produce unique directory names."""
    names = [
        _derive_method_name(sm, k, mm, rm)
        for sm, k, mm, rm, _ in TEST_CASES
    ]
    assert len(names) == len(set(names)), (
        f"Duplicate method names detected: {names}"
    )
