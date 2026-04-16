"""Test that _derive_method_name produces unique names for all 14 method variants."""

import importlib.util
import pathlib
import pytest

# Import naming.py directly to avoid triggering slime_mold/__init__.py
# (which requires the full gepa_mutations dependency stack not available in test env)
_naming_path = (
    pathlib.Path(__file__).parent.parent
    / "methods" / "slime_mold" / "slime_mold" / "naming.py"
)
_spec = importlib.util.spec_from_file_location("slime_mold.naming", _naming_path)
_naming_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_naming_mod)
_derive_method_name = _naming_mod._derive_method_name


# All 14 combinations and their expected names
TEST_CASES = [
    # (strategy_mode, k, mutation_mode, refresh_mode, expected_name)
    # Baselines
    ("personality", None, "blind", "none", "slime_mold"),
    ("prescribed8", None, "blind", "none", "slime_mold_prescribed8"),
    # K=3
    ("inductive", 3, "blind", "none", "slime_mold_inductive_k3"),
    ("inductive", 3, "crosspollin", "none", "slime_mold_inductive_k3_crosspollin"),
    ("inductive", 3, "crosspollin", "expand", "slime_mold_inductive_k3_refresh_expand"),
    ("inductive", 3, "crosspollin", "replace", "slime_mold_inductive_k3_refresh_replace"),
    # K=5
    ("inductive", 5, "blind", "none", "slime_mold_inductive_k5"),
    ("inductive", 5, "crosspollin", "none", "slime_mold_inductive_k5_crosspollin"),
    ("inductive", 5, "crosspollin", "expand", "slime_mold_inductive_k5_refresh_expand"),
    ("inductive", 5, "crosspollin", "replace", "slime_mold_inductive_k5_refresh_replace"),
    # K=adaptive
    ("inductive", None, "blind", "none", "slime_mold_inductive_kadaptive"),
    ("inductive", None, "crosspollin", "none", "slime_mold_inductive_kadaptive_crosspollin"),
    ("inductive", None, "crosspollin", "expand", "slime_mold_inductive_kadaptive_refresh_expand"),
    ("inductive", None, "crosspollin", "replace", "slime_mold_inductive_kadaptive_refresh_replace"),
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
