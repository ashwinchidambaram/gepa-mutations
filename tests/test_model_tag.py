"""Regression tests for _env_model_tag() in scripts/run_all_local.py.

Guards against the size-token collision bug where "4b" in m matched before
"14b" in m, causing Qwen3-14B to be mapped to the wrong tag "qwen3-4b".
"""

import importlib.util
import pathlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Load the script as a module without executing __main__ block.
# We must register the module in sys.modules before exec_module so that
# @dataclass (and similar decorator machinery) can look up cls.__module__.
# The function reads os.environ at *call time*, so we can extract it once
# and monkeypatch the env before each invocation.
# ---------------------------------------------------------------------------
_script_path = pathlib.Path(__file__).parent.parent / "scripts" / "run_all_local.py"
_spec = importlib.util.spec_from_file_location("run_all_local", _script_path)
assert _spec is not None and _spec.loader is not None
_run_all_mod = importlib.util.module_from_spec(_spec)
sys.modules["run_all_local"] = _run_all_mod  # required for @dataclass machinery
_spec.loader.exec_module(_run_all_mod)  # type: ignore[union-attr]
_env_model_tag = _run_all_mod._env_model_tag  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Parametrized test cases — covers all currently-known model variants.
# THE COLLISION CASE: "Qwen/Qwen3-14B" must return "qwen3-14b", not "qwen3-4b".
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_str,expected_tag", [
    # Qwen3 — ascending size order to expose any mis-ordered checks
    ("mlx-community/Qwen3-0.6B-4bit", "qwen3-0.6b"),
    ("mlx-community/Qwen3-1.7B-4bit", "qwen3-1.7b"),
    ("mlx-community/Qwen3-4B-4bit",   "qwen3-4b"),
    ("Qwen/Qwen3-8B",                  "qwen3-8b"),
    ("Qwen/Qwen3-14B",                 "qwen3-14b"),       # THE collision case
    ("Qwen/Qwen3-27B-AWQ",             "qwen3-27b-awq"),
    ("mlx-community/Qwen3-32B-4bit",   "qwen3-32b"),
    # Gemma 3
    ("mlx-community/gemma-3-1b-it-4bit",  "gemma3-1b"),
    ("mlx-community/gemma-3-4b-it-4bit",  "gemma3-4b"),
    ("mlx-community/gemma-3-12b-it-4bit", "gemma3-12b"),
    ("mlx-community/gemma-3-27b-it-4bit", "gemma3-27b"),
    # Llama 3.x
    ("mlx-community/Llama-3.2-1B-Instruct-4bit", "llama3-1b"),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "llama3-3b"),
    # Empty / unknown
    ("", ""),
])
def test_env_model_tag(model_str, expected_tag, monkeypatch):
    monkeypatch.setenv("GEPA_MODEL", model_str)
    result = _env_model_tag()
    assert result == expected_tag, (
        f"_env_model_tag() returned {result!r} for GEPA_MODEL={model_str!r}; "
        f"expected {expected_tag!r}"
    )
