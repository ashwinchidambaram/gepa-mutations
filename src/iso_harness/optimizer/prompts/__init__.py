"""Prompt template loader for ISO optimizer."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template by name (with or without .txt extension)."""
    if not name.endswith(".txt"):
        name = f"{name}.txt"
    path = _PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text()
