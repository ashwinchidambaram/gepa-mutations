"""Meta-optimizer prompt template loader."""
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent

def load_meta_prompt(name: str) -> str:
    """Load a meta-optimizer prompt template by name."""
    if not name.endswith(".txt"):
        name = f"{name}.txt"
    path = _PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Meta prompt template not found: {path}")
    return path.read_text()
