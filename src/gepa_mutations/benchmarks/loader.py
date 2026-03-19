"""Benchmark dataset loading matching GEPA paper splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import dspy


@dataclass
class BenchmarkData:
    """Container for benchmark train/val/test splits."""

    train: list[dspy.Example]
    val: list[dspy.Example]
    test: list[dspy.Example]
    metadata: dict[str, Any] = field(default_factory=dict)


BENCHMARKS = ["aime", "hotpotqa", "ifbench", "hover", "pupa", "livebench"]


def load_benchmark(name: str, seed: int = 0) -> BenchmarkData:
    """Load a benchmark dataset by name.

    Args:
        name: One of 'aime', 'hotpotqa', 'ifbench', 'hover', 'pupa', 'livebench'.
        seed: Random seed for shuffling (default 0, matching paper).

    Returns:
        BenchmarkData with train/val/test splits as dspy.Example lists.
    """
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark '{name}'. Choose from: {BENCHMARKS}")

    if name == "aime":
        from gepa_mutations.benchmarks.aime import load_aime
        return load_aime(seed=seed)
    elif name == "hotpotqa":
        from gepa_mutations.benchmarks.hotpotqa import load_hotpotqa
        return load_hotpotqa(seed=seed)
    elif name == "ifbench":
        from gepa_mutations.benchmarks.ifbench import load_ifbench
        return load_ifbench(seed=seed)
    elif name == "hover":
        from gepa_mutations.benchmarks.hover import load_hover
        return load_hover(seed=seed)
    elif name == "pupa":
        from gepa_mutations.benchmarks.pupa import load_pupa
        return load_pupa(seed=seed)
    elif name == "livebench":
        from gepa_mutations.benchmarks.livebench import load_livebench
        return load_livebench(seed=seed)
    else:
        raise ValueError(f"Unknown benchmark '{name}'")
