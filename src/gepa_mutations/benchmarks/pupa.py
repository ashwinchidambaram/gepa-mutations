"""PUPA benchmark loader.

Reference: GEPA tests use Columbia-NLP/PUPA with config pupa_tnb.
See gepa/tests/test_pareto_frontier_types/test_pareto_frontier_types.py
"""

from __future__ import annotations

import random

import dspy
from datasets import load_dataset

from gepa_mutations.benchmarks.loader import BenchmarkData


def load_pupa(seed: int = 0) -> BenchmarkData:
    """Load PUPA benchmark (Columbia-NLP, privacy-preserving delegation).

    - Source: HF `Columbia-NLP/PUPA` config `pupa_tnb` (237 examples)
    - Shuffle seed 0
    - Split proportional (small dataset): ~47/95/95
    """
    dataset = load_dataset("Columbia-NLP/PUPA", "pupa_tnb", split="train")

    examples = []
    for item in dataset:
        examples.append(
            dspy.Example(
                input=item["user_query"],
                answer=item["redacted_query"],
            ).with_inputs("input")
        )

    random.Random(0).shuffle(examples)

    # 237 examples total — proportional split: 20%/40%/40%
    n = len(examples)
    t = max(1, n // 5)
    v = max(1, (n - t) // 2)
    trainset = examples[:t]
    valset = examples[t : t + v]
    testset = examples[t + v :]

    return BenchmarkData(
        train=trainset,
        val=valset,
        test=testset,
        metadata={
            "name": "pupa",
            "source": "Columbia-NLP/PUPA",
            "config": "pupa_tnb",
            "total_examples": n,
            "shuffle_seed": 0,
        },
    )
