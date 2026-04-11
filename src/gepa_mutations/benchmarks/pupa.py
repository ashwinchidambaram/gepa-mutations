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
        pii_raw = item.get("pii_units", "")
        if isinstance(pii_raw, list):
            # Preserve list as ||-delimited string so the evaluator can split it back out.
            pii_str = "||".join(str(p) for p in pii_raw)
        else:
            pii_str = str(pii_raw)
        examples.append(
            dspy.Example(
                input=item["user_query"],
                answer=item["redacted_query"],
                pii_units=pii_str,
            ).with_inputs("input")
        )

    random.Random(seed).shuffle(examples)

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
            "shuffle_seed": seed,
        },
    )
