"""PUPA benchmark loader."""

from __future__ import annotations

import random

import dspy
from datasets import load_dataset

from gepa_mutations.benchmarks.loader import BenchmarkData


def load_pupa(seed: int = 0) -> BenchmarkData:
    """Load PUPA benchmark (Li et al. 2025).

    - Source: HF dataset (searching for the right identifier)
    - Shuffle seed 0
    - Split: 150/300/300
    """
    # PUPA dataset - try known identifiers
    try:
        dataset = load_dataset("liyucheng/pupa", split="test")
    except Exception:
        try:
            dataset = load_dataset("PUPA-benchmark/PUPA", split="test")
        except Exception:
            # Fallback: try loading from a generic name
            dataset = load_dataset("liyucheng/PUPA", split="test")

    examples = []
    for item in dataset:
        question = item.get("question", item.get("input", item.get("problem", "")))
        answer = item.get("answer", item.get("output", item.get("label", "")))

        examples.append(
            dspy.Example(
                input=str(question),
                answer=str(answer),
            ).with_inputs("input")
        )

    random.Random(0).shuffle(examples)

    # If dataset is small, adjust split sizes
    n = len(examples)
    if n >= 750:
        trainset = examples[:150]
        valset = examples[150:450]
        testset = examples[450:750]
    else:
        # Proportional split: 20%/40%/40%
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
            "source": "pupa",
            "total_examples": n,
            "shuffle_seed": 0,
        },
    )
