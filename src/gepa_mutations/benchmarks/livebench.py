"""LiveBench-Math benchmark loader.

Source: HF `livebench/math` (368 math questions).
Question text is in `turns[0]`, answer in `ground_truth`.
"""

from __future__ import annotations

import random

import dspy
from datasets import load_dataset

from gepa_mutations.benchmarks.loader import BenchmarkData


def load_livebench(seed: int = 0) -> BenchmarkData:
    """Load LiveBench-Math benchmark (White et al. 2025).

    - Source: HF `livebench/math`, split `test` (368 examples)
    - Shuffle with python seed 0
    - Split proportional: ~73/147/148
    """
    dataset = load_dataset("livebench/math", split="test")

    examples = []
    for item in dataset:
        # Question is in turns[0], answer in ground_truth
        question = item["turns"][0] if item.get("turns") else ""
        answer = item.get("ground_truth", "")

        examples.append(
            dspy.Example(
                input=str(question),
                answer=str(answer),
            ).with_inputs("input")
        )

    random.Random(seed).shuffle(examples)

    # 368 examples — proportional split: 20%/40%/40%
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
            "name": "livebench",
            "source": "livebench/math",
            "total_examples": n,
            "shuffle_seed": seed,
        },
    )
