"""LiveBench-Math benchmark loader."""

from __future__ import annotations

import random

import dspy
from datasets import load_dataset

from gepa_mutations.benchmarks.loader import BenchmarkData


def load_livebench(seed: int = 0) -> BenchmarkData:
    """Load LiveBench-Math benchmark (White et al. 2025).

    - Source: HF livebench dataset, math category
    - Shuffle with python seed 0
    - Split: 150/300/300
    """
    # Try loading LiveBench math subset
    try:
        dataset = load_dataset("livebench/livebench", split="test")
        # Filter for math category
        examples = []
        for item in dataset:
            category = item.get("category", item.get("task", ""))
            if "math" in str(category).lower():
                question = item.get("question", item.get("input", ""))
                answer = item.get("answer", item.get("ground_truth", ""))
                examples.append(
                    dspy.Example(
                        input=str(question),
                        answer=str(answer),
                    ).with_inputs("input")
                )
    except Exception:
        # Fallback: try alternative dataset identifier
        try:
            dataset = load_dataset("LiveBench/LiveBench", "math", split="test")
            examples = []
            for item in dataset:
                question = item.get("question", item.get("input", ""))
                answer = item.get("answer", item.get("ground_truth", ""))
                examples.append(
                    dspy.Example(
                        input=str(question),
                        answer=str(answer),
                    ).with_inputs("input")
                )
        except Exception:
            # Last resort: try loading all and filtering
            dataset = load_dataset("LiveBench/LiveBench", split="test")
            examples = []
            for item in dataset:
                question = item.get("question", item.get("input", item.get("problem", "")))
                answer = item.get("answer", item.get("ground_truth", item.get("solution", "")))
                examples.append(
                    dspy.Example(
                        input=str(question),
                        answer=str(answer),
                    ).with_inputs("input")
                )

    random.Random(0).shuffle(examples)

    # Adjust splits based on available data
    n = len(examples)
    if n >= 750:
        trainset = examples[:150]
        valset = examples[150:450]
        testset = examples[450:750]
    else:
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
            "source": "LiveBench",
            "total_examples": n,
            "shuffle_seed": 0,
        },
    )
