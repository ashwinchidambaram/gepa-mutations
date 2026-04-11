"""HotpotQA benchmark loader."""

from __future__ import annotations

import random

import dspy
from datasets import load_dataset

from gepa_mutations.benchmarks.loader import BenchmarkData


def load_hotpotqa(seed: int = 0) -> BenchmarkData:
    """Load HotpotQA (distractor setting).

    - Source: HF `hotpot_qa` (distractor)
    - Shuffle seed 0
    - Split: 150/300/300 (train/val/test)
    """
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    examples = []
    for item in dataset:
        # Combine supporting facts into context
        context_parts = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            context_parts.append(f"{title}: {''.join(sentences)}")
        context = "\n".join(context_parts)

        examples.append(
            dspy.Example(
                input=f"Context:\n{context}\n\nQuestion: {item['question']}",
                answer=item["answer"],
            ).with_inputs("input")
        )

    random.Random(seed).shuffle(examples)

    trainset = examples[:150]
    valset = examples[150:450]
    testset = examples[450:750]

    return BenchmarkData(
        train=trainset,
        val=valset,
        test=testset,
        metadata={
            "name": "hotpotqa",
            "source": "hotpot_qa/distractor",
            "split_sizes": "150/300/300",
            "shuffle_seed": seed,
        },
    )
