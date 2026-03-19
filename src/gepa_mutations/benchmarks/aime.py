"""AIME-2025 benchmark loader matching gepa/examples/aime_math/utils.py exactly."""

from __future__ import annotations

import random

import dspy
from datasets import load_dataset

from gepa_mutations.benchmarks.loader import BenchmarkData


def load_aime(seed: int = 0) -> BenchmarkData:
    """Load AIME-2025 benchmark matching paper protocol.

    - Train source: AI-MO/aimo-validation-aime from HF, shuffled with Random(0)
    - Split: 50/50 into train/val
    - Test source: MathArena/aime_2025 from HF
    - Test set duplicated 5x (5 runs per question, per paper protocol)
    """
    # Train/val from AIMO validation set (matches utils.py:46-54)
    train_split = []
    train_load_dataset = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
    for item in train_load_dataset:
        train_split.append(
            dspy.Example(
                input=item["problem"],
                solution=item["solution"],
                answer=item["answer"],
            ).with_inputs("input")
        )

    # Shuffle with seed 0 (matching paper, NOT experiment seed)
    random.Random(0).shuffle(train_split)

    # 50/50 split (matches utils.py:63-65)
    train_size = len(train_split)
    trainset = train_split[: train_size // 2]
    valset = train_split[train_size // 2 :]

    # Test from MathArena/aime_2025 (matches utils.py:56-61)
    test_split = []
    test_load_dataset = load_dataset("MathArena/aime_2025", "default", split="train")
    for item in test_load_dataset:
        test_split.append(
            dspy.Example(
                input=item["problem"],
                answer=item["answer"],
            ).with_inputs("input")
        )

    # IMPORTANT: Test set duplicated 5x per paper protocol (5 runs per question)
    testset = test_split * 5

    return BenchmarkData(
        train=trainset,
        val=valset,
        test=testset,
        metadata={
            "name": "aime",
            "train_source": "AI-MO/aimo-validation-aime",
            "test_source": "MathArena/aime_2025",
            "test_duplication": 5,
            "shuffle_seed": 0,
        },
    )
