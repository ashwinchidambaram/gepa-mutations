"""IFBench benchmark loader."""

from __future__ import annotations

import random

import dspy
from datasets import load_dataset

from gepa_mutations.benchmarks.loader import BenchmarkData


def load_ifbench(seed: int = 0) -> BenchmarkData:
    """Load IFBench (instruction following with multiple constraints).

    - Source: HF `allenai/IF_multi_constraints_upto5`
    - Shuffle seed 0
    - Split: 150/300/300
    """
    dataset = load_dataset("allenai/IF_multi_constraints_upto5", split="train")

    examples = []
    for item in dataset:
        # Build input with the instruction
        instruction = item.get("instruction", item.get("prompt", ""))
        constraints = item.get("constraints", [])

        # Format constraints into the input if available
        if constraints:
            constraint_text = "\n".join(f"- {c}" for c in constraints)
            input_text = f"{instruction}\n\nConstraints:\n{constraint_text}"
        else:
            input_text = instruction

        example = dspy.Example(
            input=input_text,
            answer=item.get("response", ""),
            constraints=constraints,
        ).with_inputs("input")
        examples.append(example)

    random.Random(0).shuffle(examples)

    trainset = examples[:150]
    valset = examples[150:450]
    testset = examples[450:750]

    return BenchmarkData(
        train=trainset,
        val=valset,
        test=testset,
        metadata={
            "name": "ifbench",
            "source": "allenai/IF_multi_constraints_upto5",
            "split_sizes": "150/300/300",
            "shuffle_seed": 0,
        },
    )
