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

    Dataset schema (current): each row has:
      - messages: list of {role, content} dicts; messages[0] is the user turn
        with the instruction and embedded constraints.
      - constraint: tab-separated string of individual constraint sentences.
      - ground_truth: JSON string with IFEval instruction IDs (not used here).

    The constraint text is already embedded in the user message, so we use
    messages[0]["content"] directly as the input.  We also extract the
    individual constraints from the tab-separated `constraint` field so
    IFBenchAdapter can check them programmatically.
    """
    dataset = load_dataset("allenai/IF_multi_constraints_upto5", split="train")

    examples = []
    for item in dataset:
        # Extract the user instruction from the messages list.
        # messages[0] is always the user turn; it contains the full instruction
        # with constraints already embedded in the text.
        messages = item.get("messages", [])
        if messages and isinstance(messages[0], dict):
            input_text = messages[0].get("content", "")
        else:
            input_text = ""

        # Extract individual constraints from the tab-separated constraint field.
        constraint_str = item.get("constraint", "")
        constraints = [c.strip() for c in constraint_str.split("\t") if c.strip()]

        example = dspy.Example(
            input=input_text,
            answer="",  # No gold response; IFBenchAdapter scores via constraints list
            constraints=constraints,
        ).with_inputs("input")
        examples.append(example)

    random.Random(seed).shuffle(examples)

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
            "shuffle_seed": seed,
        },
    )
