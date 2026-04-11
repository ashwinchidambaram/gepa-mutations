"""HoVer benchmark loader."""

from __future__ import annotations

import random

import dspy
from datasets import load_dataset

from gepa_mutations.benchmarks.loader import BenchmarkData


def load_hover(seed: int = 0) -> BenchmarkData:
    """Load HoVer (multi-hop fact verification).

    - Source: HF `hover_ner` or `bdsaglam/hover`
    - Shuffle seed 0
    - Split: 150/300/300
    """
    dataset = load_dataset("bdsaglam/hover", split="train")

    # Label mapping for bdsaglam/hover. Verify against dataset if scores look inverted:
    #   dataset.features["label"] will show the ClassLabel names in index order.
    # Standard HoVer convention: 0=NOT_SUPPORTED, 1=SUPPORTED (binary split).
    label_map = {0: "not_supported", 1: "supported"}

    examples = []
    for item in dataset:
        claim = item.get("claim", "")
        label_id = item.get("label", 0)
        if label_id not in label_map:
            raise ValueError(
                f"Unexpected HoVer label id {label_id!r}. "
                f"Known ids: {list(label_map)}. Update label_map if the dataset schema changed."
            )
        label = label_map[label_id]

        # Build context from supporting facts if available
        context = ""
        if "supporting_facts" in item and item["supporting_facts"]:
            facts = item["supporting_facts"]
            if isinstance(facts, list):
                context = "\n".join(str(f) for f in facts)
            elif isinstance(facts, dict):
                titles = facts.get("title", [])
                sent_ids = facts.get("sent_id", [])
                context = ", ".join(f"{t} (sent {s})" for t, s in zip(titles, sent_ids))

        input_text = f"Claim: {claim}"
        if context:
            input_text = f"Evidence: {context}\n\n{input_text}"

        examples.append(
            dspy.Example(
                input=input_text,
                answer=label,
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
            "name": "hover",
            "source": "bdsaglam/hover",
            "split_sizes": "150/300/300",
            "shuffle_seed": seed,
        },
    )
