"""Contrastive snippet injection into the reflective dataset.

Uses the side_info strategy: appends contrastive entries to the reflective
dataset sequence. These entries contain a ``contrastive_feedback`` key whose
value is rendered into the ``<side_info>`` section of the reflection prompt
via ``InstructionProposalSignature.prompt_renderer()``.

Does NOT modify ``InstructionProposalSignature`` or its prompt template.
"""

from __future__ import annotations

from typing import Any

from contrastive_reflection.config import ContrastiveReflectionConfig


def inject_contrastive_snippets(
    reflective_dataset: dict[str, list[dict[str, Any]]],
    contrastive_pairs: list[dict[str, Any]],
    components_to_update: list[str],
    config: ContrastiveReflectionConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Inject contrastive information into the reflective dataset.

    Appends contrastive examples to each component's reflective dataset
    as additional entries with a special ``contrastive_feedback`` field.

    This uses the side_info strategy: the contrastive info is added as
    extra entries in the reflective dataset, which get rendered into
    the ``<side_info>`` section of the reflection prompt via the existing
    ``InstructionProposalSignature.prompt_renderer()``.

    Args:
        reflective_dataset: Mapping from component name to list of feedback dicts.
        contrastive_pairs: Output from ``find_contrastive_candidates()``.
        components_to_update: Component names that are being mutated this iteration.
        config: Contrastive reflection configuration.

    Returns:
        Augmented reflective dataset with contrastive entries appended.
    """
    if not contrastive_pairs:
        return reflective_dataset

    augmented: dict[str, list[dict[str, Any]]] = {}
    for component_name, dataset_entries in reflective_dataset.items():
        if component_name not in components_to_update:
            augmented[component_name] = list(dataset_entries)
            continue

        augmented_entries = list(dataset_entries)

        for pair in contrastive_pairs:
            contrastive_candidate = pair["candidate"]
            # Get the contrastive candidate's text for this component
            contrastive_text = contrastive_candidate.get(component_name, "")
            if config.max_snippet_length and not config.include_full_text:
                contrastive_text = contrastive_text[: config.max_snippet_length]

            contrastive_entry = {
                "contrastive_feedback": (
                    f"A different candidate achieved a score of {pair['contrastive_score']:.2f} "
                    f"on this example (vs your {pair['current_score']:.2f}, gap={pair['score_gap']:.2f}). "
                    f"That candidate used the following instruction:\n{contrastive_text}"
                ),
            }
            augmented_entries.append(contrastive_entry)

        augmented[component_name] = augmented_entries

    return augmented
