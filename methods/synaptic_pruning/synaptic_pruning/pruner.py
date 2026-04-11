"""Pruning utilities for synaptic pruning prompt optimization.

Provides section parsing, ablation, and the full ablation loop used by the
SPDO (Synaptic Pruning Driven Optimization) method.
"""

from __future__ import annotations

import re
from typing import Any

from gepa_mutations.metrics.collector import MetricsCollector
from gepa_mutations.metrics.standalone_eval import evaluate_prompt

# Minimum character length for a section to be treated as independent.
# Shorter fragments are merged with neighbours rather than ablated separately.
_MIN_SECTION_CHARS = 80


def parse_sections(prompt: str) -> list[str]:
    """Split a prompt into logical sections for ablation.

    Strategy (in priority order):
    1. Markdown headers (## / ###) — clearest structural boundaries.
    2. Bolded single-line headers on their own paragraph (e.g. **Reasoning**).
    3. Double-newline paragraph splits — only when sections are long enough.

    Deliberately avoids splitting on numbered list items (1. 2. 3.) within a
    paragraph — those are inline enumeration, not section boundaries.  The old
    regex caused every numbered bullet to become its own "section", destroying
    the context header and producing 20+ micro-fragments.

    Args:
        prompt: The full prompt text.

    Returns:
        List of non-empty section strings, each >= _MIN_SECTION_CHARS chars
        (short trailing fragments are appended to the previous section).
    """
    # Strategy 1: markdown header splits (## or ###)
    header_splits = re.split(r"(?m)^(?=#{1,3}\s)", prompt)
    header_splits = [s.strip() for s in header_splits if s.strip()]
    if len(header_splits) >= 3 and all(len(s) >= _MIN_SECTION_CHARS for s in header_splits):
        return _merge_short(header_splits)

    # Strategy 2: bolded paragraph headers — a line that is ONLY bold text,
    # followed by a newline (e.g. "**Reasoning Strategies**\n1. ...")
    bold_header_pattern = re.compile(r"(?m)^(?=\*\*[A-Z][^*\n]{3,60}\*\*\s*$)")
    bold_splits = bold_header_pattern.split(prompt)
    bold_splits = [s.strip() for s in bold_splits if s.strip()]
    if len(bold_splits) >= 3 and all(len(s) >= _MIN_SECTION_CHARS for s in bold_splits):
        return _merge_short(bold_splits)

    # Strategy 3: double-newline paragraph splits
    para_splits = re.split(r"\n\s*\n+", prompt)
    para_splits = [s.strip() for s in para_splits if s.strip()]
    if para_splits:
        return _merge_short(para_splits)

    # Last resort: treat entire prompt as one section
    return [prompt.strip()] if prompt.strip() else []


def _merge_short(sections: list[str]) -> list[str]:
    """Merge any section shorter than _MIN_SECTION_CHARS into the previous one.

    This prevents micro-sections (single sentences, orphaned headers) from
    being ablated independently.
    """
    if not sections:
        return sections
    merged: list[str] = [sections[0]]
    for s in sections[1:]:
        if len(s) < _MIN_SECTION_CHARS:
            merged[-1] = merged[-1] + "\n\n" + s
        else:
            merged.append(s)
    return merged


def ablate_section(sections: list[str], idx: int) -> str:
    """Reconstruct a prompt with one section removed.

    Args:
        sections: List of prompt sections.
        idx: Index of the section to remove.

    Returns:
        Prompt string with the specified section removed.
    """
    remaining = [s for i, s in enumerate(sections) if i != idx]
    return "\n\n".join(remaining)


def run_ablation(
    adapter: Any,
    dataset: list,
    sections: list[str],
    candidate_template: dict[str, str],
    collector: MetricsCollector,
    eval_indices: list[int],
    budget: int | None = None,
    baseline_floor: float = 0.4,
) -> tuple[list[int], list[int], list[dict]]:
    """Run the full ablation loop over all sections.

    For each section, removes it and evaluates the remaining prompt on the
    provided eval_indices. Records score deltas to identify which sections
    are load-bearing vs prunable.

    Skips ablation entirely when baseline score is below ``baseline_floor``
    (default 0.4): a bad prompt produces noisy, unreliable ablation signals
    and all sections tend to appear prunable regardless of their content.

    Args:
        adapter: GEPAAdapter instance.
        dataset: Dataset to evaluate on (train or val).
        sections: List of prompt sections from parse_sections().
        candidate_template: Base candidate dict; its "system_prompt" key will
            be replaced during ablation.
        collector: MetricsCollector for rollout counting.
        eval_indices: Indices into dataset to use for each ablation eval.
        budget: Optional rollout budget cap.
        baseline_floor: Minimum baseline score required to proceed with
            ablation.  Below this threshold the initial prompt is considered
            unreliable and ablation is skipped.

    Returns:
        Tuple of:
            - load_bearing_indices: section indices where removal caused score drop > 0.05
            - prunable_indices: section indices where removal caused score drop < 0.01
              AND score_delta >= 0 (harmfully prunable, not accidentally helpful).
            - ablation_scores: list of dicts with per-section results
    """
    if budget is not None and collector.rollout_count >= budget:
        return [], [], []

    baseline_candidate = {**candidate_template, "system_prompt": "\n\n".join(sections)}
    baseline_score, _ = evaluate_prompt(
        adapter, dataset, baseline_candidate, collector, indices=eval_indices
    )

    # Skip ablation when baseline is too low — signals are unreliable
    if baseline_score < baseline_floor:
        return [], [], [{"section_idx": i, "score_without": None, "score_delta": None,
                         "skipped": "baseline_below_floor"} for i in range(len(sections))]

    ablation_scores = []
    load_bearing_indices = []
    prunable_indices = []

    for idx in range(len(sections)):
        if budget is not None and collector.rollout_count >= budget:
            break

        ablated_prompt = ablate_section(sections, idx)
        if not ablated_prompt.strip():
            ablation_scores.append({
                "section_idx": idx,
                "score_without": 0.0,
                "score_delta": baseline_score,
            })
            load_bearing_indices.append(idx)
            continue

        ablated_candidate = {**candidate_template, "system_prompt": ablated_prompt}
        score_without, _ = evaluate_prompt(
            adapter, dataset, ablated_candidate, collector, indices=eval_indices
        )
        score_delta = baseline_score - score_without

        ablation_scores.append({
            "section_idx": idx,
            "score_without": score_without,
            "score_delta": score_delta,
        })

        if score_delta > 0.05:
            load_bearing_indices.append(idx)
        elif 0.0 <= score_delta < 0.01:
            # Only prune if the section is truly neutral (small positive delta).
            # Negative delta (removal improves score) means the section is
            # actively harmful — it should be pruned too, but we handle it
            # separately to avoid over-pruning when baseline is noisy.
            prunable_indices.append(idx)
        elif score_delta < 0.0:
            # Removing this section improved the score — mark it for removal
            # as a harmful section, but cap total pruning at 50% of sections.
            prunable_indices.append(idx)

    # Safety cap: never prune more than half the sections in one pass.
    # Over-pruning produces incoherent prompts.
    max_prunable = max(1, len(sections) // 2)
    if len(prunable_indices) > max_prunable:
        # Keep only the least impactful prunable sections (smallest |delta|)
        prunable_with_delta = [
            (idx, next((s["score_delta"] for s in ablation_scores if s["section_idx"] == idx), 0.0))
            for idx in prunable_indices
        ]
        prunable_with_delta.sort(key=lambda x: abs(x[1]) if x[1] is not None else 0)
        prunable_indices = [idx for idx, _ in prunable_with_delta[:max_prunable]]

    return load_bearing_indices, prunable_indices, ablation_scores
