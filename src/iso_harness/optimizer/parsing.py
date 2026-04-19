"""JSON parsing from LLM responses for ISO optimizer."""

from __future__ import annotations

import json
import re
import logging
from typing import Any

logger = logging.getLogger("iso")


def parse_json_from_response(response: str) -> dict:
    """
    Extract JSON from LLM response. Handles:
      - Responses wrapped in ```json ... ``` fences
      - Responses with prose before/after the JSON object
      - Trailing commas
      - Single quotes (converted to double)
    Raises ValueError if parsing fails after all fallbacks.
    """
    text = str(response)

    # Strategy 1: Extract from ```json ... ``` fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if fence_match:
        text_candidate = fence_match.group(1).strip()
        try:
            return json.loads(text_candidate)
        except json.JSONDecodeError:
            text = text_candidate  # Use the fenced content for further cleanup

    # Strategy 2: Find JSON object/array boundaries
    # Look for the first { or [ and match to its closing counterpart
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = text.find(start_char)
        if start_idx >= 0:
            # Find the matching end bracket
            depth = 0
            in_string = False
            escape_next = False
            for i in range(start_idx, len(text)):
                c = text[i]
                if escape_next:
                    escape_next = False
                    continue
                if c == '\\':
                    escape_next = True
                    continue
                if c == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == start_char:
                    depth += 1
                elif c == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start_idx:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            # Try cleanup
                            cleaned = _cleanup_json(candidate)
                            try:
                                return json.loads(cleaned)
                            except json.JSONDecodeError:
                                break  # Try next pattern

    # Strategy 3: Try the whole text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        cleaned = _cleanup_json(text.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response (first 200 chars): {text[:200]}")


def _cleanup_json(text: str) -> str:
    """Light cleanup of common JSON issues from LLM output."""
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Replace single quotes with double quotes (naive — doesn't handle apostrophes in values well)
    # Only do this if there are no double quotes at all
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    return text


def parse_clusters_from_response(response: str) -> list:
    """Parse skill-discovery response into SkillCluster-like dicts.

    Returns list of dicts with keys: label, description, target_module, example_failure_ids
    Caller should convert to SkillCluster objects.
    """
    data = parse_json_from_response(response)
    clusters = data.get("clusters", [])
    if not isinstance(clusters, list):
        raise ValueError(f"Expected 'clusters' to be a list, got {type(clusters)}")

    result = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            raise ValueError(f"Expected cluster to be a dict, got {type(cluster)}")
        label = cluster.get("label", "")
        description = cluster.get("description", "")
        if not label or not description:
            raise ValueError(f"Cluster missing label or description: {cluster}")
        result.append({
            "label": label,
            "description": description,
            "target_module": cluster.get("target_module"),
            "example_failure_ids": cluster.get("example_failure_ids", []),
        })

    return result


def parse_prompts_from_response(response: str) -> dict[str, str]:
    """Parse reflection response into {module_name: prompt_text}."""
    data = parse_json_from_response(response)
    prompts = data.get("prompts", {})
    if not isinstance(prompts, dict):
        raise ValueError(f"Expected 'prompts' to be a dict, got {type(prompts)}")
    # Validate all values are strings
    for key, value in prompts.items():
        if not isinstance(value, str):
            raise ValueError(f"Prompt for module '{key}' is not a string: {type(value)}")
    return prompts


def parse_pairs_from_response(response: str) -> list[dict]:
    """Parse reflector-guided cross-mutation response into pair proposals.

    Returns list of dicts with keys: parent_a_id, parent_b_id, rationale
    """
    data = parse_json_from_response(response)
    pairs = data.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError(f"Expected 'pairs' to be a list, got {type(pairs)}")

    result = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        a_id = pair.get("parent_a_id", "")
        b_id = pair.get("parent_b_id", "")
        rationale = pair.get("rationale", "")
        if a_id and b_id:
            result.append({
                "parent_a_id": a_id,
                "parent_b_id": b_id,
                "rationale": rationale,
            })

    return result


def parse_insights_from_response(response: str) -> dict:
    """Parse pair-contrastive insights into {what_worked, what_failed, recommended_changes}."""
    data = parse_json_from_response(response)
    return {
        "what_worked": data.get("what_worked", ""),
        "what_failed": data.get("what_failed", ""),
        "recommended_changes": data.get("recommended_changes", ""),
    }


def parse_config_from_response(response: str) -> dict:
    """Parse meta-optimizer config proposal."""
    data = parse_json_from_response(response)
    # Validate expected MetaAction fields
    valid_keys = {
        "pool_size_seed", "mutations_per_seed", "minibatch_count",
        "minibatch_size", "prune_aggressiveness", "max_rounds",
    }
    return {k: v for k, v in data.items() if k in valid_keys}


def extract_reasoning(response: str) -> str:
    """Extract the LLM's reasoning preamble before the first JSON block."""
    text = str(response)

    # Find the start of JSON (first { or ```)
    json_start = len(text)
    for pattern in [r'```', r'\{']:
        match = re.search(pattern, text)
        if match and match.start() < json_start:
            json_start = match.start()

    reasoning = text[:json_start].strip()
    return reasoning


def parse_playbook_from_response(response: str) -> str:
    """Extract playbook text from meta-optimizer response.

    The playbook is free-form text, not JSON. Return the full response
    after stripping any JSON blocks.
    """
    text = str(response)
    # Remove any JSON blocks
    text = re.sub(r'```(?:json)?\s*\n?.*?\n?\s*```', '', text, flags=re.DOTALL)
    return text.strip()
