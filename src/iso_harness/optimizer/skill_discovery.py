"""Inductive skill discovery for ISO optimizer.

Phase I: run seed prompts, collect failures, cluster into skills,
instantiate skill-specific candidates.
"""

from __future__ import annotations

import logging
from statistics import median
from typing import Any

from iso_harness.optimizer.candidate import Candidate, ModuleTrace, SkillCluster
from iso_harness.optimizer.formatting import format_failures
from iso_harness.optimizer.helpers import (
    apply_candidate_prompts,
    ensure_example_ids,
    extract_per_module_outputs,
    log_warning,
)
from iso_harness.optimizer.parsing import parse_clusters_from_response, parse_prompts_from_response
from iso_harness.optimizer.prompts import load_prompt

logger = logging.getLogger("iso")


def discover_skills(
    student,
    trainset: list,
    n_discovery_examples: int,
    runtime,  # ISORuntime — not imported to avoid circular imports
) -> list[SkillCluster]:
    """Run generic seed on examples, collect failures, cluster into skill categories.

    Args:
        student: DSPy module with named predictors.
        trainset: Full training dataset (DSPy Example list).
        n_discovery_examples: How many examples to run during discovery.
        runtime: ISORuntime duck-typed — provides .rng, .metric, .reflection_lm.

    Returns:
        List of SkillCluster objects, one per discovered failure category.
    """
    # Step 1: Extract module names
    modules = list(student.named_predictors())
    module_names = [name for name, _ in modules]

    # Step 2: Run student with seed prompts on N training examples
    ensure_example_ids(trainset)
    if len(trainset) < n_discovery_examples:
        sample = list(trainset)
    else:
        sample = runtime.rng.sample(trainset, n_discovery_examples)

    traces = []
    for example in sample:
        prediction = student(**example.inputs())
        result = runtime.metric(example, prediction, trace=None, pred_name=None)
        trace = ModuleTrace(
            example_id=getattr(example, 'id', str(id(example))),
            prediction=prediction,
            score=result["score"],
            feedback=result["feedback"],
            metadata=result.get("metadata", {}),
            module_outputs=extract_per_module_outputs(prediction),
        )
        traces.append(trace)

    # Step 3: Filter to failures
    all_scores = [t.score for t in traces]
    failure_threshold = max(median(all_scores) if all_scores else 0.5, 0.5)
    failures = [t for t in traces if t.score < failure_threshold]

    if len(failures) < 3:
        # Insufficient failures for meaningful clustering
        return [SkillCluster(
            label="default",
            description="Generic task-solving skill",
            example_traces=traces[:5],
            target_module=None,
        )]

    # Step 4: LLM-based clustering
    clusters = cluster_failures_via_llm(
        failures=failures,
        target_n_min=3,
        target_n_max=8,
        modules=module_names,
        runtime=runtime,
    )
    return clusters


def cluster_failures_via_llm(
    failures: list[ModuleTrace],
    target_n_min: int,
    target_n_max: int,
    modules: list[str],
    runtime,
) -> list[SkillCluster]:
    """Single reflection call that clusters failures into N categories.

    Retries up to 3 times on malformed output; falls back to default cluster
    on repeated failure.

    Args:
        failures: ModuleTrace list of failed examples.
        target_n_min: Minimum acceptable number of clusters.
        target_n_max: Maximum acceptable number of clusters.
        modules: List of DSPy module names in the student.
        runtime: ISORuntime duck-typed — provides .reflection_lm.

    Returns:
        List of SkillCluster objects.
    """
    prompt_template = load_prompt("skill_discovery")
    prompt = prompt_template.format(
        n_failures=len(failures),
        failures_text=format_failures(failures),
        modules_text="\n".join(modules),
        n_min=target_n_min,
        n_max=target_n_max,
    )

    last_error = None
    for attempt in range(3):
        try:
            if attempt > 0:
                # Add retry note before the retry call
                prompt = prompt + "\n\nYour previous response did not match the required JSON schema. Please try again."
            response = runtime.reflection_lm(prompt)
            cluster_dicts = parse_clusters_from_response(response)

            if not (target_n_min <= len(cluster_dicts) <= target_n_max):
                last_error = f"Returned {len(cluster_dicts)} clusters, expected {target_n_min}-{target_n_max}"
                continue

            # Validate and build SkillCluster objects
            failures_by_id = {f.example_id: f for f in failures}
            clusters = []
            valid = True
            for cd in cluster_dicts:
                label = cd.get("label", "")
                description = cd.get("description", "")
                target_module = cd.get("target_module")

                if not label or not description:
                    valid = False
                    last_error = "Cluster missing label or description"
                    break
                if target_module is not None and target_module not in modules:
                    valid = False
                    last_error = f"Unknown target_module: {target_module}"
                    break

                # Attach example traces by looking up failure IDs
                example_ids = cd.get("example_failure_ids", [])
                example_traces = [
                    failures_by_id[fid]
                    for fid in example_ids
                    if fid in failures_by_id
                ][:5]

                clusters.append(SkillCluster(
                    label=label,
                    description=description,
                    target_module=target_module,
                    example_traces=example_traces,
                ))

            if valid:
                return clusters

        except (ValueError, RuntimeError, KeyError, TypeError) as e:
            last_error = str(e)

    # All 3 attempts failed — return default fallback
    log_warning(f"Skill discovery clustering failed after 3 attempts: {last_error}")
    return [SkillCluster(
        label="default",
        description="Generic task-solving skill (clustering failed)",
        example_traces=failures[:5],
        target_module=None,
    )]


def instantiate_candidate_from_skill(
    skill: SkillCluster,
    student,
    runtime,
) -> Candidate:
    """Write a prompt targeting this skill's failure mode.

    Args:
        skill: SkillCluster describing the failure category to target.
        student: DSPy module with named predictors.
        runtime: ISORuntime duck-typed — provides .reflection_lm.

    Returns:
        A new Candidate with prompts tailored to the skill.
    """
    prompt_template = load_prompt("skill_instantiation")
    prompt = prompt_template.format(
        skill_label=skill.label,
        skill_description=skill.description,
        example_failures=format_failures(skill.example_traces[:3]),
        target_module=skill.target_module or "all modules",
    )

    response = runtime.reflection_lm(prompt)
    new_prompts = parse_prompts_from_response(response)

    # Build prompts dict, preserving existing instructions where the reflector
    # didn't provide a new one for a given module.
    candidate_prompts = {}
    for name, predictor in student.named_predictors():
        candidate_prompts[name] = new_prompts.get(
            name, predictor.signature.instructions
        )

    return Candidate(
        parent_ids=[],
        birth_round=0,
        birth_mechanism="skill_discovery",
        skill_category=skill.label,
        prompts_by_module=candidate_prompts,
    )


def mutate_candidate(
    candidate: Candidate,
    scope: str,  # "independent" for initial pool growth
    runtime,
) -> Candidate:
    """Mutate a candidate into a variant via reflection.

    Used for initial pool expansion (post-skill-discovery). Unlike
    reflect_per_candidate, this has no failure traces available yet — the
    reflector is asked to produce a variation based on the candidate's skill
    category.

    Args:
        candidate: Candidate to mutate.
        scope: Mutation scope string (e.g. "independent" for initial pool growth).
        runtime: ISORuntime duck-typed — provides .reflection_lm.

    Returns:
        A new Candidate derived from the input with modified prompts.
    """
    prompt_template = load_prompt("initial_mutation")

    # Format prompts dict for the template
    prompts_str = "\n".join(
        f"  {mod}: {prompt}" for mod, prompt in candidate.prompts_by_module.items()
    )

    prompt = prompt_template.format(
        current_prompts=prompts_str,
        skill_category=candidate.skill_category or "generic",
    )

    try:
        response = runtime.reflection_lm(prompt)
        new_prompts = parse_prompts_from_response(response)
    except (ValueError, RuntimeError):
        # Fallback: copy the parent unchanged (still counts as a pool member)
        new_prompts = dict(candidate.prompts_by_module)

    return Candidate(
        parent_ids=[candidate.id],
        birth_round=0,
        birth_mechanism="initial_mutation",
        skill_category=candidate.skill_category,
        prompts_by_module={**candidate.prompts_by_module, **new_prompts},
    )
