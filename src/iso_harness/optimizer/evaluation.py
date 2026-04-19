"""Multi-minibatch evaluation for ISO optimizer pool."""

from __future__ import annotations

import logging
from statistics import mean, stdev

from iso_harness.optimizer.candidate import Candidate, ModuleTrace
from iso_harness.optimizer.helpers import (
    apply_candidate_prompts,
    extract_per_module_outputs,
)
from iso_harness.experiment.context import set_context

logger = logging.getLogger("iso")


def evaluate_pool_multi_minibatch(
    pool: list[Candidate],
    minibatches: list[list],
    student,
    runtime,  # ISORuntime
) -> dict[str, dict]:
    """
    Evaluate every candidate on every minibatch. Return mean + per-batch scores.
    Stores full ModuleTrace per (candidate, example) in runtime.trace_store.
    """
    results = {}
    for candidate in pool:
        patched_student = apply_candidate_prompts(student, candidate)

        # Set context for logging
        set_context(candidate_id=candidate.id, round_num=runtime.round_num)

        batch_scores = []
        per_example_scores = {}
        per_example_feedback = {}
        per_example_metadata = {}

        for batch in minibatches:
            for example in batch:
                example_id = getattr(example, 'id', str(id(example)))

                prediction = patched_student(**example.inputs())
                result = runtime.metric(example, prediction, trace=None, pred_name=None)

                score = result["score"]
                feedback = result["feedback"]
                metadata = result.get("metadata", {})

                per_example_scores[example_id] = score
                per_example_feedback[example_id] = feedback
                per_example_metadata[example_id] = metadata

                # Persist trace for reflection step
                runtime.trace_store.put(
                    candidate.id, example_id,
                    ModuleTrace(
                        example_id=example_id,
                        prediction=prediction,
                        score=score,
                        feedback=feedback,
                        metadata=metadata,
                        module_outputs=extract_per_module_outputs(prediction),
                    ),
                )

            batch_mean = mean([per_example_scores[getattr(e, 'id', str(id(e)))] for e in batch])
            batch_scores.append(batch_mean)

        results[candidate.id] = {
            "mean": mean(batch_scores) if batch_scores else 0.0,
            "per_batch": batch_scores,
            "variance": stdev(batch_scores) if len(batch_scores) > 1 else 0.0,
            "per_example": per_example_scores,
            "per_example_feedback": per_example_feedback,
            "per_example_metadata": per_example_metadata,
        }

        candidate.score_history.append((runtime.round_num, results[candidate.id]["mean"]))
        candidate.per_instance_scores.update(per_example_scores)

    return results


def evaluate_on_valset(
    pool: list[Candidate],
    valset: list,
    student,
    runtime,
) -> dict[str, float]:
    """
    Evaluate all candidates on the full validation set.
    Returns {candidate_id: mean_score}.
    Used for final winner selection (Phase III of the core loop).
    """
    results = {}
    for candidate in pool:
        if candidate.death_round is not None:
            continue  # Skip dead candidates

        patched_student = apply_candidate_prompts(student, candidate)
        set_context(candidate_id=candidate.id, round_num=runtime.round_num)

        scores = []
        for example in valset:
            example_id = getattr(example, 'id', str(id(example)))
            prediction = patched_student(**example.inputs())
            result = runtime.metric(example, prediction, trace=None, pred_name=None)
            scores.append(result["score"])

        results[candidate.id] = mean(scores) if scores else 0.0

    return results
