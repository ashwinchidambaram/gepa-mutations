"""BestOfKProposer: generate K independent mutations per iteration, keep the best.

Subclasses ReflectiveMutationProposer and overrides propose() to wrap the
mutation/evaluation section in a K-loop. When K=1, behavior is identical
to vanilla GEPA.
"""

from __future__ import annotations

import traceback
from collections.abc import Mapping, Sequence
from typing import Any

from gepa.core.adapter import DataInst, GEPAAdapter, ProposalFn, RolloutOutput, Trajectory
from gepa.core.callbacks import (
    CandidateSelectedEvent,
    EvaluationEndEvent,
    EvaluationSkippedEvent,
    EvaluationStartEvent,
    GEPACallback,
    MinibatchSampledEvent,
    ProposalEndEvent,
    ProposalStartEvent,
    ReflectiveDatasetBuiltEvent,
    notify_callbacks,
)
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal
from gepa.proposer.reflective_mutation.base import (
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.strategies.batch_sampler import BatchSampler


class BestOfKProposer(ReflectiveMutationProposer):
    """Reflective mutation proposer that generates K candidates and keeps the best.

    For each GEPA iteration, the standard reflection/evaluation of the current
    candidate happens exactly once. Then, instead of proposing a single mutation,
    we propose K independent mutations, evaluate each on the same minibatch,
    and return the one with the highest sum of scores.

    When mutation_candidates=1, behavior is identical to vanilla GEPA.
    """

    def __init__(
        self,
        logger: Any,
        trainset: list[DataInst] | DataLoader[DataId, DataInst],
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        candidate_selector: CandidateSelector,
        module_selector: ReflectionComponentSelector,
        batch_sampler: BatchSampler[DataId, DataInst],
        perfect_score: float | None,
        skip_perfect_score: bool,
        experiment_tracker: Any,
        mutation_candidates: int = 1,
        reflection_lm: LanguageModel | None = None,
        reflection_prompt_template: str | dict[str, str] | None = None,
        custom_candidate_proposer: ProposalFn | None = None,
        callbacks: list[GEPACallback] | None = None,
    ):
        super().__init__(
            logger=logger,
            trainset=trainset,
            adapter=adapter,
            candidate_selector=candidate_selector,
            module_selector=module_selector,
            batch_sampler=batch_sampler,
            perfect_score=perfect_score,
            skip_perfect_score=skip_perfect_score,
            experiment_tracker=experiment_tracker,
            reflection_lm=reflection_lm,
            reflection_prompt_template=reflection_prompt_template,
            custom_candidate_proposer=custom_candidate_proposer,
            callbacks=callbacks,
        )
        if mutation_candidates < 1:
            raise ValueError(f"mutation_candidates must be >= 1, got {mutation_candidates}")
        self.mutation_candidates = mutation_candidates

        # Find the BestOfKMetricsCallback in callbacks list (if present)
        from best_of_k.callbacks import BestOfKMetricsCallback

        self._best_of_k_cb: BestOfKMetricsCallback | None = None
        for cb in (callbacks or []):
            if isinstance(cb, BestOfKMetricsCallback):
                self._best_of_k_cb = cb
                break

    def propose(self, state: GEPAState) -> CandidateProposal | None:
        """Propose a new candidate using best-of-K selection.

        Steps 1-5 (candidate selection, minibatch sampling, current evaluation,
        skip checks, predictor selection) run exactly once. Then the K-loop
        wraps steps 6-7 (propose new texts + evaluate) to generate K independent
        mutations and return the best one.
        """
        i = state.i + 1
        K = self.mutation_candidates

        # =====================================================================
        # Step 1: Select candidate via selector
        # =====================================================================
        curr_prog_id = self.candidate_selector.select_candidate_idx(state)
        curr_prog = state.program_candidates[curr_prog_id]
        state.full_program_trace[-1]["selected_program_candidate"] = curr_prog_id
        self.logger.log(
            f"Iteration {i}: Selected program {curr_prog_id} score: {state.program_full_scores_val_set[curr_prog_id]}"
        )

        # Notify candidate selected
        notify_callbacks(
            self.callbacks,
            "on_candidate_selected",
            CandidateSelectedEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                candidate=curr_prog,
                score=state.program_full_scores_val_set[curr_prog_id],
            ),
        )

        self.experiment_tracker.log_metrics(
            {"iteration": i, "selected_program_candidate": curr_prog_id, "total_metric_calls": state.total_num_evals},
            step=i,
        )

        # =====================================================================
        # Step 2: Sample minibatch
        # =====================================================================
        subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids
        minibatch = self.trainset.fetch(subsample_ids)

        # Notify minibatch sampled
        notify_callbacks(
            self.callbacks,
            "on_minibatch_sampled",
            MinibatchSampledEvent(
                iteration=i,
                minibatch_ids=subsample_ids,
                trainset_size=len(self.trainset),
            ),
        )

        # =====================================================================
        # Step 3: Evaluate current program with traces (ONCE regardless of K)
        # =====================================================================
        curr_parent_ids = [p for p in state.parent_program_for_candidate[curr_prog_id] if p is not None]
        is_seed_candidate = curr_prog_id == 0
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                batch_size=len(minibatch),
                capture_traces=True,
                parent_ids=curr_parent_ids,
                inputs=minibatch,
                is_seed_candidate=is_seed_candidate,
            ),
        )
        eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
        state.increment_evals(len(subsample_ids))
        state.full_program_trace[-1]["subsample_scores"] = eval_curr.scores
        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                scores=eval_curr.scores,
                has_trajectories=bool(eval_curr.trajectories),
                parent_ids=curr_parent_ids,
                outputs=eval_curr.outputs,
                trajectories=eval_curr.trajectories,
                objective_scores=eval_curr.objective_scores,
                is_seed_candidate=is_seed_candidate,
            ),
        )

        # Update cache with current program evaluation results
        if state.evaluation_cache is not None:
            objective_scores_list = list(eval_curr.objective_scores) if eval_curr.objective_scores else None
            state.evaluation_cache.put_batch(
                curr_prog, subsample_ids, eval_curr.outputs, eval_curr.scores, objective_scores_list
            )

        # =====================================================================
        # Step 4: Skip if no trajectories or all perfect
        # =====================================================================
        if not eval_curr.trajectories or len(eval_curr.trajectories) == 0:
            self.logger.log(f"Iteration {i}: No trajectories captured. Skipping.")
            notify_callbacks(
                self.callbacks,
                "on_evaluation_skipped",
                EvaluationSkippedEvent(
                    iteration=i,
                    candidate_idx=curr_prog_id,
                    reason="no_trajectories",
                    scores=eval_curr.scores,
                    is_seed_candidate=is_seed_candidate,
                ),
            )
            return None

        if (
            self.skip_perfect_score
            and self.perfect_score is not None
            and all(s is not None and s >= self.perfect_score for s in eval_curr.scores)
        ):
            self.logger.log(f"Iteration {i}: All subsample scores perfect. Skipping.")
            notify_callbacks(
                self.callbacks,
                "on_evaluation_skipped",
                EvaluationSkippedEvent(
                    iteration=i,
                    candidate_idx=curr_prog_id,
                    reason="all_scores_perfect",
                    scores=eval_curr.scores,
                    is_seed_candidate=is_seed_candidate,
                ),
            )
            return None

        subsample_before = sum(eval_curr.scores)

        # =====================================================================
        # Step 5: Decide which predictors to update
        # =====================================================================
        predictor_names_to_update = self.module_selector(
            state, eval_curr.trajectories, eval_curr.scores, curr_prog_id, curr_prog
        )

        # =====================================================================
        # Steps 6-7: K-loop — propose + evaluate K candidates, keep the best
        # =====================================================================
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(
                curr_prog, eval_curr, predictor_names_to_update
            )

            # Convert to concrete types for callback
            reflective_dataset_concrete: dict[str, list[dict[str, Any]]] = {
                k: [dict(item) for item in v] for k, v in reflective_dataset.items()
            }

            # Notify reflective dataset built
            notify_callbacks(
                self.callbacks,
                "on_reflective_dataset_built",
                ReflectiveDatasetBuiltEvent(
                    iteration=i,
                    candidate_idx=curr_prog_id,
                    components=predictor_names_to_update,
                    dataset=reflective_dataset_concrete,
                ),
            )

            # Evaluator for proposal evaluation (defined once, shared across K)
            def evaluator(b, c):
                r = self.adapter.evaluate(b, c, capture_traces=False)
                return r.outputs, r.scores, list(r.objective_scores) if r.objective_scores else None

            # Track best candidate across K attempts
            best_candidate: dict[str, str] | None = None
            best_scores: list[float] | None = None
            best_sum: float = float("-inf")
            best_k_index: int = 0
            all_k_scores: list[float] = []
            seen_hashes: set[int] = set()
            unique_candidates: int = 0

            for k in range(K):
                # --- Propose new texts ---
                notify_callbacks(
                    self.callbacks,
                    "on_proposal_start",
                    ProposalStartEvent(
                        iteration=i,
                        parent_candidate=curr_prog,
                        components=predictor_names_to_update,
                        reflective_dataset=reflective_dataset_concrete,
                    ),
                )

                new_texts = self.propose_new_texts(
                    curr_prog, reflective_dataset, predictor_names_to_update
                )

                notify_callbacks(
                    self.callbacks,
                    "on_proposal_end",
                    ProposalEndEvent(
                        iteration=i,
                        new_instructions=new_texts,
                    ),
                )

                for pname, text in new_texts.items():
                    self.logger.log(f"Iteration {i} [k={k}]: Proposed new text for {pname}: {text}")

                # --- Dedup check ---
                text_hash = hash(tuple(sorted(new_texts.items())))
                if text_hash in seen_hashes:
                    self.logger.log(f"Iteration {i} [k={k}]: Duplicate candidate, skipping evaluation.")
                    # Record NaN-like sentinel for deduped candidates
                    all_k_scores.append(float("-inf"))
                    continue

                seen_hashes.add(text_hash)
                unique_candidates += 1

                # --- Create new candidate ---
                new_candidate = curr_prog.copy()
                for pname, text in new_texts.items():
                    assert pname in new_candidate, f"{pname} missing in candidate"
                    new_candidate[pname] = text

                # --- Evaluate new candidate ---
                notify_callbacks(
                    self.callbacks,
                    "on_evaluation_start",
                    EvaluationStartEvent(
                        iteration=i,
                        candidate_idx=None,
                        batch_size=len(minibatch),
                        capture_traces=False,
                        parent_ids=[curr_prog_id],
                        inputs=minibatch,
                        is_seed_candidate=False,
                    ),
                )

                outputs_by_id, scores_by_id, objective_by_id, actual_evals_count = state.cached_evaluate_full(
                    new_candidate, subsample_ids, self.trainset.fetch, evaluator
                )
                new_scores = [scores_by_id[eid] for eid in subsample_ids]
                outputs = [outputs_by_id[eid] for eid in subsample_ids]

                notify_callbacks(
                    self.callbacks,
                    "on_evaluation_end",
                    EvaluationEndEvent(
                        iteration=i,
                        candidate_idx=None,
                        scores=new_scores,
                        has_trajectories=False,
                        parent_ids=[curr_prog_id],
                        outputs=outputs,
                        trajectories=None,
                        objective_scores=(
                            [objective_by_id[eid] for eid in subsample_ids] if objective_by_id else None
                        ),
                        is_seed_candidate=False,
                    ),
                )

                # Increment evals for THIS candidate
                state.increment_evals(actual_evals_count)

                new_sum = sum(new_scores)
                all_k_scores.append(new_sum)

                self.logger.log(
                    f"Iteration {i} [k={k}]: Candidate score sum = {new_sum} "
                    f"(best so far = {best_sum})"
                )

                if new_sum > best_sum:
                    best_sum = new_sum
                    best_candidate = new_candidate
                    best_scores = new_scores
                    best_k_index = k

            # --- No valid candidates found ---
            if best_candidate is None or best_scores is None:
                self.logger.log(f"Iteration {i}: No valid candidate found across {K} attempts. Skipping.")
                return None

            # --- Record trace and metrics ---
            state.full_program_trace[-1]["new_subsample_scores"] = best_scores

            self.experiment_tracker.log_metrics(
                {
                    "subsample/before": subsample_before,
                    "subsample/after": best_sum,
                    "total_metric_calls": state.total_num_evals,
                    "best_of_k/k_value": K,
                    "best_of_k/unique_candidates": unique_candidates,
                    "best_of_k/winning_k_index": best_k_index,
                },
                step=i,
            )

            # --- Build metadata ---
            metadata: dict[str, Any] = {
                "mutation_candidates": K,
                "unique_candidates": unique_candidates,
                "winning_k_index": best_k_index,
                "all_k_scores": all_k_scores,
            }

            # Record metrics via direct callback (bypasses engine's event system)
            if self._best_of_k_cb is not None:
                self._best_of_k_cb.record_iteration(
                    iteration=i,
                    k_value=K,
                    unique_candidates=unique_candidates,
                    winning_k_index=best_k_index,
                    all_k_scores=all_k_scores,
                    best_score=best_sum,
                )

            # Use appropriate tag
            tag = "best_of_k" if K > 1 else "reflective_mutation"

            return CandidateProposal(
                candidate=best_candidate,
                parent_program_ids=[curr_prog_id],
                subsample_indices=subsample_ids,
                subsample_scores_before=eval_curr.scores,
                subsample_scores_after=best_scores,
                tag=tag,
                metadata=metadata,
            )

        except Exception as e:
            self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
            self.logger.log(traceback.format_exc())
            return None
