"""FailureStratifiedKProposer: partition failing examples across K candidates.

Extends BestOfKProposer by giving each of the K candidates a different
subset of failing examples from the reflective dataset. This creates
diversity in the mutations by ensuring each candidate focuses on
different failure patterns.

When there are fewer failing examples than K, or when stratification
is disabled, behavior falls back to standard best-of-K (all candidates
see the full reflective dataset).
"""

from __future__ import annotations

import traceback
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
from gepa.strategies.batch_sampler import BatchSampler

from best_of_k.proposer import BestOfKProposer
from failure_stratified_k.config import FailureStratifiedConfig


class FailureStratifiedKProposer(BestOfKProposer):
    """Reflective mutation proposer that partitions failures across K candidates.

    Extends BestOfKProposer by stratifying failing examples in the reflective
    dataset across K candidates. Each candidate receives ALL passing examples
    plus a different subset of failing examples (assigned via round-robin on
    score-sorted failing indices).

    When stratification is disabled or there are fewer failing examples than K,
    behavior is identical to standard BestOfKProposer.
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
        mutation_candidates: int = 3,
        stratified_config: FailureStratifiedConfig | None = None,
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
            mutation_candidates=mutation_candidates,
            reflection_lm=reflection_lm,
            reflection_prompt_template=reflection_prompt_template,
            custom_candidate_proposer=custom_candidate_proposer,
            callbacks=callbacks,
        )
        self.stratified_config = stratified_config or FailureStratifiedConfig(
            mutation_candidates=mutation_candidates,
        )
        # Ensure stratified config uses the same perfect_score as the proposer
        # to avoid divergence between skip-perfect and partition threshold
        self.stratified_config.perfect_score = self.perfect_score or 1.0

    def _partition_reflective_dataset(
        self,
        reflective_dataset: dict[str, list[dict[str, Any]]],
        scores: list[float],
        k: int,
    ) -> list[dict[str, list[dict[str, Any]]]] | None:
        """Partition reflective dataset into K versions based on failure patterns.

        Identifies failing examples globally by example index (not per-component),
        sorts them by score ascending (worst failures first), and round-robin
        assigns them to K partitions. Each partition contains ALL passing examples
        plus its assigned subset of failing examples.

        Args:
            reflective_dataset: The full reflective dataset keyed by component name,
                where each value is a list of entries corresponding to minibatch examples.
            scores: Per-example scores from the current candidate evaluation.
            k: Total number of partitions (K value).

        Returns:
            A list of K reflective datasets if there are enough failing examples
            to stratify, or None to signal fallback to standard best-of-K behavior.
        """
        # Identify failing example indices
        failing_indices = [
            idx for idx, score in enumerate(scores)
            if score < self.stratified_config.perfect_score
        ]

        if len(failing_indices) < k:
            return None  # Not enough failures to stratify

        # Sort failing indices by score ascending (worst first)
        failing_indices.sort(key=lambda idx: scores[idx])

        # Round-robin assign to K partitions
        partitions: list[list[int]] = [[] for _ in range(k)]
        for i, idx in enumerate(failing_indices):
            partitions[i % k].append(idx)

        # Identify passing indices
        passing_indices = [
            idx for idx, score in enumerate(scores)
            if score >= self.stratified_config.perfect_score
        ]

        # Build K reflective datasets
        result = []
        for partition_failing in partitions:
            # This partition includes all passing + its assigned failing
            included_indices = set(passing_indices) | set(partition_failing)

            partitioned_dataset: dict[str, list[dict[str, Any]]] = {}
            for component_name, entries in reflective_dataset.items():
                # Filter entries to only included indices
                partitioned_dataset[component_name] = [
                    entries[idx] for idx in sorted(included_indices)
                    if idx < len(entries)
                ]

            result.append(partitioned_dataset)

        return result

    def propose(self, state: GEPAState) -> CandidateProposal | None:
        """Propose a new candidate using failure-stratified best-of-K selection.

        Steps 1-5 (candidate selection, minibatch sampling, current evaluation,
        skip checks, predictor selection) run exactly once — identical to the
        parent BestOfKProposer. The reflective dataset is built once, then
        partitioned into K versions (one per candidate) based on failure patterns.
        Each candidate in the K-loop sees a different subset of failing examples.

        When stratification is disabled or there are fewer failing examples than K,
        all candidates see the full reflective dataset (standard best-of-K behavior).
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
        # Steps 6-7: K-loop with failure stratification
        # =====================================================================
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(
                curr_prog, eval_curr, predictor_names_to_update
            )

            # Convert to concrete types for callback and partitioning
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

            # =================================================================
            # NEW: Partition reflective dataset if stratification is enabled
            # =================================================================
            partitioned_datasets: list[dict[str, list[dict[str, Any]]]] | None = None
            failure_stratified = False
            num_failing_examples = 0
            partition_sizes: list[int] = []

            if self.stratified_config.use_failure_stratified_k:
                # Count failing examples for metadata
                failing_indices = [
                    idx for idx, score in enumerate(eval_curr.scores)
                    if score < self.stratified_config.perfect_score
                ]
                num_failing_examples = len(failing_indices)

                partitioned_datasets = self._partition_reflective_dataset(
                    reflective_dataset_concrete, eval_curr.scores, K
                )
                if partitioned_datasets is not None:
                    failure_stratified = True
                    # Compute partition sizes (number of entries in first component as proxy)
                    if partitioned_datasets:
                        first_component = next(iter(reflective_dataset_concrete.keys()), None)
                        if first_component:
                            partition_sizes = [
                                len(pd.get(first_component, []))
                                for pd in partitioned_datasets
                            ]
                    self.logger.log(
                        f"Iteration {i}: Stratified {num_failing_examples} failing examples "
                        f"across {K} partitions (sizes: {partition_sizes})"
                    )
                else:
                    self.logger.log(
                        f"Iteration {i}: Not enough failing examples ({num_failing_examples}) "
                        f"for stratification, using full dataset"
                    )

            # Track best candidate across K attempts
            best_candidate: dict[str, str] | None = None
            best_scores: list[float] | None = None
            best_outputs: list[Any] | None = None
            best_objective_by_id: dict | None = None
            best_sum: float = float("-inf")
            best_k_index: int = 0
            all_k_scores: list[float] = []
            seen_hashes: set[int] = set()
            unique_candidates: int = 0
            total_actual_evals: int = 0

            for k in range(K):
                # Use partitioned dataset if available, otherwise full dataset
                if partitioned_datasets is not None:
                    k_reflective_dataset = partitioned_datasets[k]
                else:
                    k_reflective_dataset = reflective_dataset_concrete

                # --- Propose new texts ---
                # Build concrete version for callback (may differ per k when stratified)
                k_dataset_concrete: dict[str, list[dict[str, Any]]] = {
                    comp: [dict(item) for item in entries]
                    for comp, entries in k_reflective_dataset.items()
                }

                notify_callbacks(
                    self.callbacks,
                    "on_proposal_start",
                    ProposalStartEvent(
                        iteration=i,
                        parent_candidate=curr_prog,
                        components=predictor_names_to_update,
                        reflective_dataset=k_dataset_concrete,
                    ),
                )

                new_texts = self.propose_new_texts(
                    curr_prog, k_reflective_dataset, predictor_names_to_update
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
                def evaluator(b, c):
                    r = self.adapter.evaluate(b, c, capture_traces=False)
                    return r.outputs, r.scores, list(r.objective_scores) if r.objective_scores else None

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
                total_actual_evals += actual_evals_count

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
                    best_outputs = outputs
                    best_objective_by_id = objective_by_id
                    best_k_index = k

            # --- No valid candidates found ---
            if best_candidate is None or best_scores is None:
                self.logger.log(f"Iteration {i}: All {K} candidates were duplicates. Skipping.")
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
                    "failure_stratified": failure_stratified,
                    "num_failing_examples": num_failing_examples,
                },
                step=i,
            )

            # --- Build metadata ---
            metadata: dict[str, Any] = {
                "mutation_candidates": K,
                "unique_candidates": unique_candidates,
                "winning_k_index": best_k_index,
                "all_k_scores": all_k_scores,
                "failure_stratified": failure_stratified,
                "num_failing_examples": num_failing_examples,
                "partition_sizes": partition_sizes,
            }

            # Use appropriate tag
            if failure_stratified:
                tag = "failure_stratified_k"
            elif K > 1:
                tag = "best_of_k"
            else:
                tag = "reflective_mutation"

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
