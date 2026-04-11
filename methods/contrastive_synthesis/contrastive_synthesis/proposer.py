"""ContrastiveSynthesisProposer: extends ContrastiveReflectionProposer with a synthesis step.

After finding contrastive pairs (via find_contrastive_candidates), instead of injecting
raw candidate snippets into the reflective dataset, calls the reflection LM to DISTILL the
pairs into a single abstract improvement principle. The principle (not raw pairs) is then
injected into the <side_info> of each reflective dataset entry.

This adds exactly one extra LLM call per iteration (~500 tokens).
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

from contrastive_reflection.config import ContrastiveReflectionConfig
from contrastive_reflection.contrastive_search import (
    ContrastiveTrainIndex,
    find_contrastive_candidates,
)
from contrastive_reflection.proposer import ContrastiveReflectionProposer
from contrastive_synthesis.callbacks import ContrastiveSynthesisCallback
from contrastive_synthesis.synthesizer import synthesize


def _inject_principle_into_dataset(
    reflective_dataset: dict[str, list[dict[str, Any]]],
    principle: str,
    components_to_update: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Inject a synthesized principle into the reflective dataset as side_info.

    Appends a single entry with a ``synthesis_principle`` key to each
    component's reflective dataset. This gets rendered into the
    ``<side_info>`` section of the reflection prompt by the existing
    InstructionProposalSignature.prompt_renderer().

    If the principle is empty, returns the dataset unchanged.

    Args:
        reflective_dataset: Mapping from component name to list of feedback dicts.
        principle: Synthesized improvement principle string.
        components_to_update: Component names being mutated this iteration.

    Returns:
        Augmented reflective dataset with principle appended for each component.
    """
    if not principle:
        return reflective_dataset

    augmented: dict[str, list[dict[str, Any]]] = {}
    for component_name, dataset_entries in reflective_dataset.items():
        augmented_entries = list(dataset_entries)
        if component_name in components_to_update:
            augmented_entries.append(
                {
                    "synthesis_principle": (
                        f"Key improvement principle distilled from better-performing variants:\n"
                        f"{principle}"
                    )
                }
            )
        augmented[component_name] = augmented_entries
    return augmented


class ContrastiveSynthesisProposer(ContrastiveReflectionProposer):
    """Extends ContrastiveReflectionProposer with a synthesis distillation step.

    After finding contrastive pairs and building the reflective dataset, calls
    the reflection LM to synthesize an abstract improvement principle from the
    pairs. Injects the principle (not raw snippets) into the reflective dataset.

    When no contrastive pairs are found, or synthesis returns an empty string,
    falls back to the standard reflective dataset (no injection).

    Args:
        contrastive_config: Contrastive search configuration (num_pairs, min_gap, etc.).
        All other args: forwarded to ContrastiveReflectionProposer / ReflectiveMutationProposer.
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
        contrastive_config: ContrastiveReflectionConfig,
        reflection_lm: LanguageModel | None = None,
        reflection_prompt_template: str | dict[str, str] | None = None,
        custom_candidate_proposer: ProposalFn | None = None,
        callbacks: list[GEPACallback] | None = None,
    ) -> None:
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
            contrastive_config=contrastive_config,
            reflection_lm=reflection_lm,
            reflection_prompt_template=reflection_prompt_template,
            custom_candidate_proposer=custom_candidate_proposer,
            callbacks=callbacks,
        )

        # Find ContrastiveSynthesisCallback in callbacks list (if present)
        self._synthesis_cb: ContrastiveSynthesisCallback | None = None
        for cb in (callbacks or []):
            if isinstance(cb, ContrastiveSynthesisCallback):
                self._synthesis_cb = cb
                break

    def propose(self, state: GEPAState) -> CandidateProposal | None:
        """Propose a new candidate with contrastive synthesis injection.

        Replicates ContrastiveReflectionProposer.propose() verbatim, with the
        synthesis step replacing raw contrastive snippet injection.

        Key difference:
            - ContrastiveReflectionProposer injects raw candidate snippets.
            - ContrastiveSynthesisProposer calls reflection_lm to synthesize
              an abstract principle, then injects the principle as side_info.
        """
        i = state.i + 1

        # =====================================================================
        # Step 1: Select candidate
        # =====================================================================
        curr_prog_id = self.candidate_selector.select_candidate_idx(state)
        curr_prog = state.program_candidates[curr_prog_id]
        state.full_program_trace[-1]["selected_program_candidate"] = curr_prog_id
        self.logger.log(
            f"Iteration {i}: Selected program {curr_prog_id} score: "
            f"{state.program_full_scores_val_set[curr_prog_id]}"
        )

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
            {
                "iteration": i,
                "selected_program_candidate": curr_prog_id,
                "total_metric_calls": state.total_num_evals,
            },
            step=i,
        )

        # =====================================================================
        # Step 2: Sample minibatch
        # =====================================================================
        subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids
        minibatch = self.trainset.fetch(subsample_ids)

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
        # Step 3: Evaluate current program with traces
        # =====================================================================
        curr_parent_ids = [
            p for p in state.parent_program_for_candidate[curr_prog_id] if p is not None
        ]
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

        # Update cache
        if state.evaluation_cache is not None:
            objective_scores_list = (
                list(eval_curr.objective_scores) if eval_curr.objective_scores else None
            )
            state.evaluation_cache.put_batch(
                curr_prog, subsample_ids, eval_curr.outputs, eval_curr.scores, objective_scores_list
            )

        # =====================================================================
        # Step 4: Skip checks
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
        # Step 6: Build reflective dataset + contrastive synthesis injection
        # =====================================================================
        principle = ""
        num_contrastive_pairs_found = 0
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(
                curr_prog, eval_curr, predictor_names_to_update
            )

            # Convert to concrete types for callback
            reflective_dataset_concrete: dict[str, list[dict[str, Any]]] = {
                k: [dict(item) for item in v] for k, v in reflective_dataset.items()
            }

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

            # ===== CONTRASTIVE SYNTHESIS INJECTION POINT =====
            # 1. Update ContrastiveTrainIndex with current candidate's scores
            self.contrastive_index.update(
                curr_prog_id, curr_prog, subsample_ids, eval_curr.scores
            )

            # 2. Find contrastive candidates (CPU-only, no LLM calls)
            current_scores = dict(zip(subsample_ids, eval_curr.scores))
            contrastive_pairs = find_contrastive_candidates(
                self.contrastive_index,
                curr_prog_id,
                current_scores,
                self.contrastive_config.num_contrastive_pairs,
                self.contrastive_config.min_score_gap,
            )

            num_contrastive_pairs_found = len(contrastive_pairs)
            self.logger.log(
                f"Iteration {i}: Contrastive search found {num_contrastive_pairs_found} pairs "
                f"(requested {self.contrastive_config.num_contrastive_pairs})"
            )

            # 3. SYNTHESIS STEP: distill pairs into an abstract principle (1 extra LLM call)
            principle = synthesize(contrastive_pairs, self.reflection_lm)

            if principle:
                self.logger.log(
                    f"Iteration {i}: Synthesized principle ({len(principle)} chars): "
                    f"{principle[:200]}{'...' if len(principle) > 200 else ''}"
                )
            else:
                self.logger.log(
                    f"Iteration {i}: No principle synthesized "
                    f"({'no pairs' if not contrastive_pairs else 'synthesis failed'})"
                )

            # 4. Inject the principle (not raw snippets) into the reflective dataset
            augmented_dataset = _inject_principle_into_dataset(
                reflective_dataset_concrete,
                principle,
                predictor_names_to_update,
            )

            # Record synthesis metrics
            if self._synthesis_cb is not None:
                self._synthesis_cb.record_synthesis(
                    iteration=i,
                    principle=principle,
                    n_pairs=num_contrastive_pairs_found,
                )
            # ===== END CONTRASTIVE SYNTHESIS INJECTION =====

            # Notify proposal start (with augmented dataset)
            notify_callbacks(
                self.callbacks,
                "on_proposal_start",
                ProposalStartEvent(
                    iteration=i,
                    parent_candidate=curr_prog,
                    components=predictor_names_to_update,
                    reflective_dataset=augmented_dataset,
                ),
            )

            # Use the augmented dataset for proposal
            new_texts = self.propose_new_texts(
                curr_prog, augmented_dataset, predictor_names_to_update
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
                self.logger.log(f"Iteration {i}: Proposed new text for {pname}: {text}")

        except Exception as e:
            self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
            self.logger.log(traceback.format_exc())
            return None

        # =====================================================================
        # Step 7: Create candidate and evaluate on same minibatch
        # =====================================================================
        new_candidate = curr_prog.copy()
        for pname, text in new_texts.items():
            assert pname in new_candidate, f"{pname} missing in candidate"
            new_candidate[pname] = text

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

        outputs_by_id, scores_by_id, objective_by_id, actual_evals_count = (
            state.cached_evaluate_full(
                new_candidate, subsample_ids, self.trainset.fetch, evaluator
            )
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
                    [objective_by_id[eid] for eid in subsample_ids]
                    if objective_by_id
                    else None
                ),
                is_seed_candidate=False,
            ),
        )

        state.increment_evals(actual_evals_count)
        state.full_program_trace[-1]["new_subsample_scores"] = new_scores

        new_sum = sum(new_scores)
        self.experiment_tracker.log_metrics(
            {
                "subsample/before": subsample_before,
                "subsample/after": new_sum,
                "total_metric_calls": state.total_num_evals,
                "contrastive_synthesis/pairs_found": num_contrastive_pairs_found,
                "contrastive_synthesis/principle_length": len(principle),
            },
            step=i,
        )

        return CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[curr_prog_id],
            subsample_indices=subsample_ids,
            subsample_scores_before=eval_curr.scores,
            subsample_scores_after=new_scores,
            tag="contrastive_synthesis",
            metadata={
                "num_contrastive_pairs_found": num_contrastive_pairs_found,
                "principle": principle,
                "contrastive_pairs_used": [
                    {
                        "candidate_idx": p["candidate_idx"],
                        "example_id": p["example_id"],
                        "current_score": p["current_score"],
                        "contrastive_score": p["contrastive_score"],
                        "score_gap": p["score_gap"],
                    }
                    for p in contrastive_pairs
                ],
            },
        )
