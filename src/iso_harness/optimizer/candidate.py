"""Runtime data structures for ISO optimizer candidates and traces.

These are plain dataclasses (not Pydantic) — they live on the hot path and must be
fast to construct and mutate. Use ``Candidate.to_record()`` to convert to the
Pydantic ``CandidateRecord`` for JSONL logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

__all__ = [
    "Candidate",
    "SkillCluster",
    "ModuleTrace",
    "MutationProposal",
]


@dataclass
class Candidate:
    """Runtime representation of a prompt candidate.

    Attributes:
        id: Unique identifier (UUID4 string).
        parent_ids: IDs of candidates this was derived from.
        birth_round: Round number when this candidate was created.
        birth_mechanism: How this candidate was created.
            One of: "seed" | "skill_discovery" | "mutation_*" |
            "cross_mutation_*" | "merge" | "initial_mutation".
        skill_category: Skill label associated with this candidate (if any).
        prompts_by_module: Maps DSPy module name → instruction string.
        score_history: List of (round_num, score) pairs.
        per_instance_scores: Maps example_id → score for the latest eval.
        pareto_frontier_rounds: Rounds where this candidate was on the frontier.
        death_round: Round when candidate was pruned (None if alive).
        death_reason: Why the candidate was pruned (None if alive).
        total_rollouts_consumed: Cumulative rollout count for this candidate.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    parent_ids: list[str] = field(default_factory=list)
    birth_round: int = 0
    birth_mechanism: str = "seed"
    skill_category: str | None = None
    prompts_by_module: dict[str, str] = field(default_factory=dict)
    score_history: list[tuple[int, float]] = field(default_factory=list)
    per_instance_scores: dict[str, float] = field(default_factory=dict)
    pareto_frontier_rounds: list[int] = field(default_factory=list)
    death_round: int | None = None
    death_reason: str | None = None
    total_rollouts_consumed: int = 0

    def to_record(self, run_id: str):
        """Convert to CandidateRecord for JSONL logging (Layer 3).

        Args:
            run_id: The experiment run ID to embed in the record.

        Returns:
            CandidateRecord Pydantic model ready for JSONL serialisation.
        """
        from iso_harness.experiment.schemas import CandidateRecord

        # Map birth_mechanism to CandidateRecord's Literal values:
        # "skill_discovery" | "reflection_mutation" | "cross_mutation" | "seed"
        _mechanism_map: dict[str, str] = {
            "seed": "seed",
            "skill_discovery": "skill_discovery",
            "initial_mutation": "reflection_mutation",
            "merge": "cross_mutation",
        }

        mapped = _mechanism_map.get(self.birth_mechanism)
        if mapped is None:
            if "cross_mutation" in self.birth_mechanism:
                mapped = "cross_mutation"
            elif "mutation" in self.birth_mechanism:
                mapped = "reflection_mutation"
            else:
                mapped = "seed"

        return CandidateRecord(
            candidate_id=self.id,
            run_id=run_id,
            parent_ids=list(self.parent_ids),
            birth_round=self.birth_round,
            birth_mechanism=mapped,  # type: ignore[arg-type]
            skill_category=self.skill_category,
            prompt_by_module=dict(self.prompts_by_module),
            score_history=list(self.score_history),
            per_instance_scores=dict(self.per_instance_scores),
            pareto_frontier_rounds=list(self.pareto_frontier_rounds),
            death_round=self.death_round,
            death_reason=self.death_reason,
            total_rollouts_consumed=self.total_rollouts_consumed,
        )


@dataclass
class SkillCluster:
    """A discovered skill cluster from inductive analysis.

    Attributes:
        label: Short skill name (e.g. "chain_of_thought").
        description: Human-readable description of the skill.
        target_module: DSPy module this skill targets (None = all modules).
        example_traces: Representative traces illustrating the skill.
    """

    label: str
    description: str
    target_module: str | None = None
    example_traces: list[ModuleTrace] = field(default_factory=list)


@dataclass
class ModuleTrace:
    """Execution trace for a single (candidate, example, module) triple.

    Attributes:
        example_id: Identifier for the input example.
        prediction: Raw DSPy prediction object (if available).
        score: Metric score for this prediction (0.0–1.0).
        feedback: Human/LM-generated feedback string.
        metadata: Arbitrary key-value metadata (latency, tokens, etc.).
        module_outputs: Named intermediate outputs from the DSPy program.
    """

    example_id: str
    prediction: Any = None
    score: float = 0.0
    feedback: str = ""
    metadata: dict = field(default_factory=dict)
    module_outputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class MutationProposal:
    """A proposed set of new prompts for a candidate.

    Attributes:
        candidate_id: ID of the candidate being mutated.
        new_prompts: Maps module name → proposed new instruction string.
        mechanism: Mutation strategy used to generate this proposal.
            One of: "per_candidate" | "population_level" | "pair_contrastive".
    """

    candidate_id: str
    new_prompts: dict[str, str] = field(default_factory=dict)
    mechanism: str = "per_candidate"
