"""Configuration for the failure-stratified K mutation."""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class FailureStratifiedConfig(BaseModel):
    """Configuration for failure-stratified K mutation.

    Controls how failing examples are partitioned across K candidates.
    When enabled, each candidate sees a different subset of failures,
    promoting diversity in the proposed mutations.
    """

    mutation_candidates: int = 3  # K value
    use_failure_stratified_k: bool = True  # Enable stratification
    perfect_score: float = 1.0  # Score threshold for "passing"

    @model_validator(mode="after")
    def validate_k(self):
        if self.use_failure_stratified_k and self.mutation_candidates <= 1:
            raise ValueError(
                "use_failure_stratified_k=True requires mutation_candidates > 1, "
                f"got {self.mutation_candidates}"
            )
        return self
