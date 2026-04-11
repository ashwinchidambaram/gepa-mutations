"""contrastive_reflection mutation: inject successful candidate snippets into reflection."""

from contrastive_reflection.config import ContrastiveReflectionConfig
from contrastive_reflection.contrastive_search import ContrastiveTrainIndex, find_contrastive_candidates
from contrastive_reflection.injection import inject_contrastive_snippets
from contrastive_reflection.proposer import ContrastiveReflectionProposer
from contrastive_reflection.runner import run_contrastive_reflection

__all__ = [
    "ContrastiveReflectionConfig",
    "ContrastiveReflectionProposer",
    "ContrastiveTrainIndex",
    "find_contrastive_candidates",
    "inject_contrastive_snippets",
    "run_contrastive_reflection",
]
