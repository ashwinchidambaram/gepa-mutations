"""contrastive_synthesis mutation: distill contrastive pairs into abstract principles."""

from contrastive_synthesis.callbacks import ContrastiveSynthesisCallback
from contrastive_synthesis.proposer import ContrastiveSynthesisProposer
from contrastive_synthesis.runner import run_contrastive_synthesis
from contrastive_synthesis.synthesizer import synthesize

__all__ = [
    "ContrastiveSynthesisProposer",
    "ContrastiveSynthesisCallback",
    "run_contrastive_synthesis",
    "synthesize",
]
