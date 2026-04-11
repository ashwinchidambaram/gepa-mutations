"""Configuration for the contrastive reflection mutation."""

from pydantic import BaseModel


class ContrastiveReflectionConfig(BaseModel):
    """Configuration for contrastive reflection mutation.

    Controls how many contrastive pairs to inject, the minimum score gap
    required to include a pair, and how candidate text is presented.
    """

    # Number of top-performing examples to include as contrastive pairs
    num_contrastive_pairs: int = 3
    # Minimum score difference between contrastive and current to include
    min_score_gap: float = 0.1
    # Whether to include the contrastive candidate's full text or just a snippet
    include_full_text: bool = False
    # Maximum length of contrastive snippet (characters)
    max_snippet_length: int = 500
