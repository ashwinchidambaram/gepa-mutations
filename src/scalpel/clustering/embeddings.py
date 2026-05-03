"""BGE-small embedding wrapper for SCALPEL failure-mode clustering.

Phase 5 of SCALPEL.  Wraps ``sentence-transformers``'s
``BAAI/bge-small-en-v1.5`` (384-d, ~33.4M params) with lazy model loading
so importing this module is cheap and tests can monkeypatch
:class:`SentenceTransformer` to avoid downloading the real model.

Public surface:

* :data:`DEFAULT_MODEL` — model identifier consumed by ``SentenceTransformer``.
* :data:`EMBEDDING_DIM` — output dimensionality (``384``).
* :class:`BGEEmbedder` — lazy-loading wrapper exposing :meth:`embed` and
  :meth:`embed_one`.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

__all__ = ["BGEEmbedder", "DEFAULT_MODEL", "EMBEDDING_DIM"]

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


class BGEEmbedder:
    """Wraps BGE-small-en-v1.5.

    The model is loaded lazily on first :meth:`embed` / :meth:`embed_one`
    call so importing this module — or constructing the embedder — does not
    trigger a ~130 MB download.  CPU is fine on the runner VM; pass
    ``device="cuda:0"`` etc. to override.
    """

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        self.model_name = model
        self.device = device
        self._model: SentenceTransformer | None = None  # lazy-loaded

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts.

        Returns shape ``(len(texts), EMBEDDING_DIM)`` of normalized float32
        vectors.  An empty input yields a ``(0, EMBEDDING_DIM)`` array
        without loading the model.
        """
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._ensure_loaded()
        assert self._model is not None  # for type-checkers
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text, returning a 1-D ``(EMBEDDING_DIM,)`` array."""
        return self.embed([text])[0]
