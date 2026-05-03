"""SCALPEL lesson book store (Phase 8).

Append-only journal of short bullets distilled from failure-cluster reflection.
Per the SCALPEL addendum Q3, lessons live exclusively in the reflection LM's
context window: this module never mutates a deployed prompt.  The only path
from a lesson to a prompt mutation is an explicit edit emitted by the
reflection LM and run through the edit grammar pipeline
(:mod:`scalpel.edits`) -- entirely outside this file.

Public surface:

* :class:`Lesson` -- pydantic model for a single bullet.
* :class:`LessonBook` -- the journal itself.
* :data:`LessonStatus` -- ``Literal["active", "negative", "evicted"]``.
* :data:`DEFAULT_MAX_SIZE`, :data:`DEFAULT_DEDUP_TAU`,
  :data:`DEFAULT_UNUSED_TTL`, :data:`DEFAULT_MAX_TOKEN_LEN` -- defaults.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DEFAULT_DEDUP_TAU",
    "DEFAULT_MAX_SIZE",
    "DEFAULT_MAX_TOKEN_LEN",
    "DEFAULT_UNUSED_TTL",
    "Lesson",
    "LessonBook",
    "LessonStatus",
]


LessonStatus = Literal["active", "negative", "evicted"]

DEFAULT_MAX_SIZE: int = 24
DEFAULT_DEDUP_TAU: float = 0.85
DEFAULT_UNUSED_TTL: int = 8
DEFAULT_MAX_TOKEN_LEN: int = 30


class Lesson(BaseModel):
    """A single lesson bullet.

    Counters (``instances_fixed``, ``age``, ``status``) are mutable; the
    ``text`` is treated as immutable after construction by convention -- the
    book's API never rewrites it, only merges duplicates by incrementing the
    survivor's ``instances_fixed``.
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    cluster_origin: int
    instances_fixed: int = 0
    age: int = 0
    status: LessonStatus = "active"
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Whitespace-tokenize, keep first ``max_tokens`` tokens, rejoin."""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text.strip() if not tokens else " ".join(tokens)
    return " ".join(tokens[:max_tokens])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for 1-D float arrays.  Returns 0.0 on zero norm."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class LessonBook:
    """Append-only journal of lessons with cosine-similarity dedup.

    IMPORTANT: per addendum Q3, this class never mutates a prompt. Its sole
    job is to track lessons and render them as reflection-prompt context. The
    deployed prompt is updated only through the EDIT grammar pipeline.
    """

    def __init__(
        self,
        embedder,
        max_size: int = DEFAULT_MAX_SIZE,
        dedup_tau: float = DEFAULT_DEDUP_TAU,
        unused_ttl: int = DEFAULT_UNUSED_TTL,
        max_token_len: int = DEFAULT_MAX_TOKEN_LEN,
    ) -> None:
        self.embedder = embedder
        self.max_size = max_size
        self.dedup_tau = dedup_tau
        self.unused_ttl = unused_ttl
        self.max_token_len = max_token_len
        self._lessons: list[Lesson] = []
        self._embeddings: list[np.ndarray] = []

    @property
    def lessons(self) -> list[Lesson]:
        """A shallow copy of the lesson list.

        Mutating the returned list does not mutate the book; mutating the
        :class:`Lesson` objects inside *does* mutate the book (they are the
        same objects).
        """
        return list(self._lessons)

    def add(self, text: str, cluster_origin: int) -> Lesson:
        """Add a new lesson, deduping by cosine similarity.

        Steps:

        1. Truncate ``text`` to ``max_token_len`` whitespace tokens.
        2. Embed via ``self.embedder.embed_one``.
        3. If any active lesson has cosine >= ``dedup_tau``, increment its
           ``instances_fixed`` and return it (no new entry added).
        4. Otherwise append the new lesson + embedding.
        5. If size exceeds ``max_size`` after the append, evict by priority
           (``instances_fixed`` desc, ``age`` asc).
        """
        truncated = _truncate_to_tokens(text, self.max_token_len)
        embedding = np.asarray(self.embedder.embed_one(truncated))

        match_idx = self._dedup_check(embedding)
        if match_idx is not None:
            existing = self._lessons[match_idx]
            existing.instances_fixed += 1
            return existing

        lesson = Lesson(text=truncated, cluster_origin=cluster_origin)
        self._lessons.append(lesson)
        self._embeddings.append(embedding)

        if len(self._lessons) > self.max_size:
            self._evict()

        return lesson

    def mark_negative(self, lesson_id: str) -> bool:
        """Flip a lesson's status to ``"negative"``.  Returns success."""
        for lesson in self._lessons:
            if lesson.id == lesson_id:
                lesson.status = "negative"
                return True
        return False

    def increment_age_and_evict(self) -> int:
        """Bump ``age`` on every lesson; evict any lesson with
        ``age >= unused_ttl`` AND ``instances_fixed == 0``.

        Returns the number of evicted lessons.
        """
        for lesson in self._lessons:
            lesson.age += 1

        survivors: list[Lesson] = []
        survivor_embeddings: list[np.ndarray] = []
        evicted = 0
        for lesson, emb in zip(self._lessons, self._embeddings):
            if lesson.age >= self.unused_ttl and lesson.instances_fixed == 0:
                evicted += 1
                continue
            survivors.append(lesson)
            survivor_embeddings.append(emb)
        self._lessons = survivors
        self._embeddings = survivor_embeddings
        return evicted

    def increment_instances_fixed(self, lesson_id: str, n: int = 1) -> bool:
        """Increment ``instances_fixed`` on a lesson by ``n``.

        Caller reports that a rollout previously failing on the lesson's
        cluster now passes.  Returns True if the lesson was found.
        """
        for lesson in self._lessons:
            if lesson.id == lesson_id:
                lesson.instances_fixed += n
                return True
        return False

    def find_by_id(self, lesson_id: str) -> Optional[Lesson]:
        """Look up a lesson by id; returns None if absent."""
        for lesson in self._lessons:
            if lesson.id == lesson_id:
                return lesson
        return None

    def render(self, top_m: int | None = None) -> str:
        """Render active+negative lessons as a reflection-prompt block.

        Active lessons come first (ranked by ``instances_fixed`` desc, then
        most-recent first via ``age`` asc), followed by negatives as
        ``- AVOID: <text>``.  If ``top_m`` is provided, the total number of
        bullets returned is capped at ``top_m`` (active first).  Returns ``""``
        when the book is empty.
        """
        if not self._lessons:
            return ""

        active = [le for le in self._lessons if le.status == "active"]
        negative = [le for le in self._lessons if le.status == "negative"]

        active.sort(key=lambda le: (-le.instances_fixed, le.age))

        active_lines = [f"- {le.text}" for le in active]
        avoid_lines = [f"- AVOID: {le.text}" for le in negative]

        all_lines = active_lines + avoid_lines
        if top_m is not None:
            all_lines = all_lines[:top_m]
        return "\n".join(all_lines)

    def to_jsonable(self) -> list[dict]:
        """Serialize lesson list to a JSON-friendly form (no embeddings).

        Embeddings are recomputed by :meth:`from_jsonable` via the injected
        embedder.  ``created_at`` is rendered as an ISO-8601 string.
        """
        out: list[dict] = []
        for lesson in self._lessons:
            out.append(
                {
                    "id": lesson.id,
                    "text": lesson.text,
                    "cluster_origin": lesson.cluster_origin,
                    "instances_fixed": lesson.instances_fixed,
                    "age": lesson.age,
                    "status": lesson.status,
                    "created_at": lesson.created_at.isoformat(),
                }
            )
        return out

    def from_jsonable(self, items: list[dict]) -> None:
        """Restore lesson list from :meth:`to_jsonable` output.

        Does NOT clear an existing book; assumes called on a fresh instance.
        Re-embeds every lesson by calling ``embedder.embed_one(lesson.text)``.
        """
        for item in items:
            created_at_raw = item.get("created_at")
            if isinstance(created_at_raw, str):
                created_at = datetime.fromisoformat(created_at_raw)
            elif isinstance(created_at_raw, datetime):
                created_at = created_at_raw
            else:
                created_at = datetime.now(timezone.utc)

            lesson = Lesson(
                id=item["id"],
                text=item["text"],
                cluster_origin=int(item["cluster_origin"]),
                instances_fixed=int(item.get("instances_fixed", 0)),
                age=int(item.get("age", 0)),
                status=item.get("status", "active"),
                created_at=created_at,
            )
            embedding = np.asarray(self.embedder.embed_one(lesson.text))
            self._lessons.append(lesson)
            self._embeddings.append(embedding)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evict(self) -> None:
        """Drop the lowest-priority lessons until ``len <= max_size``.

        Priority key: ``instances_fixed`` DESC, then ``age`` ASC.  The
        worst-ranked lesson (smallest ``instances_fixed``, largest ``age``)
        is dropped first.  Insertion order breaks remaining ties: among
        equally-ranked candidates, the earlier-inserted lesson is dropped
        first ("oldest underperformer").
        """
        while len(self._lessons) > self.max_size:
            worst_idx = 0
            for idx in range(1, len(self._lessons)):
                cur = self._lessons[idx]
                best = self._lessons[worst_idx]
                if cur.instances_fixed < best.instances_fixed:
                    worst_idx = idx
                elif cur.instances_fixed == best.instances_fixed:
                    if cur.age > best.age:
                        worst_idx = idx
                    # If age also ties, keep the earlier index as worst
                    # (i.e. evict the oldest by insertion order first).
            del self._lessons[worst_idx]
            del self._embeddings[worst_idx]

    def _dedup_check(self, embedding: np.ndarray) -> Optional[int]:
        """Return the index of the first ACTIVE lesson with cosine
        >= ``self.dedup_tau``, or ``None`` if no match."""
        for idx, (lesson, emb) in enumerate(
            zip(self._lessons, self._embeddings)
        ):
            if lesson.status != "active":
                continue
            if _cosine(embedding, emb) >= self.dedup_tau:
                return idx
        return None
