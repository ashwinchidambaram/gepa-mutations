"""SCALPEL lesson book retrieval (Phase 8).

Selects the top-m active lessons for inclusion in a reflection prompt.  The
ranking is ``instances_fixed * 0.9 ** age`` -- recently-helpful lessons rank
above old ones with the same fix count, and unproven lessons
(``instances_fixed == 0``) rank below all proven ones.

Per addendum Q3, this module returns active lessons only; negative lessons
are surfaced separately via :meth:`scalpel.lesson_book.store.LessonBook.render`.
"""

from __future__ import annotations

from scalpel.lesson_book.store import Lesson, LessonBook

__all__ = ["top_m_lessons"]


_RECENCY_DECAY: float = 0.9


def top_m_lessons(book: LessonBook, m: int = 12) -> list[Lesson]:
    """Return up to ``m`` active lessons ranked by recency-decayed fix count.

    The score is ``instances_fixed * 0.9 ** age``.  Ties break in favour of
    smaller ``age`` (more recent), then earlier insertion order via the
    book's lesson list.
    """
    active = [le for le in book.lessons if le.status == "active"]

    def priority(lesson: Lesson) -> tuple[float, int]:
        score = lesson.instances_fixed * (_RECENCY_DECAY**lesson.age)
        return (-score, lesson.age)

    active.sort(key=priority)
    return active[:m]
