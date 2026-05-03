"""SCALPEL lesson book (Phase 8) -- append-only journal, no auto-promotion."""

from scalpel.lesson_book.retrieval import top_m_lessons
from scalpel.lesson_book.store import (
    DEFAULT_DEDUP_TAU,
    DEFAULT_MAX_SIZE,
    DEFAULT_UNUSED_TTL,
    Lesson,
    LessonBook,
    LessonStatus,
)

__all__ = [
    "DEFAULT_DEDUP_TAU",
    "DEFAULT_MAX_SIZE",
    "DEFAULT_UNUSED_TTL",
    "Lesson",
    "LessonBook",
    "LessonStatus",
    "top_m_lessons",
]
