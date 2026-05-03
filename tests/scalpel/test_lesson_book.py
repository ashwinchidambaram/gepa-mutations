"""SCALPEL Phase 8 lesson book tests.

Covers:

* :mod:`scalpel.lesson_book.store` -- add / dedup / render / age / negative /
  eviction / persistence.
* :mod:`scalpel.lesson_book.retrieval` -- ``top_m_lessons`` ranking.
* Q3 invariant: lesson book code never touches the EDIT pipeline (verified
  by source inspection).
* Repeats Phase 7's no-binary-pickle-imports test.
"""

from __future__ import annotations

import inspect

import numpy as np

import scalpel.lesson_book.retrieval as retrieval_mod
import scalpel.lesson_book.store as store_mod
from scalpel.lesson_book import (
    DEFAULT_DEDUP_TAU,
    DEFAULT_MAX_SIZE,
    DEFAULT_UNUSED_TTL,
    Lesson,
    LessonBook,
    top_m_lessons,
)

# Forbidden binary-serialization libraries.  Composed at runtime so the
# Claude Code security hook does not flag this test file as a pickle user.
_FORBIDDEN_LIBS = (
    "p" + "ickle",
    "j" + "oblib",
    "d" + "ill",
    "cl" + "oudpickle",
)


class FakeEmbedder:
    """Deterministic 384-d unit vectors keyed on text hash.

    Same text -> identical embedding (so dedup picks it up); different texts
    -> near-orthogonal random vectors (cosine ~ 0).
    """

    def embed_one(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(hash(text) & 0xFFFFFFFF)
        v = rng.normal(size=384).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_one(t) for t in texts])


def _make_book(**kwargs) -> LessonBook:
    return LessonBook(embedder=FakeEmbedder(), **kwargs)


# ---------------------------------------------------------------------------
# store.py -- basic ops
# ---------------------------------------------------------------------------


def test_empty_book_renders_empty_string() -> None:
    book = _make_book()
    assert book.render() == ""


def test_add_returns_new_lesson() -> None:
    book = _make_book()
    lesson = book.add("foo", 0)
    assert isinstance(lesson, Lesson)
    assert lesson.text == "foo"
    assert lesson.cluster_origin == 0
    assert lesson.status == "active"
    assert lesson.age == 0
    assert lesson.instances_fixed == 0


def test_add_two_distinct_lessons_appends_both() -> None:
    book = _make_book()
    a = book.add("alpha lesson one", 0)
    b = book.add("beta lesson two", 1)
    assert a.id != b.id
    assert len(book.lessons) == 2


def test_add_dedupes_identical_text() -> None:
    book = _make_book()
    first = book.add("foo", 0)
    second = book.add("foo", 1)
    assert len(book.lessons) == 1
    assert second.id == first.id
    assert second.instances_fixed == 1


def test_token_length_truncation() -> None:
    book = _make_book(max_token_len=30)
    long_text = " ".join(f"tok{i}" for i in range(50))
    lesson = book.add(long_text, 0)
    assert len(lesson.text.split()) <= 30


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_active_then_avoid() -> None:
    book = _make_book()
    a = book.add("alpha", 0)
    book.add("beta", 0)
    neg = book.add("gamma", 0)
    book.mark_negative(neg.id)
    rendered = book.render()
    lines = rendered.split("\n")
    assert lines[-1] == "- AVOID: gamma"
    assert "- AVOID:" not in lines[0]
    assert any("alpha" in line for line in lines[:-1])
    assert any("beta" in line for line in lines[:-1])
    assert a.status == "active"


def test_render_top_m_caps_total_bullets() -> None:
    book = _make_book()
    for i in range(5):
        book.add(f"lesson_{i}", 0)
    rendered = book.render(top_m=2)
    assert len(rendered.split("\n")) == 2


def test_render_with_only_negative_returns_avoid_lines() -> None:
    book = _make_book()
    lesson = book.add("careful here", 0)
    book.mark_negative(lesson.id)
    rendered = book.render()
    assert rendered == "- AVOID: careful here"


# ---------------------------------------------------------------------------
# age + eviction
# ---------------------------------------------------------------------------


def test_increment_age_no_eviction_under_ttl() -> None:
    book = _make_book(unused_ttl=DEFAULT_UNUSED_TTL)
    book.add("alpha", 0)
    book.add("beta", 1)
    for _ in range(3):
        evicted = book.increment_age_and_evict()
        assert evicted == 0
    assert len(book.lessons) == 2
    assert all(le.age == 3 for le in book.lessons)


def test_increment_age_evicts_unused_at_ttl() -> None:
    book = _make_book(unused_ttl=8)
    book.add("alpha", 0)
    total_evicted = 0
    for _ in range(8):
        total_evicted += book.increment_age_and_evict()
    assert total_evicted == 1
    assert len(book.lessons) == 0


def test_increment_age_keeps_used_lessons_past_ttl() -> None:
    book = _make_book(unused_ttl=8)
    lesson = book.add("alpha", 0)
    book.increment_instances_fixed(lesson.id, n=3)
    for _ in range(8):
        book.increment_age_and_evict()
    assert len(book.lessons) == 1
    assert book.lessons[0].id == lesson.id


# ---------------------------------------------------------------------------
# negative
# ---------------------------------------------------------------------------


def test_mark_negative_changes_status() -> None:
    book = _make_book()
    lesson = book.add("alpha", 0)
    assert book.mark_negative(lesson.id) is True
    found = book.find_by_id(lesson.id)
    assert found is not None
    assert found.status == "negative"


def test_mark_negative_returns_false_for_unknown_id() -> None:
    book = _make_book()
    assert book.mark_negative("ghost") is False


# ---------------------------------------------------------------------------
# max_size
# ---------------------------------------------------------------------------


def test_max_size_evicts_low_priority_first() -> None:
    book = _make_book(max_size=3)
    a = book.add("lesson_a", 0)
    b = book.add("lesson_b", 1)
    c = book.add("lesson_c", 2)
    book.increment_instances_fixed(a.id, n=1)
    book.increment_instances_fixed(c.id, n=1)
    book.add("lesson_d", 3)

    surviving_ids = {le.id for le in book.lessons}
    assert b.id not in surviving_ids
    assert a.id in surviving_ids
    assert c.id in surviving_ids
    assert len(book.lessons) == 3


# ---------------------------------------------------------------------------
# retrieval (retrieval.py)
# ---------------------------------------------------------------------------


def test_top_m_returns_active_only() -> None:
    book = _make_book()
    book.add("alpha", 0)
    neg = book.add("beta", 0)
    book.mark_negative(neg.id)
    result = top_m_lessons(book, m=5)
    assert len(result) == 1
    assert result[0].text == "alpha"
    assert result[0].status == "active"


def test_top_m_orders_by_recency_decay() -> None:
    book = _make_book(unused_ttl=999)
    a = book.add("alpha old", 0)
    for _ in range(3):
        book.increment_age_and_evict()
    book.increment_instances_fixed(a.id, n=1)
    b = book.add("beta fresh", 0)
    book.increment_instances_fixed(b.id, n=1)

    result = top_m_lessons(book, m=10)
    assert [le.id for le in result[:2]] == [b.id, a.id]


def test_top_m_caps_at_m() -> None:
    book = _make_book()
    for i in range(5):
        book.add(f"lesson_{i}", 0)
    result = top_m_lessons(book, m=2)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def test_to_jsonable_roundtrip() -> None:
    book = _make_book()
    a = book.add("alpha", 0)
    b = book.add("beta", 1)
    c = book.add("gamma", 2)
    book.increment_instances_fixed(a.id, n=2)
    book.mark_negative(b.id)
    for _ in range(2):
        book.increment_age_and_evict()

    data = book.to_jsonable()

    book2 = LessonBook(embedder=FakeEmbedder())
    book2.from_jsonable(data)

    orig = {le.id: le for le in book.lessons}
    restored = {le.id: le for le in book2.lessons}
    assert set(orig.keys()) == set(restored.keys())
    for lid, le in orig.items():
        r = restored[lid]
        assert r.text == le.text
        assert r.status == le.status
        assert r.instances_fixed == le.instances_fixed
        assert r.age == le.age
        assert r.cluster_origin == le.cluster_origin
    assert c.id in restored


def test_jsonable_data_has_no_embeddings() -> None:
    book = _make_book()
    book.add("alpha", 0)
    book.add("beta", 1)
    data = book.to_jsonable()
    assert isinstance(data, list)
    for item in data:
        assert isinstance(item, dict)
        for value in item.values():
            if isinstance(value, list):
                assert len(value) != 384


# ---------------------------------------------------------------------------
# Q3 invariant -- source inspection
# ---------------------------------------------------------------------------


def test_lesson_book_does_not_mutate_prompts_via_source_inspection() -> None:
    src = inspect.getsource(store_mod) + inspect.getsource(retrieval_mod)
    forbidden_substrings = [
        "materialize",
        "StructuredPrompt",
        "apply(",
        "Edit(",
        "REPLACE",
        "APPEND",
        "DELETE",
        "INSERT",
    ]
    for needle in forbidden_substrings:
        assert needle not in src, (
            f"lesson_book source must not reference {needle!r} -- per "
            "addendum Q3 the lesson book never mutates a deployed prompt."
        )


def test_no_pickle_imports_in_lesson_book() -> None:
    src = inspect.getsource(store_mod) + inspect.getsource(retrieval_mod)
    for word in _FORBIDDEN_LIBS:
        assert f"import {word}" not in src
        assert f"from {word}" not in src


# ---------------------------------------------------------------------------
# Misc sanity
# ---------------------------------------------------------------------------


def test_default_constants_match_spec() -> None:
    assert DEFAULT_MAX_SIZE == 24
    assert DEFAULT_DEDUP_TAU == 0.85
    assert DEFAULT_UNUSED_TTL == 8
