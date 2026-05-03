"""Phase 9 integration tests for :mod:`scalpel.optimizer`.

Covers:

* Protocol satisfaction (``Optimizer`` and ``Checkpointable``).
* Auto-instantiation of subsystems.
* End-to-end 2-iter mock run.
* ``run_context`` wiring from the race callback.
* Cluster targeting rotation.
* System-level Pareto pool (Q9).
* Lesson book wiring without prompt mutation (Q3).
* Checkpoint roundtrip with JSON-only persistence.
* Length-cap retry behaviour.
* ``max_iters`` termination.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import scalpel.checkpoint
from iso_harness.experiment.context import get_context
from iso_harness.experiment.protocols import Checkpointable, Optimizer
from scalpel.checkpoint import load_state, save_state
from scalpel.clustering.kmeans import FailureClusterer
from scalpel.edits.apply import apply
from scalpel.edits.grammar import StructuredPrompt
from scalpel.edits.span_index import parse
from scalpel.optimizer import SCALPEL, Candidate, IterationLog

# ---------------------------------------------------------------- stub LMs


class _Usage:
    def __init__(self, prompt: int = 10, completion: int = 5) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class StubTaskLM:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.calls = 0
        self.responses = responses or ["answer-{i}"]
        self._last_usage = _Usage(prompt=10, completion=5)
        self.context_snapshots: list[dict] = []

    def __call__(self, prompt) -> str:
        self.context_snapshots.append(get_context())
        r = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return r.replace("{i}", str(self.calls))


class StubReflectLM:
    def __init__(self, edit_list_json: str) -> None:
        self.calls = 0
        self.json = edit_list_json
        self._last_usage = _Usage(prompt=100, completion=50)

    def reflect(self, prompt, guided_json_schema=None) -> str:
        self.calls += 1
        return self.json

    def __call__(self, prompt) -> str:
        return self.reflect(prompt)


_TINY_EDIT_LIST = json.dumps(
    {
        "edits": [
            {"operation": "APPEND", "target_span": "S5", "content": "Be careful."}
        ],
        "lessons": [],
    }
)

_EDIT_WITH_LESSONS = json.dumps(
    {
        "edits": [
            {"operation": "APPEND", "target_span": "S5", "content": "Verify."}
        ],
        "lessons": ["Always verify retrieval evidence"],
    }
)

_HUGE_EDIT_LIST = json.dumps(
    {
        "edits": [
            {
                "operation": "APPEND",
                "target_span": "S5",
                "content": " ".join(["verify"] * 80),
            }
        ],
        "lessons": [],
    }
)


def _make_dataset(n: int = 5) -> list[dict]:
    return [
        {"id": f"i-{i}", "input": f"q-{i}", "gold": f"answer-{i + 1}"}
        for i in range(n)
    ]


# ---------------------------------------------------------- Protocol checks


def test_scalpel_satisfies_optimizer_protocol():
    opt = SCALPEL(task_lm=StubTaskLM(), reflect_lm=StubReflectLM(_TINY_EDIT_LIST))
    assert isinstance(opt, Optimizer)


def test_scalpel_satisfies_checkpointable_protocol():
    opt = SCALPEL(task_lm=StubTaskLM(), reflect_lm=StubReflectLM(_TINY_EDIT_LIST))
    assert isinstance(opt, Checkpointable)


def test_constructor_auto_instantiates_subsystems():
    opt = SCALPEL(task_lm=StubTaskLM(), reflect_lm=StubReflectLM(_TINY_EDIT_LIST))
    assert opt.embedder is not None
    assert opt.clusterer is not None
    assert opt.targeting_history is not None
    assert opt.race is not None
    assert opt.surrogate is not None
    assert opt.lesson_book is not None


# ----------------------------------------------------- 2-iter mock run path


def test_two_iter_mock_run_completes():
    task = StubTaskLM()
    opt = SCALPEL(
        task_lm=task,
        reflect_lm=StubReflectLM(_TINY_EDIT_LIST),
        K=4,
        rungs=(2, 4),
        max_iters=2,
        seed=1,
    )
    seed_id = {"id": None}

    def metric(gold, pred):  # noqa: ARG001
        cid = get_context().get("candidate_id", "")
        if seed_id["id"] is None:
            seed_id["id"] = cid
        return 0.5 if cid == seed_id["id"] else 0.7

    student = {"default": "You are a helpful assistant."}
    result = opt.compile(
        student,
        trainset=_make_dataset(5),
        valset=_make_dataset(5),
        metric=metric,
    )
    assert isinstance(result, dict)
    assert "default" in result
    assert isinstance(result["default"], StructuredPrompt)
    assert len(opt.iteration_logs) >= 1
    assert len(opt.candidates) >= 2  # seed + at least one accepted child


# --------------------------------------------------------- iter logging


def test_iteration_log_recorded_per_iter():
    opt = SCALPEL(
        task_lm=StubTaskLM(),
        reflect_lm=StubReflectLM(_TINY_EDIT_LIST),
        K=2,
        rungs=(2, 4),
        max_iters=2,
        seed=1,
    )
    opt.compile(
        {"default": "You are a helpful assistant."},
        trainset=_make_dataset(3),
        valset=_make_dataset(3),
        metric=lambda g, p: 0.5,
    )
    assert len(opt.iteration_logs) == 2
    for log in opt.iteration_logs:
        assert isinstance(log, IterationLog)


def test_run_context_set_for_each_rollout():
    task = StubTaskLM()
    opt = SCALPEL(
        task_lm=task,
        reflect_lm=StubReflectLM(_TINY_EDIT_LIST),
        K=2,
        rungs=(2,),
        max_iters=1,
        seed=2,
    )
    opt.compile(
        {"default": "You are a helpful assistant."},
        trainset=_make_dataset(3),
        valset=_make_dataset(3),
        metric=lambda g, p: 0.5,
    )
    race_snaps = [s for s in task.context_snapshots if s.get("phase") == "race"]
    assert race_snaps, "expected at least one task call inside a race phase"
    for snap in race_snaps:
        assert snap.get("run_id")
        assert snap.get("round_num", -1) >= 1
        assert snap.get("candidate_id")


# --------------------------------------------------- cluster rotation (Q1)


def test_target_cluster_rotates_across_iters():
    opt = SCALPEL(
        task_lm=StubTaskLM(),
        reflect_lm=StubReflectLM(_TINY_EDIT_LIST),
        K=2,
        rungs=(2,),
        max_iters=3,
        seed=3,
    )
    rng = np.random.default_rng(0)
    embs = []
    for cluster_id in range(8):
        center = np.zeros(384, dtype=np.float32)
        center[cluster_id] = 10.0
        for _ in range(12):
            embs.append(
                center + 0.01 * rng.standard_normal(384).astype(np.float32)
            )
    opt.clusterer.add(np.stack(embs))
    opt.clusterer.recluster()

    opt.compile(
        {"default": "You are a helpful assistant."},
        trainset=_make_dataset(3),
        valset=_make_dataset(3),
        metric=lambda g, p: 0.5,
    )
    ids = [log.target_cluster_id for log in opt.iteration_logs]
    assert ids[0] != ids[2], (
        f"expected cluster rotation across 3 iters, got: {ids}"
    )


# ------------------------------------------------ Pareto pool / Q9 system


def test_accepted_child_added_to_pool_with_score():
    opt = SCALPEL(
        task_lm=StubTaskLM(),
        reflect_lm=StubReflectLM(_TINY_EDIT_LIST),
        K=2,
        rungs=(2,),
        max_iters=1,
        seed=4,
    )
    seed_id = {"id": None}

    def metric(gold, pred):  # noqa: ARG001
        cid = get_context().get("candidate_id", "")
        if seed_id["id"] is None:
            seed_id["id"] = cid
        return 0.4 if cid == seed_id["id"] else 0.9

    opt.compile(
        {"default": "You are a helpful assistant."},
        trainset=_make_dataset(3),
        valset=_make_dataset(3),
        metric=metric,
    )
    accepted = [c for c in opt.candidates if c.parent_id is not None]
    assert accepted, "expected at least one accepted child"
    for child in accepted:
        assert child.pareto_score > 0


def test_q9_system_level_candidate_holds_all_module_prompts():
    opt = SCALPEL(
        task_lm=StubTaskLM(),
        reflect_lm=StubReflectLM(_TINY_EDIT_LIST),
        K=2,
        rungs=(2,),
        max_iters=2,
        seed=5,
    )
    student = {"a": "You are module A.", "b": "You are module B."}
    seed_id = {"id": None}

    def metric(gold, pred):  # noqa: ARG001
        cid = get_context().get("candidate_id", "")
        if seed_id["id"] is None:
            seed_id["id"] = cid
        return 0.4 if cid == seed_id["id"] else 0.9

    opt.compile(
        student,
        trainset=_make_dataset(3),
        valset=_make_dataset(3),
        metric=metric,
    )
    seed = opt.candidates[0]
    assert set(seed.prompts.keys()) == {"a", "b"}
    accepted = [c for c in opt.candidates if c.parent_id is not None]
    assert accepted
    for child in accepted:
        assert set(child.prompts.keys()) == {"a", "b"}


# ------------------------------------------------------- Lesson book (Q3)


def test_lessons_added_after_acceptance():
    opt = SCALPEL(
        task_lm=StubTaskLM(),
        reflect_lm=StubReflectLM(_EDIT_WITH_LESSONS),
        K=2,
        rungs=(2,),
        max_iters=1,
        seed=6,
    )
    seed_id = {"id": None}

    def metric(gold, pred):  # noqa: ARG001
        cid = get_context().get("candidate_id", "")
        if seed_id["id"] is None:
            seed_id["id"] = cid
        return 0.3 if cid == seed_id["id"] else 0.95

    opt.compile(
        {"default": "You are a helpful assistant."},
        trainset=_make_dataset(3),
        valset=_make_dataset(3),
        metric=metric,
    )
    assert len(opt.lesson_book.lessons) >= 1


def test_lessons_do_not_mutate_deployed_prompt():
    """Q3 invariant: a lesson string never auto-mutates the deployed prompt."""
    opt = SCALPEL(
        task_lm=StubTaskLM(),
        reflect_lm=StubReflectLM(_EDIT_WITH_LESSONS),
        K=2,
        rungs=(2,),
        max_iters=1,
        seed=7,
    )
    seed_id = {"id": None}

    def metric(gold, pred):  # noqa: ARG001
        cid = get_context().get("candidate_id", "")
        if seed_id["id"] is None:
            seed_id["id"] = cid
        return 0.3 if cid == seed_id["id"] else 0.95

    opt.compile(
        {"default": "You are a helpful assistant."},
        trainset=_make_dataset(3),
        valset=_make_dataset(3),
        metric=metric,
    )
    seed = opt.candidates[0]
    accepted = [c for c in opt.candidates if c.parent_id is not None]
    assert accepted
    for child in accepted:
        recomputed = apply(seed.prompts["default"], child.edits_applied)
        assert child.prompts["default"].raw_text == recomputed.raw_text
        assert (
            "Always verify retrieval evidence"
            not in child.prompts["default"].raw_text
        )


# ---------------------------------------------------- checkpoint roundtrip


def _run_short_optimizer(tmp_seed: int = 9) -> SCALPEL:
    opt = SCALPEL(
        task_lm=StubTaskLM(),
        reflect_lm=StubReflectLM(_EDIT_WITH_LESSONS),
        K=2,
        rungs=(2,),
        max_iters=2,
        seed=tmp_seed,
    )
    seed_id = {"id": None}

    def metric(gold, pred):  # noqa: ARG001
        cid = get_context().get("candidate_id", "")
        if seed_id["id"] is None:
            seed_id["id"] = cid
        return 0.3 if cid == seed_id["id"] else 0.95

    opt.compile(
        {"default": "You are a helpful assistant."},
        trainset=_make_dataset(3),
        valset=_make_dataset(3),
        metric=metric,
    )
    return opt


def test_save_load_roundtrip_preserves_candidates(tmp_path):
    opt = _run_short_optimizer(tmp_seed=11)
    save_state(opt, tmp_path)

    fresh = SCALPEL(
        task_lm=StubTaskLM(),
        reflect_lm=StubReflectLM(_EDIT_WITH_LESSONS),
        K=2,
        rungs=(2,),
        max_iters=2,
        seed=99,
    )
    load_state(fresh, tmp_path)
    assert len(fresh.candidates) == len(opt.candidates)
    assert len(fresh.iteration_logs) == len(opt.iteration_logs)
    assert len(fresh.lesson_book.lessons) == len(opt.lesson_book.lessons)


def test_save_load_roundtrip_no_forbidden_libs():
    """Forbidden serialization libraries must not appear in the source."""
    src = inspect.getsource(scalpel.checkpoint)
    forbidden = [
        "import " + "p" + "ickle",
        "import " + "j" + "oblib",
        "import " + "d" + "ill",
        "import " + "c" + "loudpickle",
        "from " + "p" + "ickle",
        "from " + "j" + "oblib",
        "from " + "d" + "ill",
        "from " + "c" + "loudpickle",
    ]
    for token in forbidden:
        assert token not in src, (
            f"forbidden serializer token in source: {token!r}"
        )


def test_save_writes_json_only(tmp_path):
    opt = _run_short_optimizer(tmp_seed=12)
    save_state(opt, tmp_path)
    bad_extensions = {".pkl", ".pickle", ".joblib", ".dill", ".npy", ".npz"}
    allowed_extensions = {".json", ".txt"}
    for f in Path(tmp_path).rglob("*"):
        if f.is_file():
            assert f.suffix not in bad_extensions, (
                f"forbidden binary persistence file: {f}"
            )
            assert f.suffix in allowed_extensions, (
                f"unexpected extension {f.suffix!r} for {f}"
            )


# ---------------------------------------------------- length-cap retry path


def test_length_cap_skip_does_not_crash():
    opt = SCALPEL(
        task_lm=StubTaskLM(),
        reflect_lm=StubReflectLM(_HUGE_EDIT_LIST),
        K=4,
        rungs=(2,),
        max_iters=1,
        seed=8,
    )
    opt.compile(
        {"default": "You are a helpful assistant."},
        trainset=_make_dataset(3),
        valset=_make_dataset(3),
        metric=lambda g, p: 0.5,
    )
    log = opt.iteration_logs[0]
    assert log.n_candidates_after_apply < opt.K


# --------------------------------------------------------- termination


def test_max_iters_terminates():
    opt = SCALPEL(
        task_lm=StubTaskLM(),
        reflect_lm=StubReflectLM(_TINY_EDIT_LIST),
        K=2,
        rungs=(2,),
        max_iters=1,
        seed=10,
    )
    opt.compile(
        {"default": "You are a helpful assistant."},
        trainset=_make_dataset(3),
        valset=_make_dataset(3),
        metric=lambda g, p: 0.5,
    )
    assert len(opt.iteration_logs) == 1


# --------------------------------------------------------- helpers / smoke


def test_candidate_and_iteration_log_are_pydantic():
    c = Candidate(id="x", prompts={"default": parse("Test prompt.")})
    log = IterationLog(
        iter_num=1,
        target_cluster_id=0,
        target_module="default",
        n_proposals=1,
        n_candidates_after_apply=1,
        race_total_rollouts=2,
        race_total_skipped=0,
        survivor_score=0.5,
        accepted=False,
        cluster_k=1,
        cluster_silhouette=0.0,
        n_active_lessons=0,
    )
    Candidate.model_validate(json.loads(c.model_dump_json()))
    IterationLog.model_validate(json.loads(log.model_dump_json()))


def test_clusterer_default_is_failure_clusterer():
    opt = SCALPEL(task_lm=StubTaskLM(), reflect_lm=StubReflectLM(_TINY_EDIT_LIST))
    assert isinstance(opt.clusterer, FailureClusterer)


_ = pytest
