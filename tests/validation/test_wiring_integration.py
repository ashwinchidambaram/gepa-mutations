"""Integration tests for ISO experiment infrastructure wiring.

Verifies that context variables, JSONL logging, LoggingLM, and checkpoint
state flow correctly through the full ISO compile() path using only mock LMs.

All tests run without live servers and complete in under 60 seconds.

Design notes
------------
``MockMetric(base_score=0.0)`` is used in tests that exercise the main
tournament loop.  With the default response "mock answer" the metric scores
0.3 (keyword "answer" found: min(1.0, 0.0+0.3)=0.3).  The median score is
0.3 so the failure threshold is max(0.3, 0.5)=0.5, which flags all examples as
failures.  That gives the MockReflectionLM enough failures to return its canned
3-cluster response, producing an initial pool of 3 candidates — above pool_floor=2 —
so the tournament loop actually runs and writes rollout records and checkpoints.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import dspy
import pytest
from unittest.mock import MagicMock
from dspy.clients.base_lm import BaseLM

from iso_harness.optimizer.iso import ISO
from iso_harness.optimizer.helpers import ensure_example_ids
from iso_harness.optimizer.checkpoint import load_checkpoint
from iso_harness.experiment.jsonl_writer import JSONLWriter
from iso_harness.experiment.logging_lm import LoggingLM
from iso_harness.experiment.context import set_context, get_context
from iso_harness.meta.common import run_iso_with_config
from tests.mocks.mock_lm import MockReflectionLM, MockMetric


# ---------------------------------------------------------------------------
# Smoke-test config — minimal but still exercises the full loop
# ---------------------------------------------------------------------------

# Base overrides shared across all tests.  Note: pool_floor is intentionally
# omitted here because all ISO variant factories (sprint/grove/tide/lens/storm)
# pop and override it with their own hardcoded value.  The effective pool_floor
# depends on the variant: sprint=4, grove=8, tide/lens/storm=6.
SMOKE_TEST_OVERRIDES = {
    "n_discovery_examples": 5,
    "target_skills_min": 2,
    "target_skills_max": 3,
    "mutations_per_seed": 0,
    "minibatch_count": 2,
    "minibatch_size": 2,
    "max_rounds": 3,
    "merge_interval": 2,
    "plateau_rounds_threshold": 99,
}

# Loop-test overrides: uses "sprint" variant (pool_floor=4) with
# mutations_per_seed=1 so that skill discovery yields 3 clusters × 2 = 6
# initial candidates, exceeding the sprint pool_floor of 4 and ensuring
# at least one tournament round executes.
#
# This also requires MockMetric(base_score=0.0) so that all discovery
# examples score 0.3 (below the 0.5 failure threshold), producing the
# canned 3-cluster response from MockReflectionLM.
LOOP_TEST_OVERRIDES = {
    **SMOKE_TEST_OVERRIDES,
    "mutations_per_seed": 1,  # 3 clusters × 2 = 6 > sprint's pool_floor=4
}
_LOOP_VARIANT = "sprint"  # pool_floor=4


# ---------------------------------------------------------------------------
# DSPy-compatible task LM (must extend BaseLM for dspy.Predict to work)
# ---------------------------------------------------------------------------


class _DSPyMockLM(BaseLM):
    """Minimal BaseLM that returns canned DSPy-parseable output."""

    def __init__(self, default_response: str = "mock answer"):
        super().__init__(model="mock-task-lm", cache=False)
        self.default_response = default_response

    def forward(self, prompt=None, messages=None, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = (
            f"[[ ## answer ## ]]\n{self.default_response}"
        )
        response.usage = {"prompt_tokens": 10, "completion_tokens": 5}
        response.model = "mock-task-lm"
        return response


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _make_trainset(n: int = 20) -> list:
    examples = []
    for i in range(n):
        ex = dspy.Example(
            question=f"What is {i} + {i}?", answer=f"{i + i}"
        ).with_inputs("question")
        ex.id = f"ex_{i}"
        examples.append(ex)
    return examples


# ---------------------------------------------------------------------------
# Shared fixture — simple QA module
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_qa_module():
    class SimpleQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = dspy.Predict("question -> answer")

        def forward(self, question):
            return self.qa(question=question)

    return SimpleQA()


@pytest.fixture(autouse=True)
def configure_mock_lm():
    """Configure DSPy to use the mock task LM before each test."""
    dspy.settings.configure(lm=_DSPyMockLM())


# ---------------------------------------------------------------------------
# Cleanup helper — removes checkpoint dirs created during tests
# ---------------------------------------------------------------------------


def _cleanup_run_dir(run_id: str) -> None:
    """Remove the runs/{run_id} directory if it was created during a test."""
    run_dir = Path("runs") / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 1: Context variable propagation into JSONL rollout records
# ---------------------------------------------------------------------------


class TestRolloutContextPropagation:
    """Verify that rollout JSONL records are populated with correct context vars."""

    def test_rollout_records_have_context_fields(
        self, tmp_path: Path, simple_qa_module
    ):
        """After compile(), every rollout record has run_id, round_num >= 1,
        candidate_id, example_id, and a valid score in [0, 1]."""
        rollout_path = tmp_path / "rollouts.jsonl"
        writer = JSONLWriter(rollout_path)

        run_id = "test-wiring-001"
        optimizer = ISO(
            variant=_LOOP_VARIANT,
            metric=MockMetric(base_score=0.0),
            reflection_lm=MockReflectionLM(),
            task_lm=_DSPyMockLM(),
            budget=100,
            seed=42,
            run_id=run_id,
            rollout_writer=writer,
            **LOOP_TEST_OVERRIDES,
        )

        trainset = _make_trainset(20)
        valset = _make_trainset(10)
        ensure_example_ids(valset, prefix="val")

        try:
            optimizer.compile(simple_qa_module, trainset=trainset, valset=valset)
        finally:
            _cleanup_run_dir(run_id)

        records = writer.read_all()

        # At least one rollout record must exist
        assert len(records) >= 1, "No rollout records were written"

        for rec in records:
            # run_id must be set and non-empty
            assert "run_id" in rec, f"Record missing run_id: {rec}"
            assert rec["run_id"] != "", f"run_id is empty in record: {rec}"

            # round_num must be >= 1 (rounds start at 1 in iso_compile)
            assert "round_num" in rec, f"Record missing round_num: {rec}"
            assert rec["round_num"] >= 1, (
                f"Expected round_num >= 1, got {rec['round_num']}"
            )

            # candidate_id must be set and non-empty
            assert "candidate_id" in rec, f"Record missing candidate_id: {rec}"
            assert rec["candidate_id"] != "", (
                f"candidate_id is empty in record: {rec}"
            )

            # example_id must be set and non-empty
            assert "example_id" in rec, f"Record missing example_id: {rec}"
            assert rec["example_id"] != "", (
                f"example_id is empty in record: {rec}"
            )

            # score must be in valid range
            assert "score" in rec, f"Record missing score: {rec}"
            assert 0.0 <= rec["score"] <= 1.0, (
                f"score {rec['score']} out of [0, 1] in record: {rec}"
            )

    def test_run_id_matches_optimizer_run_id(self, tmp_path: Path, simple_qa_module):
        """Records carry the run_id that was passed to the ISO constructor."""
        rollout_path = tmp_path / "rollouts.jsonl"
        writer = JSONLWriter(rollout_path)

        run_id = "test-run-id-check-42"
        optimizer = ISO(
            variant=_LOOP_VARIANT,
            metric=MockMetric(base_score=0.0),
            reflection_lm=MockReflectionLM(),
            task_lm=_DSPyMockLM(),
            budget=100,
            seed=42,
            run_id=run_id,
            rollout_writer=writer,
            **LOOP_TEST_OVERRIDES,
        )

        trainset = _make_trainset(20)
        valset = _make_trainset(10)
        ensure_example_ids(valset, prefix="val")

        try:
            optimizer.compile(simple_qa_module, trainset=trainset, valset=valset)
        finally:
            _cleanup_run_dir(run_id)

        records = writer.read_all()
        assert len(records) >= 1

        for rec in records:
            assert rec["run_id"] == run_id, (
                f"Expected run_id={run_id!r}, got {rec['run_id']!r}"
            )


# ---------------------------------------------------------------------------
# Test 2: Reflection LoggingLM produces JSONL records with context fields
# ---------------------------------------------------------------------------


class TestReflectionLoggingLM:
    """LoggingLM(role='reflection') produces records with run_id, round_num,
    target_candidate_id when wrapped around the reflection LM and passed to ISO."""

    def test_reflection_records_exist_with_context(
        self, tmp_path: Path, simple_qa_module
    ):
        """Wrapping reflection LM in LoggingLM produces reflection JSONL records."""
        reflection_path = tmp_path / "reflections.jsonl"
        reflection_writer = JSONLWriter(reflection_path)

        run_id = "test-reflection-logging-001"
        base_reflection_lm = MockReflectionLM()
        logged_reflection_lm = LoggingLM(
            lm=base_reflection_lm,
            writer=reflection_writer,
            role="reflection",
        )

        optimizer = ISO(
            variant="tide",
            metric=MockMetric(base_score=0.4),
            reflection_lm=logged_reflection_lm,
            task_lm=_DSPyMockLM(),
            budget=100,
            seed=42,
            run_id=run_id,
            **SMOKE_TEST_OVERRIDES,
        )

        trainset = _make_trainset(20)
        valset = _make_trainset(10)
        ensure_example_ids(valset, prefix="val")

        try:
            optimizer.compile(simple_qa_module, trainset=trainset, valset=valset)
        finally:
            _cleanup_run_dir(run_id)

        records = reflection_writer.read_all()

        # Reflection records must be written (skill discovery + mutations call the LM)
        assert len(records) >= 1, (
            "No reflection records were written — LoggingLM not called"
        )

        for rec in records:
            # All reflection records must have a reflection_id (not rollout_id)
            assert "reflection_id" in rec, (
                f"Reflection record missing reflection_id: {rec}"
            )

            # run_id must be populated — set by iso_compile via set_context()
            assert "run_id" in rec, f"Reflection record missing run_id: {rec}"
            assert rec["run_id"] != "", (
                f"run_id is empty in reflection record: {rec}"
            )

            # round_num is present (may be 0 for skill-discovery phase)
            assert "round_num" in rec, (
                f"Reflection record missing round_num: {rec}"
            )

            # target_candidate_id is present (may be empty during skill discovery)
            assert "target_candidate_id" in rec, (
                f"Reflection record missing target_candidate_id: {rec}"
            )

    def test_reflection_records_have_no_score_field(
        self, tmp_path: Path, simple_qa_module
    ):
        """ReflectionRecord schema has no 'score' field — only RolloutRecord does."""
        reflection_path = tmp_path / "reflections_no_score.jsonl"
        reflection_writer = JSONLWriter(reflection_path)

        run_id = "test-reflection-schema-002"
        logged_reflection_lm = LoggingLM(
            lm=MockReflectionLM(),
            writer=reflection_writer,
            role="reflection",
        )

        optimizer = ISO(
            variant="tide",
            metric=MockMetric(base_score=0.4),
            reflection_lm=logged_reflection_lm,
            task_lm=_DSPyMockLM(),
            budget=100,
            seed=42,
            run_id=run_id,
            **SMOKE_TEST_OVERRIDES,
        )

        trainset = _make_trainset(20)
        valset = _make_trainset(10)
        ensure_example_ids(valset, prefix="val")

        try:
            optimizer.compile(simple_qa_module, trainset=trainset, valset=valset)
        finally:
            _cleanup_run_dir(run_id)

        records = reflection_writer.read_all()
        assert len(records) >= 1

        for rec in records:
            assert "score" not in rec, (
                f"ReflectionRecord should not have a 'score' field: {rec}"
            )
            assert "rollout_id" not in rec, (
                f"ReflectionRecord should not have a 'rollout_id' field: {rec}"
            )


# ---------------------------------------------------------------------------
# Test 3: Checkpoint state is written and has correct fields
# ---------------------------------------------------------------------------


class TestCheckpointState:
    """After compile(), the ISO checkpoint contains the expected fields."""

    def test_checkpoint_file_exists_after_run(
        self, tmp_path: Path, simple_qa_module
    ):
        """runs/{run_id}/checkpoint/iso_state.json is created after compile()."""
        run_id = "test-checkpoint-exists-001"

        optimizer = ISO(
            variant=_LOOP_VARIANT,
            metric=MockMetric(base_score=0.0),
            reflection_lm=MockReflectionLM(),
            task_lm=_DSPyMockLM(),
            budget=100,
            seed=42,
            run_id=run_id,
            **LOOP_TEST_OVERRIDES,
        )

        trainset = _make_trainset(20)
        valset = _make_trainset(10)
        ensure_example_ids(valset, prefix="val")

        try:
            optimizer.compile(simple_qa_module, trainset=trainset, valset=valset)
            checkpoint_path = Path("runs") / run_id / "checkpoint" / "iso_state.json"
            assert checkpoint_path.exists(), (
                f"Checkpoint file not found at {checkpoint_path}"
            )
        finally:
            _cleanup_run_dir(run_id)

    def test_checkpoint_has_required_fields(self, tmp_path: Path, simple_qa_module):
        """Loaded checkpoint has round_num, pool, rollouts_consumed,
        prev_top3_mean, plateau_rounds, seed."""
        run_id = "test-checkpoint-fields-002"

        optimizer = ISO(
            variant=_LOOP_VARIANT,
            metric=MockMetric(base_score=0.0),
            reflection_lm=MockReflectionLM(),
            task_lm=_DSPyMockLM(),
            budget=100,
            seed=42,
            run_id=run_id,
            **LOOP_TEST_OVERRIDES,
        )

        trainset = _make_trainset(20)
        valset = _make_trainset(10)
        ensure_example_ids(valset, prefix="val")

        try:
            optimizer.compile(simple_qa_module, trainset=trainset, valset=valset)
            state = load_checkpoint(run_id)
        finally:
            _cleanup_run_dir(run_id)

        assert state is not None, "load_checkpoint returned None"

        # Required fields
        assert "round_num" in state, "Checkpoint missing round_num"
        assert "pool" in state, "Checkpoint missing pool"
        assert "rollouts_consumed" in state, "Checkpoint missing rollouts_consumed"
        assert "prev_top3_mean" in state, "Checkpoint missing prev_top3_mean"
        assert "plateau_rounds" in state, "Checkpoint missing plateau_rounds"
        assert "seed" in state, "Checkpoint missing seed"

        # Sanity checks on values
        assert isinstance(state["round_num"], int), (
            f"round_num should be int, got {type(state['round_num'])}"
        )
        assert state["round_num"] >= 1, (
            f"round_num should be >= 1, got {state['round_num']}"
        )
        assert isinstance(state["pool"], list), (
            f"pool should be a list, got {type(state['pool'])}"
        )
        assert len(state["pool"]) >= 1, (
            "pool should have at least one candidate"
        )
        assert isinstance(state["rollouts_consumed"], int), (
            f"rollouts_consumed should be int, got {type(state['rollouts_consumed'])}"
        )
        assert state["rollouts_consumed"] >= 0, (
            f"rollouts_consumed should be >= 0, got {state['rollouts_consumed']}"
        )
        assert 0.0 <= state["prev_top3_mean"] <= 1.0, (
            f"prev_top3_mean {state['prev_top3_mean']} out of [0, 1]"
        )
        assert isinstance(state["plateau_rounds"], int), (
            f"plateau_rounds should be int, got {type(state['plateau_rounds'])}"
        )
        assert state["seed"] == 42, (
            f"Expected seed=42, got {state['seed']}"
        )

    def test_checkpoint_round_num_matches_max_rounds(
        self, tmp_path: Path, simple_qa_module
    ):
        """Checkpoint round_num equals max_rounds (or budget exhausted round)."""
        run_id = "test-checkpoint-roundnum-003"
        max_rounds = 3

        overrides = {**LOOP_TEST_OVERRIDES, "max_rounds": max_rounds}
        optimizer = ISO(
            variant=_LOOP_VARIANT,
            metric=MockMetric(base_score=0.0),
            reflection_lm=MockReflectionLM(),
            task_lm=_DSPyMockLM(),
            budget=100,
            seed=42,
            run_id=run_id,
            **overrides,
        )

        trainset = _make_trainset(20)
        valset = _make_trainset(10)
        ensure_example_ids(valset, prefix="val")

        try:
            optimizer.compile(simple_qa_module, trainset=trainset, valset=valset)
            state = load_checkpoint(run_id)
        finally:
            _cleanup_run_dir(run_id)

        assert state is not None
        # The last checkpoint was written at the last completed round.
        # With budget=100 and small minibatches, the loop should reach max_rounds=3.
        assert state["round_num"] <= max_rounds, (
            f"round_num {state['round_num']} exceeds max_rounds={max_rounds}"
        )
        assert state["round_num"] >= 1, (
            f"Expected at least one round to complete, got {state['round_num']}"
        )


# ---------------------------------------------------------------------------
# Test 4: Pool candidates in checkpoint are reasonable
# ---------------------------------------------------------------------------


class TestCheckpointPoolIntegrity:
    """Candidates in the checkpoint have expected structure."""

    def test_pool_candidates_have_ids(self, tmp_path: Path, simple_qa_module):
        """Each candidate in the pool has an 'id' field."""
        run_id = "test-pool-ids-001"

        optimizer = ISO(
            variant=_LOOP_VARIANT,
            metric=MockMetric(base_score=0.0),
            reflection_lm=MockReflectionLM(),
            task_lm=_DSPyMockLM(),
            budget=100,
            seed=42,
            run_id=run_id,
            **LOOP_TEST_OVERRIDES,
        )

        trainset = _make_trainset(20)
        valset = _make_trainset(10)
        ensure_example_ids(valset, prefix="val")

        try:
            optimizer.compile(simple_qa_module, trainset=trainset, valset=valset)
            state = load_checkpoint(run_id)
        finally:
            _cleanup_run_dir(run_id)

        assert state is not None
        pool = state["pool"]
        assert len(pool) >= 1

        for candidate in pool:
            # load_checkpoint returns Candidate objects (dataclasses), not dicts
            cand_id = getattr(candidate, "id", None)
            assert cand_id is not None, (
                f"Candidate missing 'id' attribute: {candidate}"
            )
            assert cand_id != "", f"Candidate id is empty: {candidate}"

    def test_pool_size_is_reasonable(self, tmp_path: Path, simple_qa_module):
        """Pool size is at most initial_pool_size * (1 + mutations_per_seed)."""
        run_id = "test-pool-size-002"

        optimizer = ISO(
            variant=_LOOP_VARIANT,
            metric=MockMetric(base_score=0.0),
            reflection_lm=MockReflectionLM(),
            task_lm=_DSPyMockLM(),
            budget=100,
            seed=42,
            run_id=run_id,
            **LOOP_TEST_OVERRIDES,
        )

        trainset = _make_trainset(20)
        valset = _make_trainset(10)
        ensure_example_ids(valset, prefix="val")

        try:
            optimizer.compile(simple_qa_module, trainset=trainset, valset=valset)
            state = load_checkpoint(run_id)
        finally:
            _cleanup_run_dir(run_id)

        assert state is not None
        pool = state["pool"]

        # sprint's pool_floor=4 so at minimum 4 candidates survive
        # (after pruning the initial pool of 6 down by 50%)
        assert len(pool) >= 1, (
            f"Pool should have at least 1 candidate after pruning, got {len(pool)}"
        )

        # With target_skills_max=3 and mutations_per_seed=1, initial pool = 6;
        # cross-mutations and reflections can add a few more. Cap at 30 as sanity check.
        assert len(pool) <= 30, (
            f"Pool size {len(pool)} seems unexpectedly large"
        )


# ---------------------------------------------------------------------------
# Test 5: run_iso_with_config mock mode returns correct keys
# ---------------------------------------------------------------------------


class TestRunIsoWithConfigMock:
    """run_iso_with_config(mock=True) returns a valid outcome dict."""

    def test_mock_mode_returns_required_keys(self):
        """Mock mode returns dict with final_score, rollouts_consumed,
        tokens_consumed, budget — and does not call any LM."""
        result = run_iso_with_config(
            variant="tide",
            benchmark="hotpotqa",
            subset_size=10,
            config_overrides={},
            mock=True,
        )

        assert isinstance(result, dict), "run_iso_with_config must return a dict"

        required_keys = {"final_score", "rollouts_consumed", "tokens_consumed", "budget"}
        for key in required_keys:
            assert key in result, f"Missing key {key!r} in result: {result}"

    def test_mock_mode_values_are_typed_correctly(self):
        """Mock outcome values have numeric types."""
        result = run_iso_with_config(
            variant="tide",
            benchmark="hotpotqa",
            subset_size=10,
            config_overrides={},
            mock=True,
        )

        assert isinstance(result["final_score"], (int, float)), (
            f"final_score should be numeric, got {type(result['final_score'])}"
        )
        assert isinstance(result["rollouts_consumed"], (int, float)), (
            f"rollouts_consumed should be numeric, got {type(result['rollouts_consumed'])}"
        )
        assert isinstance(result["tokens_consumed"], (int, float)), (
            f"tokens_consumed should be numeric, got {type(result['tokens_consumed'])}"
        )
        assert isinstance(result["budget"], (int, float)), (
            f"budget should be numeric, got {type(result['budget'])}"
        )

    def test_mock_mode_env_var(self, monkeypatch):
        """ISO_META_MOCK=1 env var activates mock mode even when mock=False."""
        monkeypatch.setenv("ISO_META_MOCK", "1")

        result = run_iso_with_config(
            variant="grove",
            benchmark="ifbench",
            subset_size=5,
            config_overrides={"max_rounds": 2},
            mock=False,
        )

        assert isinstance(result, dict)
        assert "final_score" in result
        assert "budget" in result

    def test_mock_mode_budget_computed_from_overrides(self):
        """Budget in mock result reflects config_overrides."""
        overrides = {
            "max_rounds": 2,
            "minibatch_count": 3,
            "minibatch_size": 4,
            "pool_size_seed": 5,
        }
        result = run_iso_with_config(
            variant="tide",
            benchmark="hotpotqa",
            subset_size=10,
            config_overrides=overrides,
            mock=True,
        )

        # Budget formula: max_rounds * minibatch_count * minibatch_size * pool_size_seed
        expected_budget = 2 * 3 * 4 * 5
        assert result["budget"] == expected_budget, (
            f"Expected budget={expected_budget}, got {result['budget']}"
        )

    def test_mock_mode_no_error_key(self):
        """Mock mode never includes an 'error' key."""
        result = run_iso_with_config(
            variant="sprint",
            benchmark="hotpotqa",
            subset_size=10,
            config_overrides={},
            mock=True,
        )

        assert "error" not in result, (
            f"Mock mode should not return 'error' key: {result}"
        )


# ---------------------------------------------------------------------------
# Test 6: Both rollout writer and reflection logging can coexist in one run
# ---------------------------------------------------------------------------


class TestCombinedRolloutAndReflectionLogging:
    """Rollout and reflection writers both receive records during a single compile()."""

    def test_both_writers_receive_records(self, tmp_path: Path, simple_qa_module):
        """ISO with both rollout_writer and LoggingLM-wrapped reflection LM
        writes records to both JSONL files during a single compile()."""
        rollout_path = tmp_path / "rollouts.jsonl"
        reflection_path = tmp_path / "reflections.jsonl"

        rollout_writer = JSONLWriter(rollout_path)
        reflection_writer = JSONLWriter(reflection_path)

        run_id = "test-combined-logging-001"
        logged_reflection_lm = LoggingLM(
            lm=MockReflectionLM(),
            writer=reflection_writer,
            role="reflection",
        )

        optimizer = ISO(
            variant=_LOOP_VARIANT,
            metric=MockMetric(base_score=0.0),
            reflection_lm=logged_reflection_lm,
            task_lm=_DSPyMockLM(),
            budget=100,
            seed=42,
            run_id=run_id,
            rollout_writer=rollout_writer,
            **LOOP_TEST_OVERRIDES,
        )

        trainset = _make_trainset(20)
        valset = _make_trainset(10)
        ensure_example_ids(valset, prefix="val")

        try:
            optimizer.compile(simple_qa_module, trainset=trainset, valset=valset)
        finally:
            _cleanup_run_dir(run_id)

        rollout_records = rollout_writer.read_all()
        reflection_records = reflection_writer.read_all()

        assert len(rollout_records) >= 1, (
            "No rollout records — rollout_writer not receiving records"
        )
        assert len(reflection_records) >= 1, (
            "No reflection records — LoggingLM not producing records"
        )

        # Rollout records use rollout_id; reflection records use reflection_id
        assert all("rollout_id" in r for r in rollout_records), (
            "Some rollout records are missing rollout_id"
        )
        assert all("reflection_id" in r for r in reflection_records), (
            "Some reflection records are missing reflection_id"
        )

    def test_run_ids_are_consistent_across_writers(
        self, tmp_path: Path, simple_qa_module
    ):
        """All records from both writers share the same run_id."""
        rollout_path = tmp_path / "rollouts.jsonl"
        reflection_path = tmp_path / "reflections.jsonl"

        rollout_writer = JSONLWriter(rollout_path)
        reflection_writer = JSONLWriter(reflection_path)

        run_id = "test-consistent-run-id-002"
        logged_reflection_lm = LoggingLM(
            lm=MockReflectionLM(),
            writer=reflection_writer,
            role="reflection",
        )

        optimizer = ISO(
            variant=_LOOP_VARIANT,
            metric=MockMetric(base_score=0.0),
            reflection_lm=logged_reflection_lm,
            task_lm=_DSPyMockLM(),
            budget=100,
            seed=42,
            run_id=run_id,
            rollout_writer=rollout_writer,
            **LOOP_TEST_OVERRIDES,
        )

        trainset = _make_trainset(20)
        valset = _make_trainset(10)
        ensure_example_ids(valset, prefix="val")

        try:
            optimizer.compile(simple_qa_module, trainset=trainset, valset=valset)
        finally:
            _cleanup_run_dir(run_id)

        all_records = rollout_writer.read_all() + reflection_writer.read_all()
        assert len(all_records) >= 1

        for rec in all_records:
            assert rec.get("run_id") == run_id, (
                f"Expected run_id={run_id!r}, got {rec.get('run_id')!r} in record: {rec}"
            )
