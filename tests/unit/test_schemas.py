"""Comprehensive tests for ISO experiment 7-layer Pydantic schemas.

Tests cover:
- Roundtrip (create -> JSON -> deserialise -> equality) for every model
- Validation rejection for out-of-range / malformed values
- JSON output stability (expected keys present)
- schema_version default
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from iso_harness.experiment.schemas import (
    CandidateRecord,
    MetaOptimizerRecord,
    ReflectionRecord,
    RolloutRecord,
    RoundSummary,
    RunSummary,
    SystemTelemetryRecord,
)

# ---------------------------------------------------------------------------
# Helpers — canonical valid instances
# ---------------------------------------------------------------------------

_NOW = datetime.now(timezone.utc)


def _uid() -> str:
    return str(uuid.uuid4())


def _rollout(**overrides) -> RolloutRecord:
    defaults = dict(
        rollout_id=_uid(),
        run_id="run-abc",
        round_num=0,
        candidate_id="cand-1",
        module_name=None,
        example_id="ex-42",
        prompt="What is 2+2?",
        response="4",
        score=0.85,
        feedback="Correct.",
        metadata={"key": "val"},
        tokens_input=100,
        tokens_output=20,
        latency_ms=123.4,
        timestamp=_NOW,
    )
    defaults.update(overrides)
    return RolloutRecord(**defaults)


def _reflection(**overrides) -> ReflectionRecord:
    defaults = dict(
        reflection_id=_uid(),
        run_id="run-abc",
        round_num=1,
        triggered_by="mutation",
        target_candidate_id="cand-1",
        target_module=None,
        input_traces=["trace-1", "trace-2"],
        input_prompt="Improve this prompt.",
        output="Improved prompt text.",
        parsed_candidate_before="old prompt",
        parsed_candidate_after="new prompt",
        diff="--- old\n+++ new",
        tokens_input=200,
        tokens_output=50,
        latency_ms=456.7,
        timestamp=_NOW,
    )
    defaults.update(overrides)
    return ReflectionRecord(**defaults)


def _candidate(**overrides) -> CandidateRecord:
    defaults = dict(
        candidate_id=_uid(),
        run_id="run-abc",
        parent_ids=[],
        birth_round=0,
        birth_mechanism="seed",
        skill_category=None,
        prompt_by_module={"default": "You are a helpful assistant."},
        score_history=[(0, 0.5), (1, 0.7)],
        per_instance_scores={"ex-1": 1.0, "ex-2": 0.0},
        pareto_frontier_rounds=[1],
        death_round=None,
        death_reason=None,
        total_rollouts_consumed=10,
    )
    defaults.update(overrides)
    return CandidateRecord(**defaults)


def _round_summary(**overrides) -> RoundSummary:
    defaults = dict(
        round_num=3,
        run_id="run-abc",
        pool_size_before=5,
        pool_size_after=4,
        minibatch_example_ids=["ex-1", "ex-2", "ex-3"],
        score_stats={
            "mean": 0.6,
            "median": 0.65,
            "stdev": 0.1,
            "max": 0.9,
            "min": 0.3,
            "top3_mean": 0.85,
        },
        pareto_frontier_ids=["cand-1", "cand-3"],
        rollouts_this_round=15,
        rollouts_cumulative=60,
        tokens_task_this_round=3000,
        tokens_reflection_this_round=500,
        tokens_cumulative=14000,
        wall_clock_seconds=12.5,
        pruning_decisions={"pruned": ["cand-2"]},
        reflection_decisions={"mutated": ["cand-1"]},
    )
    defaults.update(overrides)
    return RoundSummary(**defaults)


def _run_summary(**overrides) -> RunSummary:
    defaults = dict(
        run_id=_uid(),
        start_time=_NOW,
        end_time=_NOW,
        duration_seconds=120.0,
        optimizer="iso",
        variant_config={"budget": 300},
        benchmark="hotpotqa",
        seed=42,
        git_sha="abc1234",
        model_task="Qwen3-4B",
        model_reflection="Qwen3-4B",
        budget_rollouts=300,
        rollouts_consumed_total=280,
        tokens_consumed_total=50000,
        final_score_val=0.78,
        final_score_test=0.75,
        final_candidate_id="cand-best",
        final_candidate_prompts={"default": "best prompt"},
        failure_mode=None,
        cost_estimate_usd=0.05,
    )
    defaults.update(overrides)
    return RunSummary(**defaults)


def _telemetry(**overrides) -> SystemTelemetryRecord:
    defaults = dict(
        run_id="run-abc",
        timestamp=_NOW,
        gpu_utilization_pct=75.0,
        gpu_memory_used_mb=8192.0,
        gpu_temp_c=62.0,
        gpu_power_w=250.0,
        kv_cache_util_task=45.0,
        kv_cache_util_reflection=30.0,
        task_server_queue_depth=2,
        reflection_server_queue_depth=0,
        task_throughput_tokens_per_sec=1200.0,
        reflection_throughput_tokens_per_sec=800.0,
        volume_used_pct=55.0,
        volume_free_gb=120.0,
    )
    defaults.update(overrides)
    return SystemTelemetryRecord(**defaults)


def _meta_optimizer(**overrides) -> MetaOptimizerRecord:
    defaults = dict(
        episode_id=_uid(),
        meta_run_id="meta-run-1",
        episode_num=0,
        proposed_config={"temp": 0.6, "top_p": 0.95},
        meta_llm_reasoning="Trying higher temperature.",
        episode_outcome={"final_score": 0.82},
        reward={"delta": 0.03},
        playbook_state="exploring",
        frontier_state={"best": 0.82},
    )
    defaults.update(overrides)
    return MetaOptimizerRecord(**defaults)


# ======================================================================
# Roundtrip tests — create -> model_dump_json -> model_validate_json
# ======================================================================


class TestRoundtrip:
    """Serialise to JSON and back; verify equality."""

    def test_rollout_roundtrip(self):
        original = _rollout()
        json_str = original.model_dump_json()
        restored = RolloutRecord.model_validate_json(json_str)
        assert original == restored

    def test_reflection_roundtrip(self):
        original = _reflection()
        json_str = original.model_dump_json()
        restored = ReflectionRecord.model_validate_json(json_str)
        assert original == restored

    def test_candidate_roundtrip(self):
        original = _candidate()
        json_str = original.model_dump_json()
        restored = CandidateRecord.model_validate_json(json_str)
        assert original == restored

    def test_round_summary_roundtrip(self):
        original = _round_summary()
        json_str = original.model_dump_json()
        restored = RoundSummary.model_validate_json(json_str)
        assert original == restored

    def test_run_summary_roundtrip(self):
        original = _run_summary()
        json_str = original.model_dump_json()
        restored = RunSummary.model_validate_json(json_str)
        assert original == restored

    def test_telemetry_roundtrip(self):
        original = _telemetry()
        json_str = original.model_dump_json()
        restored = SystemTelemetryRecord.model_validate_json(json_str)
        assert original == restored

    def test_meta_optimizer_roundtrip(self):
        original = _meta_optimizer()
        json_str = original.model_dump_json()
        restored = MetaOptimizerRecord.model_validate_json(json_str)
        assert original == restored


# ======================================================================
# Validation rejection tests
# ======================================================================


class TestValidationRejection:
    """Out-of-range or malformed values must raise ValidationError."""

    # -- RolloutRecord --

    def test_rollout_score_above_1(self):
        with pytest.raises(ValidationError):
            _rollout(score=1.01)

    def test_rollout_score_below_0(self):
        with pytest.raises(ValidationError):
            _rollout(score=-0.1)

    def test_rollout_negative_tokens_input(self):
        with pytest.raises(ValidationError):
            _rollout(tokens_input=-1)

    def test_rollout_negative_tokens_output(self):
        with pytest.raises(ValidationError):
            _rollout(tokens_output=-5)

    def test_rollout_negative_latency(self):
        with pytest.raises(ValidationError):
            _rollout(latency_ms=-10.0)

    def test_rollout_negative_round(self):
        with pytest.raises(ValidationError):
            _rollout(round_num=-1)

    def test_rollout_invalid_uuid(self):
        with pytest.raises(ValidationError):
            _rollout(rollout_id="not-a-uuid")

    # -- ReflectionRecord --

    def test_reflection_invalid_triggered_by(self):
        with pytest.raises(ValidationError):
            _reflection(triggered_by="invalid_trigger")

    def test_reflection_negative_tokens(self):
        with pytest.raises(ValidationError):
            _reflection(tokens_input=-1)

    def test_reflection_negative_latency(self):
        with pytest.raises(ValidationError):
            _reflection(latency_ms=-1.0)

    def test_reflection_invalid_uuid(self):
        with pytest.raises(ValidationError):
            _reflection(reflection_id="xxx")

    # -- CandidateRecord --

    def test_candidate_invalid_birth_mechanism(self):
        with pytest.raises(ValidationError):
            _candidate(birth_mechanism="random")

    def test_candidate_negative_birth_round(self):
        with pytest.raises(ValidationError):
            _candidate(birth_round=-1)

    def test_candidate_negative_rollouts(self):
        with pytest.raises(ValidationError):
            _candidate(total_rollouts_consumed=-1)

    def test_candidate_invalid_uuid(self):
        with pytest.raises(ValidationError):
            _candidate(candidate_id="bad")

    # -- RoundSummary --

    def test_round_summary_negative_round(self):
        with pytest.raises(ValidationError):
            _round_summary(round_num=-1)

    def test_round_summary_negative_rollouts(self):
        with pytest.raises(ValidationError):
            _round_summary(rollouts_this_round=-1)

    def test_round_summary_negative_tokens(self):
        with pytest.raises(ValidationError):
            _round_summary(tokens_cumulative=-1)

    def test_round_summary_negative_wall_clock(self):
        with pytest.raises(ValidationError):
            _round_summary(wall_clock_seconds=-0.1)

    # -- RunSummary --

    def test_run_summary_score_above_1(self):
        with pytest.raises(ValidationError):
            _run_summary(final_score_val=1.5)

    def test_run_summary_score_below_0(self):
        with pytest.raises(ValidationError):
            _run_summary(final_score_test=-0.01)

    def test_run_summary_negative_duration(self):
        with pytest.raises(ValidationError):
            _run_summary(duration_seconds=-1.0)

    def test_run_summary_negative_budget(self):
        with pytest.raises(ValidationError):
            _run_summary(budget_rollouts=-1)

    def test_run_summary_negative_cost(self):
        with pytest.raises(ValidationError):
            _run_summary(cost_estimate_usd=-0.5)

    def test_run_summary_invalid_uuid(self):
        with pytest.raises(ValidationError):
            _run_summary(run_id="nope")

    # -- SystemTelemetryRecord --

    def test_telemetry_gpu_util_above_100(self):
        with pytest.raises(ValidationError):
            _telemetry(gpu_utilization_pct=101.0)

    def test_telemetry_gpu_util_below_0(self):
        with pytest.raises(ValidationError):
            _telemetry(gpu_utilization_pct=-1.0)

    def test_telemetry_gpu_mem_negative(self):
        with pytest.raises(ValidationError):
            _telemetry(gpu_memory_used_mb=-1.0)

    def test_telemetry_gpu_power_negative(self):
        with pytest.raises(ValidationError):
            _telemetry(gpu_power_w=-10.0)

    def test_telemetry_kv_cache_above_100(self):
        with pytest.raises(ValidationError):
            _telemetry(kv_cache_util_task=101.0)

    def test_telemetry_kv_cache_below_0(self):
        with pytest.raises(ValidationError):
            _telemetry(kv_cache_util_reflection=-1.0)

    # -- MetaOptimizerRecord --

    def test_meta_optimizer_negative_episode(self):
        with pytest.raises(ValidationError):
            _meta_optimizer(episode_num=-1)

    def test_meta_optimizer_invalid_uuid(self):
        with pytest.raises(ValidationError):
            _meta_optimizer(episode_id="bad-uuid")


# ======================================================================
# JSON output stability — expected keys present in serialised dict
# ======================================================================


class TestJsonStability:
    """Verify all expected keys appear in model_dump() output."""

    def test_rollout_keys(self):
        d = _rollout().model_dump()
        expected = {
            "schema_version", "rollout_id", "run_id", "round_num",
            "candidate_id", "module_name", "example_id", "prompt",
            "response", "score", "feedback", "metadata",
            "tokens_input", "tokens_output", "latency_ms", "timestamp",
        }
        assert expected <= set(d.keys())

    def test_reflection_keys(self):
        d = _reflection().model_dump()
        expected = {
            "schema_version", "reflection_id", "run_id", "round_num",
            "triggered_by", "target_candidate_id", "target_module",
            "input_traces", "input_prompt", "output",
            "parsed_candidate_before", "parsed_candidate_after", "diff",
            "tokens_input", "tokens_output", "latency_ms", "timestamp",
        }
        assert expected <= set(d.keys())

    def test_candidate_keys(self):
        d = _candidate().model_dump()
        expected = {
            "schema_version", "candidate_id", "run_id", "parent_ids",
            "birth_round", "birth_mechanism", "skill_category",
            "prompt_by_module", "score_history", "per_instance_scores",
            "pareto_frontier_rounds", "death_round", "death_reason",
            "total_rollouts_consumed",
        }
        assert expected <= set(d.keys())

    def test_round_summary_keys(self):
        d = _round_summary().model_dump()
        expected = {
            "schema_version", "round_num", "run_id",
            "pool_size_before", "pool_size_after",
            "minibatch_example_ids", "score_stats",
            "pareto_frontier_ids", "rollouts_this_round",
            "rollouts_cumulative", "tokens_task_this_round",
            "tokens_reflection_this_round", "tokens_cumulative",
            "wall_clock_seconds", "pruning_decisions",
            "reflection_decisions",
        }
        assert expected <= set(d.keys())

    def test_run_summary_keys(self):
        d = _run_summary().model_dump()
        expected = {
            "schema_version", "run_id", "start_time", "end_time",
            "duration_seconds", "optimizer", "variant_config",
            "benchmark", "seed", "git_sha", "model_task",
            "model_reflection", "budget_rollouts",
            "rollouts_consumed_total", "tokens_consumed_total",
            "final_score_val", "final_score_test",
            "final_candidate_id", "final_candidate_prompts",
            "failure_mode", "cost_estimate_usd",
        }
        assert expected <= set(d.keys())

    def test_telemetry_keys(self):
        d = _telemetry().model_dump()
        expected = {
            "schema_version", "run_id", "timestamp",
            "gpu_utilization_pct", "gpu_memory_used_mb",
            "gpu_temp_c", "gpu_power_w",
            "kv_cache_util_task", "kv_cache_util_reflection",
            "task_server_queue_depth", "reflection_server_queue_depth",
            "task_throughput_tokens_per_sec",
            "reflection_throughput_tokens_per_sec",
            "volume_used_pct", "volume_free_gb",
        }
        assert expected <= set(d.keys())

    def test_meta_optimizer_keys(self):
        d = _meta_optimizer().model_dump()
        expected = {
            "schema_version", "episode_id", "meta_run_id",
            "episode_num", "proposed_config", "meta_llm_reasoning",
            "episode_outcome", "reward", "playbook_state",
            "frontier_state",
        }
        assert expected <= set(d.keys())


# ======================================================================
# Default value tests
# ======================================================================


class TestDefaults:
    """Verify schema_version defaults to '1.0' for all models."""

    def test_rollout_schema_version(self):
        r = _rollout()
        assert r.schema_version == "1.0"

    def test_reflection_schema_version(self):
        r = _reflection()
        assert r.schema_version == "1.0"

    def test_candidate_schema_version(self):
        c = _candidate()
        assert c.schema_version == "1.0"

    def test_round_summary_schema_version(self):
        rs = _round_summary()
        assert rs.schema_version == "1.0"

    def test_run_summary_schema_version(self):
        rs = _run_summary()
        assert rs.schema_version == "1.0"

    def test_telemetry_schema_version(self):
        t = _telemetry()
        assert t.schema_version == "1.0"

    def test_meta_optimizer_schema_version(self):
        m = _meta_optimizer()
        assert m.schema_version == "1.0"

    def test_rollout_metadata_default(self):
        """metadata defaults to empty dict via default_factory."""
        # Build without overriding metadata at all (omit from kwargs)
        r = RolloutRecord(
            rollout_id=_uid(),
            run_id="run-abc",
            round_num=0,
            candidate_id="cand-1",
            example_id="ex-42",
            prompt="What is 2+2?",
            response="4",
            score=0.85,
            feedback="Correct.",
            tokens_input=100,
            tokens_output=20,
            latency_ms=123.4,
            timestamp=_NOW,
        )
        assert isinstance(r.metadata, dict)
        assert r.metadata == {}

    def test_candidate_optional_death_fields(self):
        c = _candidate()
        assert c.death_round is None
        assert c.death_reason is None

    def test_telemetry_optional_fields_none(self):
        t = _telemetry(
            kv_cache_util_task=None,
            kv_cache_util_reflection=None,
            task_server_queue_depth=None,
            reflection_server_queue_depth=None,
            task_throughput_tokens_per_sec=None,
            reflection_throughput_tokens_per_sec=None,
            volume_used_pct=None,
            volume_free_gb=None,
        )
        assert t.kv_cache_util_task is None
        assert t.volume_free_gb is None

    def test_meta_optimizer_frontier_state_none(self):
        m = _meta_optimizer(frontier_state=None)
        assert m.frontier_state is None


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    """Boundary values and special cases."""

    def test_rollout_score_exactly_0(self):
        r = _rollout(score=0.0)
        assert r.score == 0.0

    def test_rollout_score_exactly_1(self):
        r = _rollout(score=1.0)
        assert r.score == 1.0

    def test_telemetry_gpu_util_exactly_0(self):
        t = _telemetry(gpu_utilization_pct=0.0)
        assert t.gpu_utilization_pct == 0.0

    def test_telemetry_gpu_util_exactly_100(self):
        t = _telemetry(gpu_utilization_pct=100.0)
        assert t.gpu_utilization_pct == 100.0

    def test_candidate_empty_score_history(self):
        c = _candidate(score_history=[])
        assert c.score_history == []

    def test_round_summary_zero_rollouts(self):
        rs = _round_summary(rollouts_this_round=0, rollouts_cumulative=0)
        assert rs.rollouts_this_round == 0

    def test_run_summary_zero_cost(self):
        rs = _run_summary(cost_estimate_usd=0.0)
        assert rs.cost_estimate_usd == 0.0
