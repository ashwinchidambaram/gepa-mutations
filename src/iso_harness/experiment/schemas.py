"""7-layer Pydantic schemas for ISO experiment JSONL logging.

Layer 1: RolloutRecord       — per-rollout evaluation data
Layer 2: ReflectionRecord    — per-reflection / mutation data
Layer 3: CandidateRecord     — candidate lifecycle
Layer 4: RoundSummary        — per-round aggregate statistics
Layer 5: RunSummary          — experiment-level summary
Layer 6: SystemTelemetryRecord — GPU / KV-cache / disk telemetry
Layer 7: MetaOptimizerRecord — Track 2 meta-optimizer episodes

Field names align with the Parquet schemas in
``src/gepa_mutations/logging/parquet_logger.py`` where they overlap
(e.g. ``run_id``, ``score``, ``tokens_input`` / ``prompt_tokens``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "RolloutRecord",
    "ReflectionRecord",
    "CandidateRecord",
    "RoundSummary",
    "RunSummary",
    "SystemTelemetryRecord",
    "MetaOptimizerRecord",
]

_UUID4_PATTERN = r"^[0-9a-f-]{36}$"


# ---------------------------------------------------------------------------
# Layer 1 — per-rollout data
# ---------------------------------------------------------------------------

class RolloutRecord(BaseModel):
    """One evaluation of a candidate prompt on a single example."""

    model_config = ConfigDict(strict=False)

    schema_version: str = "1.0"

    rollout_id: str = Field(..., pattern=_UUID4_PATTERN)
    run_id: str
    round_num: int = Field(..., ge=0)
    candidate_id: str
    module_name: str | None = None
    example_id: str
    prompt: str
    response: str
    score: float = Field(..., ge=0.0, le=1.0)
    feedback: str
    metadata: dict = Field(default_factory=dict)
    tokens_input: int = Field(..., ge=0)
    tokens_output: int = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0)
    timestamp: datetime


# ---------------------------------------------------------------------------
# Layer 2 — per-reflection data
# ---------------------------------------------------------------------------

class ReflectionRecord(BaseModel):
    """One reflection / mutation call and its outcome."""

    model_config = ConfigDict(strict=False)

    schema_version: str = "1.0"

    reflection_id: str = Field(..., pattern=_UUID4_PATTERN)
    run_id: str
    round_num: int = Field(..., ge=0)
    triggered_by: Literal[
        "skill_discovery",
        "mutation",
        "cross_mutation",
        "reflection_mutation",
        "seed",
    ]
    target_candidate_id: str
    target_module: str | None = None
    input_traces: list[str] = Field(default_factory=list)
    input_prompt: str
    output: str
    parsed_candidate_before: str
    parsed_candidate_after: str
    diff: str
    tokens_input: int = Field(..., ge=0)
    tokens_output: int = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0)
    timestamp: datetime


# ---------------------------------------------------------------------------
# Layer 3 — candidate lifecycle
# ---------------------------------------------------------------------------

class CandidateRecord(BaseModel):
    """Full lifecycle record for a single candidate prompt."""

    model_config = ConfigDict(strict=False)

    schema_version: str = "1.0"

    candidate_id: str = Field(..., pattern=_UUID4_PATTERN)
    run_id: str
    parent_ids: list[str] = Field(default_factory=list)
    birth_round: int = Field(..., ge=0)
    birth_mechanism: Literal[
        "skill_discovery",
        "reflection_mutation",
        "cross_mutation",
        "seed",
    ]
    skill_category: str | None = None
    prompt_by_module: dict = Field(default_factory=dict)
    score_history: list[tuple[int, float]] = Field(default_factory=list)
    per_instance_scores: dict = Field(default_factory=dict)
    pareto_frontier_rounds: list[int] = Field(default_factory=list)
    death_round: int | None = None
    death_reason: str | None = None
    total_rollouts_consumed: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Layer 4 — per-round aggregates
# ---------------------------------------------------------------------------

class RoundSummary(BaseModel):
    """Aggregate statistics for one optimisation round."""

    model_config = ConfigDict(strict=False)

    schema_version: str = "1.0"

    round_num: int = Field(..., ge=0)
    run_id: str
    pool_size_before: int = 0
    pool_size_after: int = 0
    minibatch_example_ids: list[str] = Field(default_factory=list)
    score_stats: dict = Field(default_factory=dict)
    pareto_frontier_ids: list[str] = Field(default_factory=list)
    rollouts_this_round: int = Field(default=0, ge=0)
    rollouts_cumulative: int = Field(default=0, ge=0)
    tokens_task_this_round: int = Field(default=0, ge=0)
    tokens_reflection_this_round: int = Field(default=0, ge=0)
    tokens_cumulative: int = Field(default=0, ge=0)
    wall_clock_seconds: float = Field(default=0.0, ge=0)
    pruning_decisions: dict = Field(default_factory=dict)
    reflection_decisions: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Layer 5 — experiment-level summary
# ---------------------------------------------------------------------------

class RunSummary(BaseModel):
    """End-of-run summary for one experiment."""

    model_config = ConfigDict(strict=False)

    schema_version: str = "1.0"

    run_id: str = Field(..., pattern=_UUID4_PATTERN)
    start_time: datetime
    end_time: datetime
    duration_seconds: float = Field(..., ge=0)
    optimizer: str
    variant_config: dict = Field(default_factory=dict)
    benchmark: str
    seed: int
    git_sha: str
    model_task: str
    model_reflection: str
    budget_rollouts: int = Field(..., ge=0)
    rollouts_consumed_total: int = Field(default=0, ge=0)
    tokens_consumed_total: int = Field(default=0, ge=0)
    final_score_val: float = Field(..., ge=0.0, le=1.0)
    final_score_test: float = Field(..., ge=0.0, le=1.0)
    final_candidate_id: str
    final_candidate_prompts: dict = Field(default_factory=dict)
    failure_mode: str | None = None
    cost_estimate_usd: float = Field(default=0.0, ge=0)


# ---------------------------------------------------------------------------
# Layer 6 — GPU / KV-cache / disk telemetry
# ---------------------------------------------------------------------------

class SystemTelemetryRecord(BaseModel):
    """Periodic system health sample."""

    model_config = ConfigDict(strict=False)

    schema_version: str = "1.0"

    run_id: str
    timestamp: datetime
    gpu_utilization_pct: float = Field(..., ge=0, le=100)
    gpu_memory_used_mb: float = Field(..., ge=0)
    gpu_temp_c: float
    gpu_power_w: float = Field(..., ge=0)
    kv_cache_util_task: float | None = Field(default=None, ge=0, le=100)
    kv_cache_util_reflection: float | None = Field(default=None, ge=0, le=100)
    task_server_queue_depth: int | None = None
    reflection_server_queue_depth: int | None = None
    task_throughput_tokens_per_sec: float | None = None
    reflection_throughput_tokens_per_sec: float | None = None
    volume_used_pct: float | None = None
    volume_free_gb: float | None = None


# ---------------------------------------------------------------------------
# Layer 7 — Track 2 meta-optimizer
# ---------------------------------------------------------------------------

class MetaOptimizerRecord(BaseModel):
    """One episode of the meta-optimizer outer loop."""

    model_config = ConfigDict(strict=False)

    schema_version: str = "1.0"

    episode_id: str = Field(..., pattern=_UUID4_PATTERN)
    meta_run_id: str
    episode_num: int = Field(..., ge=0)
    proposed_config: dict = Field(default_factory=dict)
    meta_llm_reasoning: str
    episode_outcome: dict = Field(default_factory=dict)
    reward: dict = Field(default_factory=dict)
    playbook_state: str
    frontier_state: dict | None = None
