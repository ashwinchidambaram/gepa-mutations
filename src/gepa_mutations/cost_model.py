"""Parameter-weighted and token-weighted cost models for fair cross-method comparison.

Primary metric (parameter-weighted):
    pw_cost = rollouts + Σ_reflections (param_count_b / task_param_b)
    Each reflection call weighted proportional to model size.
    A Qwen3-32B reflection call counts as 4 Qwen3-8B rollout calls.

Secondary metric (token-weighted, sensitivity analysis):
    tw_cost = Σ_rollouts (prompt_tokens + completion_tokens) × task_param_b
            + Σ_reflections (input_tokens + output_tokens) × param_count_b
    True FLOPs proxy: accounts for both model size AND token count.

The reflection_model parameter refers to the LM responsible for generating
or modifying candidate prompts. In GEPA this is the reflector; in MIPROv2
it is the proposer; in ISO it handles discovery, mutation, and
cross-pollination. The role is structurally analogous but mechanically
distinct across methods — this is noted as a methodology consideration
in the paper.
"""

from __future__ import annotations

from pathlib import Path


def parameter_weighted_cost(
    rollouts: int,
    reflection_calls: int,
    task_param_b: float,
    reflection_param_b: float,
) -> float:
    """Compute parameter-weighted total cost.

    Returns cost in task-model-equivalent calls.
    """
    weight = reflection_param_b / task_param_b
    return rollouts + reflection_calls * weight


def token_weighted_cost(
    rollout_tokens: int,
    reflection_tokens: int,
    task_param_b: float,
    reflection_param_b: float,
) -> float:
    """Compute token-weighted total cost (FLOPs proxy).

    Returns cost in token·param units.
    """
    return rollout_tokens * task_param_b + reflection_tokens * reflection_param_b


def parameter_weighted_cost_from_parquet(
    run_dir: Path, task_param_b: float = 8.0
) -> float:
    """Compute parameter-weighted cost from a run's Parquet logs."""
    import pyarrow.parquet as pq

    rollouts_path = run_dir / "rollouts.parquet"
    reflections_path = run_dir / "reflections.parquet"

    n_rollouts = pq.read_table(rollouts_path).num_rows if rollouts_path.exists() else 0

    if reflections_path.exists():
        ref_table = pq.read_table(reflections_path)
        if "param_count_b" in ref_table.column_names:
            params = ref_table.column("param_count_b").to_pylist()
            return n_rollouts + sum(p / task_param_b for p in params if p)
        return float(n_rollouts + ref_table.num_rows)

    return float(n_rollouts)


def token_weighted_cost_from_parquet(
    run_dir: Path, task_param_b: float = 8.0
) -> float:
    """Compute token-weighted cost from a run's Parquet logs."""
    import pyarrow.parquet as pq

    total = 0.0

    rollouts_path = run_dir / "rollouts.parquet"
    if rollouts_path.exists():
        r_table = pq.read_table(rollouts_path)
        prompt_tokens = r_table.column("prompt_tokens").to_pylist()
        completion_tokens = r_table.column("completion_tokens").to_pylist()
        for pt, ct in zip(prompt_tokens, completion_tokens):
            total += (pt + ct) * task_param_b

    reflections_path = run_dir / "reflections.parquet"
    if reflections_path.exists():
        ref_table = pq.read_table(reflections_path)
        input_tokens = ref_table.column("input_tokens").to_pylist()
        output_tokens = ref_table.column("output_tokens").to_pylist()
        param_counts = ref_table.column("param_count_b").to_pylist()
        for it, ot, pb in zip(input_tokens, output_tokens, param_counts):
            total += (it + ot) * (pb if pb else task_param_b)

    return total
