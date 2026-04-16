"""Test MetricsCollector's trajectory point recording (backward compat + new fields)."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gepa_mutations.metrics.collector import MetricsCollector


def test_record_val_score_backward_compat():
    """Existing record_val_score API still works."""
    c = MetricsCollector()
    c.record_val_score(iteration=0, score=0.5, prompt_length=100)
    c.record_val_score(iteration=1, score=0.7, prompt_length=120)
    assert len(c.val_score_trajectory) == 2
    # Existing format: (iteration, score) tuples
    assert c.val_score_trajectory[0] == (0, 0.5)
    assert c.val_score_trajectory[1] == (1, 0.7)


def test_record_trajectory_point_new():
    """New record_trajectory_point captures rollouts + reflection + holdout."""
    c = MetricsCollector()
    c.record_rollouts(n=100)
    c.record_reflection_call()
    c.record_trajectory_point(
        iteration=1,
        holdout_score=0.65,
        best_so_far=0.70,
        prompt_length=150,
    )
    # Should populate the new trajectory list with dict entries
    assert hasattr(c, "holdout_trajectory"), "Need a holdout_trajectory field"
    assert len(c.holdout_trajectory) == 1
    point = c.holdout_trajectory[0]
    assert point["iteration"] == 1
    assert point["cumulative_rollouts"] == 100
    assert point["cumulative_reflection_calls"] == 1
    assert point["holdout_score"] == 0.65
    assert point["best_so_far"] == 0.70
    assert point["prompt_length"] == 150


def test_trajectory_point_in_finalize():
    """holdout_trajectory should be serialized by finalize()."""
    c = MetricsCollector()
    c.record_rollouts(n=50)
    c.record_trajectory_point(
        iteration=0, holdout_score=0.4, best_so_far=0.4, prompt_length=100,
    )
    out = c.finalize(
        test_score=0.5,
        best_prompt={"system_prompt": "x"},
        test_example_scores=[],
        test_example_ids=[],
        model="test-model",
        model_tag="test",
        benchmark="hotpotqa",
        seed=42,
        method="test_method",
        seed_prompt="x",
    )
    assert "holdout_trajectory" in out
    assert len(out["holdout_trajectory"]) == 1
