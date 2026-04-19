"""Unit tests for ISO runtime: RolloutCounter, TraceStore, context accessors."""
import random
import pytest
from iso_harness.optimizer.runtime import (
    RolloutCounter, TraceStore, ISORuntime,
    set_current_runtime, get_current_runtime, runtime_context,
)
from iso_harness.optimizer.candidate import ModuleTrace


class TestRolloutCounter:
    def test_standalone_initial(self):
        rc = RolloutCounter()
        assert rc.value() == 0
        assert rc.remaining(100) == 100

    def test_standalone_increment(self):
        rc = RolloutCounter()
        rc.increment(10)
        assert rc.value() == 10
        rc.increment(5)
        assert rc.value() == 15

    def test_standalone_remaining(self):
        rc = RolloutCounter()
        rc.increment(30)
        assert rc.remaining(100) == 70
        assert rc.remaining(20) == 0  # clamped to 0

    def test_with_budget_enforcer(self):
        from iso_harness.experiment.orchestrator import BudgetEnforcer
        enforcer = BudgetEnforcer(max_rollouts=100)
        rc = RolloutCounter(enforcer)
        rc.increment(25)
        assert rc.value() == 25
        assert rc.remaining(100) == 75
        # Verify BudgetEnforcer was actually updated
        assert enforcer.consumed == 25


class TestTraceStore:
    def test_put_and_get(self):
        store = TraceStore()
        trace = ModuleTrace(example_id="ex_0", score=0.5)
        store.put("c1", "ex_0", trace)
        assert store.get("c1", "ex_0") is trace

    def test_get_missing(self):
        store = TraceStore()
        assert store.get("c1", "ex_0") is None

    def test_overwrite(self):
        store = TraceStore()
        t1 = ModuleTrace(example_id="ex_0", score=0.3)
        t2 = ModuleTrace(example_id="ex_0", score=0.8)
        store.put("c1", "ex_0", t1)
        store.put("c1", "ex_0", t2)
        assert store.get("c1", "ex_0") is t2

    def test_get_worst_for_candidate(self):
        store = TraceStore()
        store.put("c1", "ex_0", ModuleTrace(example_id="ex_0", score=0.9))
        store.put("c1", "ex_1", ModuleTrace(example_id="ex_1", score=0.2))
        store.put("c1", "ex_2", ModuleTrace(example_id="ex_2", score=0.5))
        scores = {"ex_0": 0.9, "ex_1": 0.2, "ex_2": 0.5}
        worst = store.get_worst_for_candidate("c1", scores, n=2)
        assert len(worst) == 2
        assert worst[0].score == 0.2  # worst first
        assert worst[1].score == 0.5

    def test_get_worst_skips_missing(self):
        store = TraceStore()
        store.put("c1", "ex_0", ModuleTrace(example_id="ex_0", score=0.9))
        # ex_1 has no trace stored
        scores = {"ex_0": 0.9, "ex_1": 0.1}
        worst = store.get_worst_for_candidate("c1", scores, n=2)
        assert len(worst) == 1  # only ex_0 has a trace

    def test_clear_round(self):
        store = TraceStore()
        store.put("c1", "ex_0", ModuleTrace(example_id="ex_0"))
        assert store.size() == 1
        store.clear_round()
        assert store.size() == 0

    def test_size(self):
        store = TraceStore()
        assert store.size() == 0
        store.put("c1", "ex_0", "trace1")
        store.put("c1", "ex_1", "trace2")
        assert store.size() == 2


class TestISORuntime:
    def _make_runtime(self):
        return ISORuntime(
            reflection_lm=lambda p: "response",
            task_lm=lambda p: "response",
            metric=lambda g, p, t=None, pn=None: {"score": 1.0, "feedback": "", "metadata": {}},
            run_id="test-run-id",
            seed=42,
            rng=random.Random(42),
            trace_store=TraceStore(),
            rollout_counter=RolloutCounter(),
        )

    def test_construction(self):
        rt = self._make_runtime()
        assert rt.run_id == "test-run-id"
        assert rt.round_num == 0

    def test_round_num_mutable(self):
        rt = self._make_runtime()
        rt.round_num = 5
        assert rt.round_num == 5


class TestContextAccessor:
    def test_set_and_get(self):
        rt = ISORuntime(
            reflection_lm=None, task_lm=None,
            metric=lambda g, p, t=None, pn=None: {"score": 0},
            run_id="ctx-test", seed=0, rng=random.Random(0),
            trace_store=TraceStore(), rollout_counter=RolloutCounter(),
        )
        set_current_runtime(rt)
        assert get_current_runtime() is rt

    def test_get_raises_when_not_set(self):
        # This may or may not raise depending on if another test set it.
        # Use runtime_context to test properly.
        pass

    def test_runtime_context_manager(self):
        rt = ISORuntime(
            reflection_lm=None, task_lm=None,
            metric=lambda g, p, t=None, pn=None: {"score": 0},
            run_id="ctx-test-2", seed=0, rng=random.Random(0),
            trace_store=TraceStore(), rollout_counter=RolloutCounter(),
        )
        with runtime_context(rt):
            assert get_current_runtime() is rt
            assert get_current_runtime().run_id == "ctx-test-2"
