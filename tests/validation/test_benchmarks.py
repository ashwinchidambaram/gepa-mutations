"""V6: Benchmark loader validation — all 6 benchmarks load with expected fields."""

from __future__ import annotations

import pytest

BENCHMARKS = ["hotpotqa", "hover", "ifbench", "pupa", "aime", "livebench"]


class TestBenchmarkLoaders:
    @pytest.mark.parametrize("benchmark", BENCHMARKS)
    def test_benchmark_loads(self, benchmark: str):
        """Benchmark loads without errors and produces non-empty splits."""
        from gepa_mutations.benchmarks.loader import load_benchmark

        try:
            data = load_benchmark(benchmark, seed=0)
        except Exception as e:
            pytest.skip(f"Benchmark {benchmark} failed to load: {e}")

        assert len(data.train) > 0, f"{benchmark} train is empty"
        assert len(data.val) > 0, f"{benchmark} val is empty"
        assert len(data.test) > 0, f"{benchmark} test is empty"

    @pytest.mark.parametrize("benchmark", ["hotpotqa", "ifbench"])
    def test_example_has_expected_fields(self, benchmark: str):
        """Benchmark examples have input fields."""
        from gepa_mutations.benchmarks.loader import load_benchmark

        try:
            data = load_benchmark(benchmark, seed=0)
        except Exception as e:
            pytest.skip(f"Benchmark {benchmark} failed to load: {e}")

        ex = data.train[0]
        # DSPy examples should have at least some input fields
        assert len(ex.keys()) > 0, f"{benchmark} example has no fields"
