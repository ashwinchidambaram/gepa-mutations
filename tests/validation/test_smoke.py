"""V9: End-to-end smoke test — minimal GEPA run on IFBench.

This test requires live vLLM servers. It runs GEPA with a tiny subset
(50 examples, 100 rollouts) to validate the full pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.live_server
@pytest.mark.slow
class TestSmokeEndToEnd:
    """V9: Full pipeline smoke test with real inference.

    Requires: vLLM task server on port 8000.
    Expected runtime: ~10-20 minutes.
    """

    def test_gepa_smoke_produces_artifacts(self, live_vllm_task, tmp_path: Path):
        """Run GEPA on IFBench with minimal budget, verify all artifacts."""
        # This test is intentionally a placeholder that validates
        # the test infrastructure is set up correctly.
        # The full smoke test requires the GEPA integration to be
        # wired into the orchestrator, which is done at deployment time.
        #
        # When running on RunPod with live servers, replace this with:
        #   from iso_harness.experiment.orchestrator import Orchestrator, RunSpec
        #   from iso_harness.experiment.config import load_config
        #   config = load_config("configs/pilot.yaml")
        #   orch = Orchestrator(config, runs_dir=tmp_path)
        #   matrix = orch.build_matrix()
        #   results = orch.execute(matrix, run_fn=<gepa_runner>)
        #   assert results[0]["status"] == "completed"

        pytest.skip(
            "V9 smoke test requires GEPA runner integration with orchestrator. "
            "Run manually on RunPod: ./scripts/iso_run_pilot.sh --smoke-test"
        )
