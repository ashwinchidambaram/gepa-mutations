"""Tests for system monitor: GPU parsing, cost tracker, monitor lifecycle."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from iso_harness.experiment.jsonl_writer import JSONLWriter
from iso_harness.experiment.monitor import (
    GPU_RATES,
    CostTracker,
    SystemMonitor,
    parse_nvidia_smi,
)


class TestParseNvidiaSmi:
    def test_parses_valid_output(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "85, 32000, 72, 250.5\n"

        with patch("subprocess.run", return_value=mock_result):
            result = parse_nvidia_smi()

        assert result is not None
        assert result["gpu_utilization_pct"] == 85.0
        assert result["gpu_memory_used_mb"] == 32000.0
        assert result["gpu_temp_c"] == 72.0
        assert result["gpu_power_w"] == 250.5

    def test_returns_none_on_failure(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = parse_nvidia_smi()
        assert result is None

    def test_handles_na_power(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "50, 16000, 65, [N/A]\n"

        with patch("subprocess.run", return_value=mock_result):
            result = parse_nvidia_smi()

        assert result is not None
        assert result["gpu_power_w"] == 0.0


class TestCostTracker:
    def test_initial_state(self):
        ct = CostTracker("H100")
        assert ct.total_hours == 0.0
        assert ct.estimated_cost_usd == 0.0

    def test_accumulates_time(self):
        ct = CostTracker("H100")
        ct.add_time(3600)  # 1 hour
        assert ct.total_hours == 1.0
        assert ct.estimated_cost_usd == GPU_RATES["H100"]

    def test_cost_scales_with_time(self):
        ct = CostTracker("L40S")
        ct.add_time(7200)  # 2 hours
        assert abs(ct.estimated_cost_usd - 2 * GPU_RATES["L40S"]) < 0.01

    def test_unknown_gpu_zero_rate(self):
        ct = CostTracker("UnknownGPU")
        ct.add_time(3600)
        assert ct.estimated_cost_usd == 0.0

    def test_anomaly_detection(self):
        ct = CostTracker("H100")
        ct.add_time(1000)
        assert not ct.check_anomaly(600)  # 1000 < 2*600
        assert ct.check_anomaly(400)  # 1000 > 2*400

    def test_anomaly_zero_expected(self):
        ct = CostTracker("H100")
        ct.add_time(100)
        assert not ct.check_anomaly(0)


class TestSystemMonitor:
    def test_start_stop(self, tmp_path: Path):
        writer = JSONLWriter(tmp_path / "telemetry.jsonl")
        monitor = SystemMonitor(
            writer, run_id="test", gpu_interval=1, kv_interval=1, disk_interval=1
        )
        monitor.start()
        assert monitor._thread is not None
        assert monitor._thread.is_alive()
        monitor.stop()
        assert not monitor._thread.is_alive()

    def test_no_crash_without_gpu(self, tmp_path: Path):
        """Monitor should not crash if nvidia-smi is unavailable."""
        writer = JSONLWriter(tmp_path / "telemetry.jsonl")
        monitor = SystemMonitor(writer, run_id="test", gpu_interval=1)

        with patch("iso_harness.experiment.monitor.parse_nvidia_smi", return_value=None):
            monitor.start()
            time.sleep(2)
            monitor.stop()

        # Should not crash, may have 0 records
        records = writer.read_all()
        assert isinstance(records, list)

    def test_double_start_safe(self, tmp_path: Path):
        writer = JSONLWriter(tmp_path / "telemetry.jsonl")
        monitor = SystemMonitor(writer, run_id="test")
        monitor.start()
        monitor.start()  # Should be safe
        monitor.stop()
