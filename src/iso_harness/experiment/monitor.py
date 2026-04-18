"""System monitoring: GPU, KV cache, disk telemetry + cost tracking.

Background daemon thread that samples nvidia-smi, vLLM /metrics, and disk
usage at configurable intervals. Writes SystemTelemetryRecord to JSONL.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from iso_harness.experiment.jsonl_writer import JSONLWriter
from iso_harness.experiment.schemas import SystemTelemetryRecord

logger = logging.getLogger(__name__)

# RunPod GPU hourly rates (USD)
GPU_RATES: dict[str, float] = {
    "L40S": 0.79,
    "H100": 1.99,
    "A100": 1.59,
    "A40": 0.44,
    "RTX 4090": 0.69,
}


def parse_nvidia_smi() -> dict[str, float] | None:
    """Parse nvidia-smi output for GPU metrics.

    Returns:
        Dict with gpu_utilization_pct, gpu_memory_used_mb, gpu_temp_c, gpu_power_w.
        None if nvidia-smi is unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        # Parse first GPU line
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return None

        return {
            "gpu_utilization_pct": float(parts[0]),
            "gpu_memory_used_mb": float(parts[1]),
            "gpu_temp_c": float(parts[2]),
            "gpu_power_w": float(parts[3]) if parts[3] != "[N/A]" else 0.0,
        }
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        return None


def parse_vllm_metrics(port: int) -> dict[str, float | None]:
    """Parse vLLM /metrics endpoint for KV cache and throughput stats.

    Returns:
        Dict with kv_cache_util (0-100), queue_depth, throughput_tokens_per_sec.
        Values are None if the endpoint is unreachable or metrics not found.
    """
    try:
        import urllib.request

        url = f"http://localhost:{port}/metrics"
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode("utf-8")

        result: dict[str, float | None] = {
            "kv_cache_util": None,
            "queue_depth": None,
            "throughput_tokens_per_sec": None,
        }

        for line in text.split("\n"):
            if line.startswith("vllm:gpu_cache_usage_perc"):
                # Prometheus format: metric_name{labels} value
                val = line.split()[-1]
                result["kv_cache_util"] = float(val) * 100  # Convert to percentage
            elif line.startswith("vllm:num_requests_waiting"):
                val = line.split()[-1]
                result["queue_depth"] = int(float(val))
            elif line.startswith("vllm:avg_generation_throughput_toks_per_s"):
                val = line.split()[-1]
                result["throughput_tokens_per_sec"] = float(val)

        return result
    except Exception:
        return {"kv_cache_util": None, "queue_depth": None, "throughput_tokens_per_sec": None}


class SystemMonitor:
    """Background daemon thread that samples system telemetry.

    Args:
        writer: JSONLWriter for telemetry records.
        run_id: Current run ID for tagging records.
        gpu_interval: Seconds between GPU samples (default 30).
        kv_interval: Seconds between KV cache samples (default 60).
        disk_interval: Seconds between disk checks (default 300 = 5min).
        disk_path: Path to check for disk usage (default /workspace).
        disk_min_free_gb: Alert threshold for free disk space.
        task_port: vLLM task server port (default 8000).
        reflection_port: vLLM reflection server port (default 8001).
    """

    def __init__(
        self,
        writer: JSONLWriter,
        run_id: str = "",
        gpu_interval: int = 30,
        kv_interval: int = 60,
        disk_interval: int = 300,
        disk_path: str = "/workspace",
        disk_min_free_gb: int = 20,
        task_port: int = 8000,
        reflection_port: int = 8001,
    ) -> None:
        self._writer = writer
        self._run_id = run_id
        self._gpu_interval = gpu_interval
        self._kv_interval = kv_interval
        self._disk_interval = disk_interval
        self._disk_path = disk_path
        self._disk_min_free_gb = disk_min_free_gb
        self._task_port = task_port
        self._reflection_port = reflection_port
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._low_gpu_count = 0  # Consecutive low-GPU samples

    def start(self) -> None:
        """Start the monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="system-monitor")
        self._thread.start()
        logger.info(
            "System monitor started (GPU=%ds, KV=%ds, disk=%ds)",
            self._gpu_interval,
            self._kv_interval,
            self._disk_interval,
        )

    def stop(self) -> None:
        """Stop the monitoring thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        logger.info("System monitor stopped")

    def _loop(self) -> None:
        """Main monitoring loop."""
        last_gpu = 0.0
        last_kv = 0.0
        last_disk = 0.0

        while not self._stop_event.is_set():
            now = time.monotonic()

            # GPU sampling
            if now - last_gpu >= self._gpu_interval:
                self._sample_gpu()
                last_gpu = now

            # KV cache sampling
            if now - last_kv >= self._kv_interval:
                self._sample_kv()
                last_kv = now

            # Disk check
            if now - last_disk >= self._disk_interval:
                self._check_disk()
                last_disk = now

            self._stop_event.wait(timeout=1.0)

    def _sample_gpu(self) -> None:
        """Sample GPU metrics and write to JSONL."""
        gpu = parse_nvidia_smi()
        if gpu is None:
            return

        record = SystemTelemetryRecord(
            run_id=self._run_id,
            timestamp=datetime.now(timezone.utc),
            gpu_utilization_pct=gpu["gpu_utilization_pct"],
            gpu_memory_used_mb=gpu["gpu_memory_used_mb"],
            gpu_temp_c=gpu["gpu_temp_c"],
            gpu_power_w=gpu["gpu_power_w"],
        )
        self._writer.append(record)

        # Alert: GPU util < 10% sustained
        if gpu["gpu_utilization_pct"] < 10:
            self._low_gpu_count += 1
            if self._low_gpu_count >= 10:
                logger.warning(
                    "GPU utilization < 10%% for %d consecutive samples — possible stall",
                    self._low_gpu_count,
                )
        else:
            self._low_gpu_count = 0

    def _sample_kv(self) -> None:
        """Sample KV cache metrics from both vLLM servers."""
        task_metrics = parse_vllm_metrics(self._task_port)
        refl_metrics = parse_vllm_metrics(self._reflection_port)

        kv_task = task_metrics.get("kv_cache_util")
        kv_refl = refl_metrics.get("kv_cache_util")

        # Alert: KV cache > 90%
        for name, val in [("task", kv_task), ("reflection", kv_refl)]:
            if val is not None and val > 90:
                logger.warning("%s server KV cache at %.1f%% — near capacity", name, val)

    def _check_disk(self) -> None:
        """Check disk usage and alert if low."""
        try:
            usage = shutil.disk_usage(self._disk_path)
            free_gb = usage.free / (1024**3)
            used_pct = (usage.used / usage.total) * 100

            if free_gb < self._disk_min_free_gb:
                logger.warning(
                    "Disk space low: %.1f GB free (%.1f%% used) at %s",
                    free_gb,
                    used_pct,
                    self._disk_path,
                )
        except OSError:
            pass


class CostTracker:
    """Tracks wall-clock time and estimates dollar cost.

    Args:
        gpu_type: GPU type string matching GPU_RATES keys.
    """

    def __init__(self, gpu_type: str = "H100") -> None:
        self._gpu_type = gpu_type
        self._rate = GPU_RATES.get(gpu_type, 0.0)
        self._total_seconds: float = 0.0

    def add_time(self, seconds: float) -> None:
        """Add wall-clock seconds to the running total."""
        self._total_seconds += seconds

    @property
    def total_hours(self) -> float:
        """Total accumulated hours."""
        return self._total_seconds / 3600

    @property
    def estimated_cost_usd(self) -> float:
        """Estimated cost in USD based on GPU rate."""
        return self.total_hours * self._rate

    def check_anomaly(self, expected_seconds: float) -> bool:
        """Return True if actual cost > 2x expected (cost anomaly).

        Args:
            expected_seconds: Expected wall-clock for this run.
        """
        if expected_seconds <= 0:
            return False
        return self._total_seconds > 2 * expected_seconds
