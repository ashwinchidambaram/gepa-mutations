"""Tests for per-run report generation: summary.json, report.md, CSV."""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

from iso_harness.experiment.reporter import (
    append_to_results_csv,
    generate_run_report,
    write_report_md,
    write_summary_json,
)


def _sample_summary() -> dict:
    return {
        "run_id": "test-run-001",
        "optimizer": "ISO-Tide",
        "benchmark": "hotpotqa",
        "seed": 42,
        "final_score_val": 0.75,
        "final_score_test": 0.72,
        "rollouts_consumed_total": 6500,
        "tokens_consumed_total": 1200000,
        "duration_seconds": 3600.0,
        "cost_estimate_usd": 1.99,
        "git_sha": "abc123",
        "start_time": "2026-04-18T12:00:00Z",
        "model_task": "Qwen/Qwen3-8B",
        "model_reflection": "Qwen/Qwen3-32B-AWQ",
        "final_candidate_prompts": {
            "system": "You are a helpful QA assistant.",
        },
    }


class TestWriteSummaryJson:
    def test_creates_file(self, tmp_path: Path):
        run_dir = tmp_path / "run_001"
        path = write_summary_json(run_dir, _sample_summary())
        assert path.exists()
        assert path.name == "summary.json"

    def test_valid_json(self, tmp_path: Path):
        run_dir = tmp_path / "run_001"
        write_summary_json(run_dir, _sample_summary())
        data = json.loads((run_dir / "summary.json").read_text())
        assert data["optimizer"] == "ISO-Tide"
        assert data["final_score_test"] == 0.72

    def test_atomic_no_tmp_remains(self, tmp_path: Path):
        run_dir = tmp_path / "run_001"
        write_summary_json(run_dir, _sample_summary())
        assert not (run_dir / "summary.tmp").exists()

    def test_creates_parent_dirs(self, tmp_path: Path):
        run_dir = tmp_path / "deep" / "nested" / "run"
        write_summary_json(run_dir, _sample_summary())
        assert (run_dir / "summary.json").exists()


class TestWriteReportMd:
    def test_creates_file(self, tmp_path: Path):
        run_dir = tmp_path / "run_001"
        path = write_report_md(run_dir, _sample_summary())
        assert path.exists()
        assert path.name == "report.md"

    def test_contains_scores(self, tmp_path: Path):
        run_dir = tmp_path / "run_001"
        write_report_md(run_dir, _sample_summary())
        content = (run_dir / "report.md").read_text()
        assert "0.7500" in content  # val score
        assert "0.7200" in content  # test score

    def test_contains_optimizer_info(self, tmp_path: Path):
        run_dir = tmp_path / "run_001"
        write_report_md(run_dir, _sample_summary())
        content = (run_dir / "report.md").read_text()
        assert "ISO-Tide" in content
        assert "hotpotqa" in content
        assert "seed=42" in content

    def test_contains_prompts(self, tmp_path: Path):
        run_dir = tmp_path / "run_001"
        write_report_md(run_dir, _sample_summary())
        content = (run_dir / "report.md").read_text()
        assert "You are a helpful QA assistant." in content

    def test_no_prompts_fallback(self, tmp_path: Path):
        run_dir = tmp_path / "run_001"
        summary = _sample_summary()
        summary["final_candidate_prompts"] = {}
        write_report_md(run_dir, summary)
        content = (run_dir / "report.md").read_text()
        assert "No prompts recorded" in content


class TestAppendToResultsCsv:
    def test_creates_csv_with_header(self, tmp_path: Path):
        csv_path = tmp_path / "results.csv"
        append_to_results_csv(csv_path, _sample_summary())
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["optimizer"] == "ISO-Tide"
        assert rows[0]["benchmark"] == "hotpotqa"

    def test_appends_without_duplicate_header(self, tmp_path: Path):
        csv_path = tmp_path / "results.csv"
        append_to_results_csv(csv_path, _sample_summary())
        s2 = _sample_summary()
        s2["run_id"] = "test-run-002"
        s2["seed"] = 43
        append_to_results_csv(csv_path, s2)

        with open(csv_path) as f:
            lines = f.readlines()
        # 1 header + 2 data rows
        assert len(lines) == 3

    def test_thread_safe(self, tmp_path: Path):
        csv_path = tmp_path / "results.csv"
        errors = []

        def write_rows(start: int):
            try:
                for i in range(10):
                    s = _sample_summary()
                    s["run_id"] = f"run-{start}-{i}"
                    append_to_results_csv(csv_path, s)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_rows, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 40


class TestGenerateRunReport:
    def test_generates_all_outputs(self, tmp_path: Path):
        run_dir = tmp_path / "run_001"
        csv_path = tmp_path / "experiment_results.csv"
        generate_run_report(run_dir, _sample_summary(), csv_path=csv_path)

        assert (run_dir / "summary.json").exists()
        assert (run_dir / "report.md").exists()
        assert csv_path.exists()

    def test_skip_csv_when_none(self, tmp_path: Path):
        run_dir = tmp_path / "run_001"
        generate_run_report(run_dir, _sample_summary(), csv_path=None)

        assert (run_dir / "summary.json").exists()
        assert (run_dir / "report.md").exists()
