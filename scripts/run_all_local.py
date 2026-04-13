#!/usr/bin/env python3
"""Parallel experiment orchestrator for local Ray cluster inference.

Runs all 120 experiments (6 benchmarks x 4 methods x 5 seeds) using
multiprocessing workers that each call the appropriate experiment runner.
vLLM batches concurrent requests automatically, so N workers = ~N× throughput.

Usage:
    python scripts/run_all_local.py                        # 6 workers, all methods
    python scripts/run_all_local.py --workers 4            # 4 workers
    python scripts/run_all_local.py --method gepa          # baseline only
    python scripts/run_all_local.py --benchmark hotpotqa   # one benchmark only
    python scripts/run_all_local.py --dry-run              # preview run order
    python scripts/run_all_local.py --smoke-test           # 5-example smoke per combo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re as _re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


# Propagate GEPA_BASE_URL → API_BASE_URL so method runners' Settings() picks it up.
# Method runners call Settings() directly (not via CLI), and pydantic-settings reads
# API_BASE_URL from env (field name uppercased, env_prefix=""). GEPA_BASE_URL is the
# canonical env var used by this orchestrator; propagate it here so subprocesses inherit it.
if os.environ.get("GEPA_BASE_URL") and not os.environ.get("API_BASE_URL"):
    os.environ["API_BASE_URL"] = os.environ["GEPA_BASE_URL"]

def _env_model_tag() -> str:
    """Derive a stable run-directory tag from GEPA_MODEL environment variable.

    Checks model family first so size suffixes don't collide across families
    (e.g. Gemma "4b" vs Qwen3 "4b" → different tags).
    """
    raw = os.environ.get("GEPA_MODEL", "").lower()
    # Strip quantization suffix (e.g. "-4bit", "-8bit") so it doesn't collide with
    # model-size tokens — e.g. "qwen3-0.6b-4bit" contains "4b" via the "-4bit" suffix.
    m = _re.sub(r"-\d+bit$", "", raw)

    # Qwen3 family
    if "qwen" in m:
        if "27b" in m:  return "qwen3-27b-awq"
        if "32b" in m:  return "qwen3-32b"
        if "14b" in m:  return "qwen3-14b"
        if "8b"  in m:  return "qwen3-8b"
        if "4b"  in m:  return "qwen3-4b"
        if "1.7b" in m: return "qwen3-1.7b"
        if "0.6b" in m: return "qwen3-0.6b"

    # Gemma 3 family
    if "gemma" in m:
        if "27b" in m:  return "gemma3-27b"
        if "12b" in m:  return "gemma3-12b"
        if "4b"  in m:  return "gemma3-4b"
        if "1b"  in m:  return "gemma3-1b"

    # Llama family (3.x)
    if "llama" in m:
        if "70b" in m:  return "llama3-70b"
        if "8b"  in m:  return "llama3-8b"
        if "3b"  in m:  return "llama3-3b"
        if "1b"  in m:  return "llama3-1b"

    return ""  # empty = no model prefix (backward compatible)

_MODEL_TAG = _env_model_tag()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BENCHMARKS = ["hotpotqa", "hover", "pupa", "ifbench", "livebench"]
METHODS = [
    "gepa", "best_of_k_K3", "contrastive_reflection", "failure_stratified_k_K3",
    "synaptic_pruning", "tournament", "slime_mold", "ant_colony",
    "active_minibatch", "contrastive_synthesis", "ecological_succession", "modular",
]
SEEDS = [42, 123, 456, 789, 1024]

# Fallback method order when no measured timing data is available.
# Used for (benchmark, method) pairs not in EXPERIMENT_DURATION_MINS.
METHOD_PRIORITY = {
    "gepa": 0, "best_of_k_K3": 1,
    "synaptic_pruning": 2, "slime_mold": 3, "tournament": 4,
    "ant_colony": 5, "active_minibatch": 6, "ecological_succession": 7,
    "modular": 8, "contrastive_synthesis": 9,
    "contrastive_reflection": 10, "failure_stratified_k_K3": 11,
}

# Benchmark run order: fastest wall-clock first (actual durations, not paper rollout budget)
BENCHMARK_PRIORITY = {"ifbench": 0, "pupa": 1, "livebench": 2, "hotpotqa": 3, "hover": 4, "aime": 5}

# Measured average wall-clock duration in minutes per (benchmark, method) from 27B runs.
# Within each benchmark, experiments are ordered by this — fastest first.
# Pairs not listed fall back to: 10000 + METHOD_PRIORITY * 100 (always after measured entries).
EXPERIMENT_DURATION_MINS: dict[tuple[str, str], float] = {
    # ifbench
    ("ifbench", "synaptic_pruning"):        30,
    ("ifbench", "slime_mold"):              89,
    ("ifbench", "tournament"):             168,
    ("ifbench", "gepa"):                   197,
    ("ifbench", "contrastive_reflection"): 215,
    ("ifbench", "best_of_k_K3"):           218,
    ("ifbench", "ant_colony"):             240,
    # pupa
    ("pupa", "slime_mold"):                 75,
    ("pupa", "synaptic_pruning"):           75,
    ("pupa", "tournament"):                104,
    ("pupa", "best_of_k_K3"):             222,
    ("pupa", "contrastive_reflection"):    274,
    ("pupa", "gepa"):                      328,
    # livebench
    ("livebench", "failure_stratified_k_K3"): 102,
    ("livebench", "synaptic_pruning"):     128,
    ("livebench", "slime_mold"):           217,
    ("livebench", "contrastive_reflection"): 274,
    ("livebench", "best_of_k_K3"):         317,
    ("livebench", "tournament"):           317,
    ("livebench", "gepa"):                 355,
    # hotpotqa
    ("hotpotqa", "modular"):                20,
    ("hotpotqa", "active_minibatch"):       27,
    ("hotpotqa", "contrastive_synthesis"):  28,
    ("hotpotqa", "ecological_succession"):  48,
    ("hotpotqa", "ant_colony"):             62,
    ("hotpotqa", "synaptic_pruning"):      100,
    ("hotpotqa", "slime_mold"):            149,
    ("hotpotqa", "tournament"):            188,
    ("hotpotqa", "best_of_k_K3"):          235,
    ("hotpotqa", "gepa"):                  490,
    # hover
    ("hover", "slime_mold"):               119,
    ("hover", "tournament"):               245,
    ("hover", "best_of_k_K3"):             313,
    ("hover", "gepa"):                     391,
    ("hover", "contrastive_reflection"):   454,
    ("hover", "failure_stratified_k_K3"):  519,
    # aime — no timing data yet; all fall back to METHOD_PRIORITY
}

# Map method names to runner commands
METHOD_COMMANDS = {
    "gepa": lambda bm, seed, subset, mmc: _gepa_cmd(bm, seed, subset, mmc),
    "best_of_k_K3": lambda bm, seed, subset, mmc: _mutation_cmd("best_of_k", bm, seed, 3, subset, mmc),
    "contrastive_reflection": lambda bm, seed, subset, mmc: _mutation_cmd("contrastive_reflection", bm, seed, None, subset, mmc),
    "failure_stratified_k_K3": lambda bm, seed, subset, mmc: _mutation_cmd("failure_stratified_k", bm, seed, 3, subset, mmc),
    "synaptic_pruning": lambda bm, seed, subset, mmc: _mutation_cmd("synaptic_pruning", bm, seed, None, subset, mmc),
    "tournament": lambda bm, seed, subset, mmc: _mutation_cmd("tournament", bm, seed, None, subset, mmc),
    "slime_mold": lambda bm, seed, subset, mmc: _mutation_cmd("slime_mold", bm, seed, None, subset, mmc),
    "ant_colony": lambda bm, seed, subset, mmc: _mutation_cmd("ant_colony", bm, seed, None, subset, mmc),
    "active_minibatch": lambda bm, seed, subset, mmc: _mutation_cmd("active_minibatch", bm, seed, None, subset, mmc),
    "contrastive_synthesis": lambda bm, seed, subset, mmc: _mutation_cmd("contrastive_synthesis", bm, seed, None, subset, mmc),
    "ecological_succession": lambda bm, seed, subset, mmc: _mutation_cmd("ecological_succession", bm, seed, None, subset, mmc),
    "modular": lambda bm, seed, subset, mmc: _mutation_cmd("modular", bm, seed, None, subset, mmc),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/orchestration.log"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment descriptor
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    benchmark: str
    method: str
    seed: int

    @property
    def result_path(self) -> Path:
        if _MODEL_TAG:
            return Path(f"runs/{_MODEL_TAG}/{self.benchmark}/{self.method}/{self.seed}/result.json")
        return Path(f"runs/{self.benchmark}/{self.method}/{self.seed}/result.json")

    @property
    def is_done(self) -> bool:
        return self.result_path.exists()

    @property
    def sort_key(self) -> tuple:
        duration = EXPERIMENT_DURATION_MINS.get(
            (self.benchmark, self.method),
            10000 + METHOD_PRIORITY.get(self.method, 99) * 100,
        )
        return (BENCHMARK_PRIORITY.get(self.benchmark, 99), duration, self.seed)

    def __str__(self) -> str:
        return f"{self.benchmark}/{self.method}/seed={self.seed}"


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _venv_python() -> str:
    """Path to the venv Python executable."""
    venv = Path(__file__).parent.parent / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _gepa_cmd(benchmark: str, seed: int, subset: int | None, max_metric_calls: int | None = None) -> list[str]:
    """Build command to run GEPA baseline."""
    cmd = [_venv_python(), "-m", "gepa_mutations.cli", "run", benchmark,
           "--seed", str(seed)]
    if subset is not None:
        cmd += ["--subset", str(subset)]
    if max_metric_calls is not None:
        cmd += ["--max-metric-calls", str(max_metric_calls)]
    return cmd


def _mutation_cmd(mutation: str, benchmark: str, seed: int, k: int | None, subset: int | None, max_metric_calls: int | None = None) -> list[str]:
    """Build command to run a mutation experiment."""
    script_map = {
        "best_of_k": "methods/best_of_k/best_of_k/runner.py",
        "contrastive_reflection": "methods/contrastive_reflection/contrastive_reflection/runner.py",
        "failure_stratified_k": "methods/failure_stratified_k/failure_stratified_k/runner.py",
        "synaptic_pruning": "methods/synaptic_pruning/synaptic_pruning/runner.py",
        "tournament": "methods/tournament/tournament/runner.py",
        "slime_mold": "methods/slime_mold/slime_mold/runner.py",
        "ant_colony": "methods/ant_colony/ant_colony/runner.py",
        "active_minibatch": "methods/active_minibatch/active_minibatch/runner.py",
        "contrastive_synthesis": "methods/contrastive_synthesis/contrastive_synthesis/runner.py",
        "ecological_succession": "methods/ecological_succession/ecological_succession/runner.py",
        "modular": "methods/modular/modular/runner.py",
    }
    script = Path(__file__).parent.parent / script_map[mutation]
    cmd = [_venv_python(), str(script), "--benchmark", benchmark, "--seed", str(seed)]
    if k is not None:
        cmd += ["--k", str(k)]
    if subset is not None:
        cmd += ["--subset", str(subset)]
    if max_metric_calls is not None:
        cmd += ["--max-metric-calls", str(max_metric_calls)]
    return cmd


# ---------------------------------------------------------------------------
# Telegram helpers (fail silently if not configured)
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    total_min = s // 60
    h, m = divmod(total_min, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def _esc(s: str) -> str:
    """Escape HTML special characters for Telegram HTML mode."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _progress_bar(done: int, total: int, width: int = 10) -> str:
    """Return a Unicode block progress bar, e.g. ████░░░░░░ 40/100 (40%)."""
    pct = done / total if total else 0
    filled = round(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar}  {done}/{total} ({pct:.0%})"


def _notify_html(html: str) -> None:
    """Send an HTML-formatted Telegram message. Never raises."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from gepa_mutations.config import Settings
        from gepa_mutations.notifications.notifier import Notifier
        settings = Settings()
        if settings.telegram_bot_token and settings.telegram_chat_id:
            notifier = Notifier(settings)
            notifier.send_telegram(html)
    except Exception as e:
        log.debug(f"Telegram notification failed (non-fatal): {e}")


def _build_digest(
    completed_window: list[dict],
    running_exps: set[str],
    total: int,
    completed_total: int,
    failed_total: int,
    scores: dict[str, list[float]],
    wall_elapsed: float,
    window_elapsed: float,
    workers: int,
    model_tag: str = "",
) -> str:
    """Build a mobile-friendly HTML digest message."""
    from collections import defaultdict

    lines: list[str] = []
    model_label = model_tag or "unknown-model"
    status_icon = "⚠️" if failed_total else "✅"

    # ── GPU Status Card ──────────────────────────────────────
    lines.append(f"📊 <b>Sweep Update · {_esc(model_label)}</b>")
    lines.append(f"⏱ {_fmt_duration(wall_elapsed)} elapsed  ·  +{_fmt_duration(window_elapsed)} this window")
    lines.append("")
    lines.append(
        f"{status_icon} <code>{_progress_bar(completed_total, total)}</code>"
    )
    if failed_total:
        lines.append(f"   {failed_total} failed  ·  {total - completed_total} remaining")
    else:
        lines.append(f"   {total - completed_total} remaining")

    # ── Active methods ───────────────────────────────────────
    if running_exps:
        run_groups: dict[tuple, list[str]] = defaultdict(list)
        for exp_str in running_exps:
            parts = exp_str.split("/")
            if len(parts) == 3:
                bm, method, seed_str = parts
                seed = seed_str.split("=")[-1]
                run_groups[(method, bm)].append(seed)
        active_parts = []
        for (method, bm), seeds in sorted(run_groups.items()):
            seeds_sorted = sorted(seeds, key=lambda s: int(s) if s.isdigit() else 0)
            active_parts.append(f"{_esc(method)} ×{len(seeds_sorted)}")
        lines.append(f"🔄 <b>Active ({min(workers, len(running_exps))}):</b> {', '.join(active_parts)}")

    # ── Completed this window ────────────────────────────────
    lines.append("")
    if completed_window:
        lines.append(f"<b>✅ Completed this window ({len(completed_window)}):</b>")
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for item in completed_window:
            groups[(item["exp"].method, item["exp"].benchmark)].append(item)

        if len(groups) > 6:
            method_groups: dict[str, list[dict]] = defaultdict(list)
            for item in completed_window:
                method_groups[item["exp"].method].append(item)
            for method, items in sorted(method_groups.items()):
                sc = [i["score"] for i in items if i.get("score") is not None]
                avg = f"{sum(sc)/len(sc):.0%}" if sc else "N/A"
                lines.append(f"  <code>{_esc(method)}</code>  ×{len(items)}  avg {avg}")
        else:
            for (method, bm), items in sorted(groups.items()):
                pairs = "  ".join(
                    f"s{i['exp'].seed}→{i['score']:.0%}" if i.get("score") is not None
                    else f"s{i['exp'].seed}→?"
                    for i in sorted(items, key=lambda x: x["exp"].seed)
                )
                lines.append(f"  <code>{_esc(method)}/{_esc(bm)}</code>  {_esc(pairs)}")
    else:
        lines.append("<i>No completions this window</i>")

    # ── Method scores ────────────────────────────────────────
    method_scores_lines = []
    for m in METHODS:
        sc = scores.get(m)
        if sc:
            mean = sum(sc) / len(sc)
            method_scores_lines.append(
                f"  <code>{_esc(m):<28} {mean:.1%}  ({len(sc)} seeds)</code>"
            )
    if method_scores_lines:
        lines.append("")
        lines.append("<b>Method scores so far:</b>")
        lines.extend(method_scores_lines)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess for isolation)
# ---------------------------------------------------------------------------

def _run_experiment(exp: Experiment, subset: int | None = None, max_metric_calls: int | None = None) -> dict:
    """Run one experiment. Returns result dict."""
    start = time.time()
    log.info(f"START  {exp}")

    cmd_builder = METHOD_COMMANDS.get(exp.method)
    if cmd_builder is None:
        return {"exp": str(exp), "status": "error", "error": f"Unknown method: {exp.method}"}

    cmd = cmd_builder(exp.benchmark, exp.seed, subset, max_metric_calls)
    project_root = str(Path(__file__).parent.parent)

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=36000,  # 10 hour hard limit per run
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            error_tail = result.stderr[-500:] if result.stderr else result.stdout[-500:]
            log.error(f"FAILED {exp} ({elapsed:.0f}s): {error_tail}")
            return {
                "exp": str(exp),
                "status": "failed",
                "elapsed": elapsed,
                "error": error_tail,
            }

        # Read test score from result.json
        score = None
        if exp.result_path.exists():
            try:
                data = json.loads(exp.result_path.read_text())
                score = data.get("test_score")
            except Exception:
                pass

        log.info(f"DONE   {exp} ({elapsed:.0f}s) score={score}")
        return {"exp": str(exp), "status": "ok", "elapsed": elapsed, "score": score}

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        log.error(f"TIMEOUT {exp} after {elapsed:.0f}s")
        return {"exp": str(exp), "status": "timeout", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        log.error(f"ERROR  {exp} ({elapsed:.0f}s): {e}")
        return {"exp": str(exp), "status": "error", "elapsed": elapsed, "error": str(e)}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all GEPA experiments locally")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers (default: 6)")
    parser.add_argument("--method", nargs="+", help="Filter to one or more methods")
    parser.add_argument("--benchmark", nargs="+", help="Filter to one or more benchmarks")
    parser.add_argument("--seeds", help="Comma-separated seeds (default: all 5)")
    parser.add_argument("--dry-run", action="store_true", help="Print run order without running")
    parser.add_argument("--smoke-test", action="store_true", help="Run with subset=5 (quick validation)")
    parser.add_argument("--max-metric-calls", type=int, default=None,
                        help="Override rollout budget for all runs (default: paper budget)")
    args = parser.parse_args()

    # Ensure logs dir exists
    Path("logs").mkdir(exist_ok=True)

    # Build experiment list
    methods = args.method if args.method else METHODS
    benchmarks = args.benchmark if args.benchmark else BENCHMARKS
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    subset = 5 if args.smoke_test else None

    experiments = [
        Experiment(bm, method, seed)
        for method in methods
        for bm in benchmarks
        for seed in seeds
    ]
    experiments.sort(key=lambda e: e.sort_key)

    # Filter already-done runs
    pending = [e for e in experiments if not e.is_done]
    done_count = len(experiments) - len(pending)

    print(f"\nExperiment plan:")
    print(f"  Total:   {len(experiments)}")
    print(f"  Done:    {done_count} (skipping)")
    print(f"  Pending: {len(pending)}")
    print(f"  Workers: {args.workers}")
    if subset:
        print(f"  Subset:  {subset} examples (smoke test)")
    print()

    if args.dry_run:
        print("Run order (first 20):")
        for i, e in enumerate(pending[:20]):
            print(f"  {i+1:3d}. {e}")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        return

    if not pending:
        print("All experiments complete!")
        return

    # Send start notification
    model_label = _MODEL_TAG or "unknown"
    _notify_html(
        f"🚀 <b>gepa-mutations launched</b>\n"
        f"<b>Model:</b> <code>{_esc(model_label)}</code>\n"
        f"\n"
        f"<code>{_progress_bar(done_count, len(experiments))}</code>\n"
        f"{args.workers} workers  ·  {len(pending)} pending"
    )

    # Run experiments
    total = len(pending)
    completed = 0
    failed = 0
    scores: dict[str, list[float]] = {}
    wall_start = time.time()
    last_digest = time.time()

    # Track which experiments are actively executing (approximated by pool FIFO order)
    pending_list = list(pending)  # already sorted
    queue_pos = min(args.workers, len(pending_list))
    running_exps: set[str] = {str(e) for e in pending_list[:args.workers]}
    completed_window: list[dict] = []  # accumulates since last digest

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_experiment, exp, subset, args.max_metric_calls): exp
            for exp in pending_list
        }

        for future in as_completed(futures):
            exp = futures[future]
            result = future.result()
            completed += 1

            # Update running set
            running_exps.discard(str(exp))
            if queue_pos < len(pending_list):
                running_exps.add(str(pending_list[queue_pos]))
                queue_pos += 1

            if result["status"] == "ok":
                score = result.get("score")
                elapsed = result.get("elapsed", 0)
                score_str = f"{score:.1%}" if score is not None else "N/A"
                log.info(f"[{completed}/{total}] {exp} -> {score_str} ({elapsed:.0f}s)")

                if score is not None:
                    scores.setdefault(exp.method, []).append(score)

                completed_window.append({"exp": exp, "score": score, "elapsed": elapsed})
            else:
                failed += 1
                error = result.get("error", result["status"])[:300]
                log.error(f"[{completed}/{total}] {exp} FAILED: {error}")
                # Immediate failure alert
                _notify_html(
                    f"❌ <b>[{_esc(_MODEL_TAG or 'unknown')}] FAILED</b>: <code>{_esc(str(exp))}</code>\n"
                    f"<code>{completed}/{total}  ·  {failed} failed so far</code>\n"
                    f"<code>{_esc(error[:250])}</code>"
                )

            # 30-minute digest
            now = time.time()
            if now - last_digest >= 1800:
                digest = _build_digest(
                    completed_window=completed_window,
                    running_exps=set(running_exps),
                    total=total,
                    completed_total=completed,
                    failed_total=failed,
                    scores=scores,
                    wall_elapsed=now - wall_start,
                    window_elapsed=now - last_digest,
                    workers=args.workers,
                    model_tag=_MODEL_TAG,
                )
                _notify_html(digest)
                completed_window.clear()
                last_digest = now

    # Final summary
    wall_secs = time.time() - wall_start
    result_lines: list[str] = []
    for m in METHODS:
        sc = scores.get(m, [])
        if sc:
            mean = sum(sc) / len(sc)
            result_lines.append(f"  <code>{_esc(m):<28} {mean:.1%}  ({len(sc)} seeds)</code>")
        else:
            result_lines.append(f"  <code>{_esc(m):<28} —</code>")

    model_label = _MODEL_TAG or "unknown-model"
    finish_icon = "🏁" if failed == 0 else "⚠️"
    summary_html = (
        f"{finish_icon} <b>gepa-mutations complete · {_esc(model_label)}</b>\n"
        f"⏱ {_fmt_duration(wall_secs)}\n"
        f"\n"
        f"<code>{_progress_bar(completed, total)}</code>\n"
        f"{failed} failed\n"
        f"\n"
        f"<b>Results (mean test score):</b>\n"
        + "\n".join(result_lines)
    )
    log.info(summary_html.replace("<b>", "").replace("</b>", "")
             .replace("<code>", "").replace("</code>", "")
             .replace("<i>", "").replace("</i>", ""))
    _notify_html(summary_html)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
