#!/usr/bin/env python3
"""
Multi-model sweep monitor.

Modes:
  --mode 15min   Send 4 separate Telegram messages (one per model) with per-model status
  --mode 30min   Send 1 consolidated Telegram message with full sweep health

Usage:
  .venv/bin/python scripts/monitor_multi_model.py --mode 15min
  .venv/bin/python scripts/monitor_multi_model.py --mode 30min
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
RUNS_DIR = BASE_DIR / "runs"
LOGS_DIR = BASE_DIR / "logs"
SNAPSHOT  = LOGS_DIR / ".multi_model_snapshot.json"

BENCHMARKS = ["hotpotqa", "hover", "pupa", "ifbench", "livebench"]
METHODS = [
    "gepa", "best_of_k_K3",
    "synaptic_pruning", "tournament", "slime_mold",
    "ant_colony", "active_minibatch", "ecological_succession",
    "modular", "contrastive_synthesis",
    "contrastive_reflection", "failure_stratified_k_K3",
]
METHOD_PRIORITY = {m: i for i, m in enumerate(METHODS)}
SEEDS = [42, 123, 456, 789, 1024]

TOTAL_PER_MODEL = len(BENCHMARKS) * len(METHODS) * len(SEEDS)  # 300

BENCHMARK_PRIORITY = {"ifbench": 0, "pupa": 1, "livebench": 2, "hotpotqa": 3, "hover": 4}

# Measured avg wall-clock (minutes) per (benchmark, method) from 27B runs.
# Used to order the "Queued Next" display to match actual execution order.
EXPERIMENT_DURATION_MINS: dict[tuple[str, str], float] = {
    ("ifbench", "synaptic_pruning"): 30, ("ifbench", "slime_mold"): 89,
    ("ifbench", "tournament"): 168, ("ifbench", "gepa"): 197,
    ("ifbench", "contrastive_reflection"): 215, ("ifbench", "best_of_k_K3"): 218,
    ("ifbench", "ant_colony"): 240,
    ("pupa", "slime_mold"): 75, ("pupa", "synaptic_pruning"): 75,
    ("pupa", "tournament"): 104, ("pupa", "best_of_k_K3"): 222,
    ("pupa", "contrastive_reflection"): 274, ("pupa", "gepa"): 328,
    ("livebench", "failure_stratified_k_K3"): 102, ("livebench", "synaptic_pruning"): 128,
    ("livebench", "slime_mold"): 217, ("livebench", "contrastive_reflection"): 274,
    ("livebench", "best_of_k_K3"): 317, ("livebench", "tournament"): 317,
    ("livebench", "gepa"): 355,
    ("hotpotqa", "modular"): 20, ("hotpotqa", "active_minibatch"): 27,
    ("hotpotqa", "contrastive_synthesis"): 28, ("hotpotqa", "ecological_succession"): 48,
    ("hotpotqa", "ant_colony"): 62, ("hotpotqa", "synaptic_pruning"): 100,
    ("hotpotqa", "slime_mold"): 149, ("hotpotqa", "tournament"): 188,
    ("hotpotqa", "best_of_k_K3"): 235, ("hotpotqa", "gepa"): 490,
    ("hover", "slime_mold"): 119, ("hover", "tournament"): 245,
    ("hover", "best_of_k_K3"): 313, ("hover", "gepa"): 391,
    ("hover", "contrastive_reflection"): 454, ("hover", "failure_stratified_k_K3"): 519,
}

_ACTIVE_BENCHMARKS = set(BENCHMARKS)

# Model definitions
# cluster = node currently serving this model (updated 2026-04-10: bourbaki/ansatz/deepseek down,
#           substituted with manifold/sapphire/sar-gpu-vm on capstone partition)
MODELS = [
    {
        "tag":        "qwen3-27b-awq",
        "display":    "Qwen3-27B-AWQ",
        "cluster":    "manifold (capstone)",
        "health_url": "http://10.0.10.69:8124/v1/models",
        "log_glob":   "orchestrator_27b.log",
    },
    {
        "tag":        "qwen3-8b",
        "display":    "Qwen3-8B",
        "cluster":    "archimedes",
        "health_url": "http://10.0.10.58:8125/v1/models",
        "log_glob":   "orchestrator_8b.log",
    },
    {
        "tag":        "qwen3-4b",
        "display":    "Qwen3-4B",
        "cluster":    "sapphire (capstone)",
        "health_url": "http://10.0.100.99:8126/v1/models",
        "log_glob":   "orchestrator_4b.log",
    },
    {
        "tag":        "qwen3-1.7b",
        "display":    "Qwen3-1.7B",
        "cluster":    "sar-gpu-vm (capstone)",
        "health_url": "http://10.0.50.99:8127/v1/models",
        "log_glob":   "orchestrator_1b.log",
    },
]


# ---------------------------------------------------------------------------
# State collection
# ---------------------------------------------------------------------------

def model_runs_dir(tag: str | None) -> Path:
    return RUNS_DIR if tag is None else RUNS_DIR / tag


def get_done(tag: str | None) -> set[tuple[str, str, int]]:
    """Return (bm, method, seed) for completed runs."""
    done: set[tuple[str, str, int]] = set()
    base = model_runs_dir(tag)
    if not base.exists():
        return done
    for bm_dir in base.iterdir():
        if not bm_dir.is_dir() or bm_dir.name.startswith("."):
            continue
        # Skip model-tag subdirs when reading non-tagged (27B) path
        if tag is None and bm_dir.name.startswith("qwen3-"):
            continue
        for method_dir in bm_dir.iterdir():
            if not method_dir.is_dir():
                continue
            for seed_dir in method_dir.iterdir():
                try:
                    seed = int(seed_dir.name)
                except ValueError:
                    continue
                if (seed_dir / "result.json").exists():
                    done.add((bm_dir.name, method_dir.name, seed))
    return done


def get_in_progress(tag: str | None, stale_minutes: float = 45.0) -> set[tuple[str, str, int]]:
    """Return (bm, method, seed) for runs with activity but no result (recently modified)."""
    active: set[tuple[str, str, int]] = set()
    base = model_runs_dir(tag)
    if not base.exists():
        return active
    cutoff = time.time() - stale_minutes * 60
    for bm_dir in base.iterdir():
        if not bm_dir.is_dir() or bm_dir.name.startswith("."):
            continue
        if tag is None and bm_dir.name.startswith("qwen3-"):
            continue
        # Only consider active benchmarks — ignore aime and other excluded dirs
        if bm_dir.name not in _ACTIVE_BENCHMARKS:
            continue
        for method_dir in bm_dir.iterdir():
            if not method_dir.is_dir():
                continue
            for seed_dir in method_dir.iterdir():
                try:
                    seed = int(seed_dir.name)
                except ValueError:
                    continue
                result = seed_dir / "result.json"
                if result.exists():
                    continue
                # Check if any file in the run dir was modified recently
                try:
                    latest_mtime = max(
                        (f.stat().st_mtime for f in seed_dir.rglob("*") if f.is_file()),
                        default=0,
                    )
                    if latest_mtime > cutoff:
                        active.add((bm_dir.name, method_dir.name, seed))
                except Exception:
                    pass
    return active


def get_queued(
    done: set[tuple[str, str, int]],
    in_progress: set[tuple[str, str, int]],
    n: int = 5,
) -> list[tuple[str, str, int]]:
    """Return next N pending experiments in actual execution order (benchmark-first, fastest method first)."""
    def sort_key(bm: str, method: str) -> tuple:
        duration = EXPERIMENT_DURATION_MINS.get(
            (bm, method), 10000 + METHOD_PRIORITY.get(method, 99) * 100
        )
        return (BENCHMARK_PRIORITY.get(bm, 99), duration)

    pending = []
    for bm in sorted(BENCHMARKS, key=lambda b: BENCHMARK_PRIORITY.get(b, 99)):
        for method in sorted(METHODS, key=lambda m: sort_key(bm, m)):
            for seed in SEEDS:
                key = (bm, method, seed)
                if key not in done and key not in in_progress:
                    pending.append(key)
    return pending[:n]


def check_health(url: str) -> bool:
    try:
        r = subprocess.run(
            ["curl", "-sf", "--max-time", "5", url],
            capture_output=True, timeout=8,
        )
        return r.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def load_snapshot() -> dict:
    try:
        return json.loads(SNAPSHOT.read_text())
    except Exception:
        return {}


def save_snapshot(data: dict) -> None:
    SNAPSHOT.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Telegram sender
# ---------------------------------------------------------------------------

def send_telegram(message: str) -> None:
    try:
        sys.path.insert(0, str(BASE_DIR / "src"))
        from gepa_mutations.config import Settings
        import asyncio
        import telegram
        s = Settings()
        if not (s.telegram_bot_token and s.telegram_chat_id):
            print("WARNING: Telegram not configured")
            return
        bot = telegram.Bot(token=s.telegram_bot_token)
        asyncio.run(bot.send_message(
            chat_id=s.telegram_chat_id,
            text=message,
            parse_mode="HTML",
        ))
        print(f"Sent ({len(message)} chars)")
    except Exception as e:
        print(f"Telegram send failed: {e}")


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def status_icon(healthy: bool, done: int, prev_done: int) -> str:
    """Return 🟢/🟡/🔴 based on vLLM health and progress."""
    if not healthy:
        return "🔴"
    if done > prev_done:
        return "🟢"
    return "🟡"


def fmt_exp(bm: str, method: str, seed: int) -> str:
    return f"{method} ; {bm}/seed={seed}"


# ---------------------------------------------------------------------------
# 15-min mode: 4 separate per-model messages
# ---------------------------------------------------------------------------

def run_15min() -> None:
    now_str = datetime.now().strftime("%b %-d  %H:%M")
    snap = load_snapshot()
    new_snap: dict = {"ts": datetime.now().isoformat(), "models": {}}

    for m in MODELS:
        tag     = m["tag"]
        display = m["display"]
        cluster = m["cluster"]
        key     = tag or "27b"

        done        = get_done(tag)
        in_progress = get_in_progress(tag)
        queued      = get_queued(done, in_progress, n=5)
        healthy     = check_health(m["health_url"])

        n_done     = len(done)
        prev_done  = snap.get("models", {}).get(key, {}).get("done", 0)
        delta      = n_done - prev_done
        n_running  = len(in_progress)
        n_remaining = TOTAL_PER_MODEL - n_done

        icon = status_icon(healthy, n_done, prev_done)
        delta_str = f" (+{delta} since last check)" if delta > 0 else (" (no new completions)" if prev_done > 0 else "")

        lines = [
            f"<b>{display}: {cluster}</b>  {icon}",
            f"{n_done}/{TOTAL_PER_MODEL} tests completed{delta_str}",
            "",
        ]

        # Currently running
        lines.append("<b>Currently Running Tests:</b>")
        if in_progress:
            for bm, method, seed in sorted(in_progress)[:8]:
                lines.append(f"  - {fmt_exp(bm, method, seed)}")
        else:
            lines.append("  (none detected)")

        # Queued next
        lines.append("")
        lines.append("<b>Queued Next:</b>")
        if queued:
            for bm, method, seed in queued[:5]:
                lines.append(f"  - {fmt_exp(bm, method, seed)}")
        else:
            lines.append("  (queue empty — all experiments started or done)")

        lines.append("")
        lines.append(f"<b>Number of tests remaining:</b> {n_remaining}")

        send_telegram("\n".join(lines))

        new_snap["models"][key] = {"done": n_done, "running": n_running}

    save_snapshot(new_snap)
    print(f"15-min check complete at {now_str}")


# ---------------------------------------------------------------------------
# 30-min mode: 1 consolidated message
# ---------------------------------------------------------------------------

def run_30min() -> None:
    now_str = datetime.now().strftime("%b %-d  %H:%M")
    snap = load_snapshot()
    new_snap: dict = {"ts": datetime.now().isoformat(), "models": {}}

    total_done = 0
    total_prev = 0
    total_remaining = 0
    total_all = len(MODELS) * TOTAL_PER_MODEL

    cluster_lines = []
    issues = []

    for m in MODELS:
        tag     = m["tag"]
        display = m["display"]
        cluster = m["cluster"]
        key     = tag or "27b"

        done        = get_done(tag)
        in_progress = get_in_progress(tag)
        healthy     = check_health(m["health_url"])

        n_done     = len(done)
        prev_done  = snap.get("models", {}).get(key, {}).get("done", 0)
        n_remaining = TOTAL_PER_MODEL - n_done

        icon = status_icon(healthy, n_done, prev_done)

        total_done      += n_done
        total_prev      += prev_done
        total_remaining += n_remaining

        cluster_lines.append(f"  - {display}: {cluster} {icon}")
        new_snap["models"][key] = {"done": n_done, "running": len(in_progress)}

        if not healthy:
            issues.append(f"⚠ {display} ({cluster}) vLLM endpoint is DOWN")
        elif n_done == prev_done and prev_done > 0 and len(in_progress) == 0:
            issues.append(f"⚠ {display} ({cluster}) — no new completions and no active runners")

    total_delta = total_done - total_prev
    delta_str = f" (+{total_delta} since last 30 min)" if total_delta != 0 else ""

    lines = [
        f"📊 <b>30 Minute Update</b>  ·  {now_str}",
        "",
        f"<b>{total_done}/{total_all} tests completed{delta_str}</b>",
        "",
        "<b>Cluster Health:</b>",
    ]
    lines.extend(cluster_lines)

    lines.append("")
    lines.append("<b>Issues (if any):</b>")
    if issues:
        for issue in issues:
            lines.append(f"  {issue}")
    else:
        lines.append("  None — all clusters healthy and progressing ✅")

    lines.append("")
    lines.append(f"<b>Total Number of tests remaining:</b> {total_remaining}")

    send_telegram("\n".join(lines))
    save_snapshot(new_snap)
    print(f"30-min check complete at {now_str}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["15min", "30min"], required=True)
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    if args.mode == "15min":
        run_15min()
    else:
        run_30min()


if __name__ == "__main__":
    main()
