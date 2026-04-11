# Monitoring Setup

## Cron Jobs

Two cron jobs run automatically from the login node crontab (`crontab -e` to inspect):

| Cron | Schedule | Command | What it does |
|------|----------|---------|--------------|
| 15-min check | `*/15 * * * *` | `monitor_multi_model.py --mode 15min` | Per-model Telegram messages with running/queued status |
| 30-min check | `*/30 * * * *` | `monitor_multi_model.py --mode 30min` | Consolidated sweep health dashboard |
| Node recovery | `*/15 * * * *` | `scripts/check_node_recovery.sh` | Pings downed nodes; Telegram alert if they come back |

Run manually:
```bash
cd /users/achidamb/projects/gepa-mutations/source/gepa-mutations-raycluster
.venv/bin/python scripts/monitor_multi_model.py --mode 15min
.venv/bin/python scripts/monitor_multi_model.py --mode 30min
```

## Monitoring Script Config

`scripts/monitor_multi_model.py` defines `MODELS` list (lines ~76–105) — each entry has:
- `tag`: run directory prefix (e.g. `qwen3-8b`)
- `display`: human label for Telegram messages
- `cluster`: which node it's running on
- `health_url`: vLLM `/v1/models` endpoint for liveness check
- `log_glob`: orchestrator log filename in `logs/`

**To add a new model worker:** append a new dict to the `MODELS` list. Update `TOTAL_PER_MODEL` if adding a new unique model (not a 2nd worker for an existing model tag).

## vLLM Health Check

The monitor uses `curl -sf --max-time 5 <url>/v1/models` to check liveness.

To check manually:
```bash
curl -sf http://10.0.10.58:8125/v1/models | python3 -m json.tool
```

For actual request load (more reliable than filesystem detection):
```bash
curl -s http://10.0.10.58:8125/metrics | grep 'num_requests_running'
```

## Snapshot File

`logs/.multi_model_snapshot.json` — written by monitor after each run.
Stores `{models: {tag: {done: N, running: N}}}` to compute deltas between checks.
Gitignored.

## Experiment Progress Detection

`get_done()` — scans `runs/<tag>/*/result.json` files. Reliable.

`get_in_progress()` — scans run dirs for recently modified files (within 45 min). 
**Known limitation:** `tournament` and other methods that don't write intermediate state files
won't appear as "running" until they checkpoint. Use vLLM `/metrics` to confirm real activity.

## Status Icons in Reports

| Icon | Meaning |
|------|---------|
| 🟢 | vLLM healthy + new completions since last check |
| 🟡 | vLLM healthy but no new completions |
| 🔴 | vLLM endpoint unreachable |

## Telegram Notifications

Configured via `.env` or environment variables:
- `TELEGRAM_BOT_TOKEN` — bot token from BotFather
- `TELEGRAM_CHAT_ID` — target chat/channel ID

The orchestrator (`run_all_local.py`) sends:
- Launch notification on start
- Immediate alert on any experiment failure
- 30-min digest during run
- Final summary on completion

## Node Recovery Monitoring

`scripts/check_node_recovery.sh` is a lightweight script that pings the three downed nodes
(bourbaki, ansatz, deepseek). On recovery it sends a Telegram alert.

**Do NOT auto-relaunch on recovery** — discuss with user first before relaunching vLLM
servers or orchestrators on recovered nodes. The recovered node may have stale SLURM state.
