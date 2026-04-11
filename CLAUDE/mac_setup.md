# Mac Setup (sweep/mac branch)

## Hardware

- MacBook Pro M5 Max
- 40-core GPU (unified memory architecture)
- 48GB unified memory — shared between CPU and GPU
- No SLURM, no NFS, no HF_HOME needed

## MLX-LM

MLX-LM serves an OpenAI-compatible API on Apple Silicon. The orchestrator
(`scripts/run_all_local.py`) uses `GEPA_BASE_URL` to point to it — drop-in replacement
for vLLM with no code changes.

### Install

```bash
pip install mlx-lm
```

### Serve command

```bash
python -m mlx_lm.server \
  --model <model-id> \
  --port <port> \
  --host 127.0.0.1
```

Key flags:
- `--host 127.0.0.1` — localhost only (no external exposure needed)
- No `--enforce-eager`, `--dtype`, `--gpu-memory-utilization` — those are vLLM-only flags

---

## Active Models

| Model | MLX model ID | Port | 4-bit size | Notes |
|-------|-------------|------|-----------|-------|
| Qwen3-1.7B | `mlx-community/Qwen3-1.7B-4bit` | 8125 | ~1 GB | Fast, high throughput |
| Qwen3-4B | `mlx-community/Qwen3-4B-4bit` | 8126 | ~3 GB | Good quality/speed balance |
| Qwen3-32B | `mlx-community/Qwen3-32B-4bit` | 8131 | ~20 GB | Largest model, unique size |

Total memory: ~24GB — fits all three simultaneously in 48GB.

Model files cache to `~/.cache/huggingface/` by default (no HF_HOME override needed).

### Alternative: higher-precision 32B

If 4-bit feels too lossy, `mlx-community/Qwen3-32B-8bit` uses ~38GB — still fits in 48GB
but leaves little headroom. Stick with 4-bit unless you have a specific reason to upgrade.

---

## Serve Scripts

| Script | Model | Port |
|--------|-------|------|
| `scripts/serve_mlx_1b_mac.sh` | Qwen3-1.7B | 8125 |
| `scripts/serve_mlx_4b_mac.sh` | Qwen3-4B | 8126 |
| `scripts/serve_mlx_32b_mac.sh` | Qwen3-32B | 8131 |

Run directly: `bash scripts/serve_mlx_32b_mac.sh`

---

## Worker Count Guidelines

MLX-LM is single-process but handles concurrent async requests well.

| Model | Recommended workers | Notes |
|-------|--------------------|----|
| 1.7B | 6–8 | Very fast; can saturate quickly |
| 4B | 5–7 | Good throughput |
| 32B | 2–4 | Slower per call; keep low to avoid timeouts |

---

## Port Assignments

Mac ports use localhost only — no conflict with cluster IPs:

| Port | Model | Machine |
|------|-------|---------|
| 8125 | Qwen3-1.7B | Mac (localhost) |
| 8126 | Qwen3-4B | Mac (localhost) |
| 8131 | Qwen3-32B | Mac (localhost) |

(Cluster uses 8124=27B, 8125=8B, 8128=14B on remote IPs — no overlap with Mac localhost.)

---

## Model Download

Pre-download before starting experiments (saves time mid-run):

```bash
python -c "from mlx_lm import load; load('mlx-community/Qwen3-1.7B-4bit')"
python -c "from mlx_lm import load; load('mlx-community/Qwen3-4B-4bit')"
python -c "from mlx_lm import load; load('mlx-community/Qwen3-32B-4bit')"
```

Downloads go to `~/.cache/huggingface/hub/`. Qwen3-32B-4bit is ~20GB — allow time on first run.

---

## Monitoring Without Crons

No system crontab available on Mac. Check progress manually:

```bash
# Quick progress counts
for tag in qwen3-1.7b qwen3-4b qwen3-32b; do
  count=$(find runs/$tag -name result.json 2>/dev/null | wc -l | tr -d ' ')
  echo "$tag: $count / 300"
done

# Send Telegram update manually
.venv/bin/python scripts/monitor_multi_model.py --mode 30min

# Watch a log live
tail -f logs/orchestrator_32b.log
```
