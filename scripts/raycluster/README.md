# Raycluster Experiment Runner

Run prompt optimization experiments on the SupportVectors GPU cluster.

## Architecture

```
Mac (local)                    gho-vm-2 (cluster)              GPU server
─────────────                  ────────────────                ──────────
deploy.sh ──rsync──→  scripts/ + src/               
run_remote.sh ──ssh──→  run_baseline.py ──http──→  10.0.10.66:8123
                                                    Qwen3.5-27B
results ←──scp──────  runs/qwen3.5-27b/
```

## Setup

1. **Activate WireGuard VPN** (config at `DO-NOT-TOUCH/gpu-cluster-access/`)
2. **Deploy to cluster:**
   ```bash
   bash scripts/raycluster/deploy.sh
   ```
3. **Test connectivity (runs on cluster):**
   ```bash
   bash scripts/raycluster/run_remote.sh status
   ```

## Running Experiments

```bash
# Start baseline (all benchmarks, all seeds — runs in background on cluster)
bash scripts/raycluster/run_remote.sh baseline

# Start baseline on specific benchmarks
bash scripts/raycluster/run_remote.sh baseline --benchmark hotpotqa hover

# Monitor progress
bash scripts/raycluster/run_remote.sh logs

# Check if jobs are still running
bash scripts/raycluster/run_remote.sh status

# Copy results back to local runs/ directory
bash scripts/raycluster/run_remote.sh results
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Cluster settings (endpoints, model, hyperparameters) |
| `deploy.sh` | Rsync minimal code to gho-vm-2 |
| `run_remote.sh` | SSH helper (start/monitor/fetch results) |
| `test_connectivity.py` | Verify API + measure latency |
| `run_baseline.py` | Baseline evaluation (no optimization) |
| `stubs/gepa/` | Minimal type stubs (avoids pulling full gepa submodule) |

## Model Details

- **Model:** Qwen/Qwen3.5-27B (131k context, reasoning model)
- **Alias:** `openai/gpt-oss-120b` (SupportVectors internal naming)
- **Endpoint:** `http://10.0.10.66:8123/v1` (OpenAI-compatible)
- **Behavior:** Reasoning model — responses may be in `reasoning_content` field

## Results

Stored in `runs/qwen3.5-27b/{benchmark}/baseline/{seed}/`:
- `result.json` — scores + metadata (includes `"infra": "raycluster"`)
- `metrics.json` — token usage, timing
- `test_outputs.json` — raw responses for analysis
