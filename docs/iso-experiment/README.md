# ISO Experiment Infrastructure

Production-grade experiment harness for ISO prompt optimizer research on RunPod.
Supports dual-model vLLM serving (Qwen3-8B task + Qwen3-32B-AWQ reflection),
7-layer data collection, checkpoint/resume, and automated validation.

## Quick Start

1. Provision RunPod pod (L40S for pilot, H100 for full run, Secure Cloud)
2. Clone repo and run setup
3. Download models
4. Start servers
5. Validate system
6. Run pilot

## Prerequisites

- RunPod account with Secure Cloud access
- Network Volume (200GB minimum) mounted at `/workspace`
- SSH key pair for rsync (local machine -> pod)

## Step-by-Step Guide

### 1. Pod Provisioning

- Template: PyTorch 2.4+ / CUDA 12.4 / Ubuntu 22.04
- Phase 0 (pilot): L40S 48GB (~$0.79/hr)
- Phase 1 (full): H100 PCIe 80GB (~$1.99/hr)
- Mount Network Volume at `/workspace`

### 2. Environment Setup

```bash
cd /workspace
git clone https://github.com/ashwinchidambaram/gepa-mutations.git
cd gepa-mutations
bash scripts/iso_setup.sh
```

### 3. Model Download

```bash
bash scripts/iso_download_models.sh
# Pin the printed SHAs in configs/pilot.yaml and configs/full.yaml
```

### 4. Server Launch

```bash
# Pilot (L40S):
bash scripts/iso_start_servers.sh --phase pilot

# Full (H100):
bash scripts/iso_start_servers.sh --phase full
```

Inspect servers: `tmux attach -t iso-task`, `tmux attach -t iso-reflection`

### 5. System Validation

```bash
bash scripts/iso_validate_system.sh
# Produces validation_report.md — all V1-V10 must pass
```

### 6. Run Pilot

```bash
# Dry run (validate config, no rollouts):
bash scripts/iso_run_pilot.sh --dry-run

# Smoke test (50 examples, 100 rollouts, ~20 min):
bash scripts/iso_run_pilot.sh --smoke-test

# Full pilot:
bash scripts/iso_run_pilot.sh
```

### 7. Run Full Experiment

```bash
# Requires clean git tree (all changes committed):
bash scripts/iso_run_full.sh --config configs/full.yaml
```

### 8. Data Sync (from local machine)

```bash
# Configure .env with POD_SSH_TARGET, SSH_KEY
bash scripts/iso_sync_from_pod.sh --dry-run  # Preview
bash scripts/iso_sync_from_pod.sh             # Sync + git commit
```

## Architecture

```
src/iso_harness/experiment/
  schemas.py         # 7-layer Pydantic data models
  config.py          # YAML config validation
  context.py         # contextvars propagation
  jsonl_writer.py    # Atomic append-only JSONL
  logging_lm.py      # LM wrapper with JSONL streaming
  mlflow_setup.py    # MLflow integration (SQLite)
  checkpoint.py      # Round-level checkpoint/resume
  orchestrator.py    # Run matrix + budget enforcement
  monitor.py         # GPU/KV cache/disk telemetry
  consolidate.py     # JSONL -> Parquet pipeline
  reporter.py        # Per-run reports (summary.json, report.md, CSV)
  protocols.py       # Optimizer/FeedbackFunction/Checkpointable interfaces
```

## Recovery Procedures

### Pod restart (planned or unplanned)

1. Network Volume preserves all data
2. Re-run `bash scripts/iso_start_servers.sh`
3. Re-run experiment -- orchestrator auto-resumes from last checkpoint

### Checkpoint resume

The orchestrator checks for existing checkpoints before each run.
If found, it skips completed rounds and continues from the next.

### Failed run

Failed runs are logged to `runs/{run_id}/errors.log`.
The orchestrator continues to the next run unless 3+ consecutive failures occur.

## Data Layout

```
runs/{run_id}/
  config.json        # Run configuration snapshot
  rollouts.jsonl     # Per-rollout data (Layer 1)
  reflections.jsonl  # Per-reflection data (Layer 2)
  candidates.jsonl   # Candidate lifecycle (Layer 3)
  rounds.jsonl       # Per-round aggregates (Layer 4)
  summary.json       # Run summary (Layer 5)
  telemetry.jsonl    # System telemetry (Layer 6)
  report.md          # Human-readable summary
  COMPLETE           # Completion marker (for sync detection)
  checkpoints/       # Round-level checkpoints
    round_000/
      state.json
      candidates.json
      metrics.json
```
