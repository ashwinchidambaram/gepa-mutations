# AWS Implementation: contrastive_reflection Experiment Sweep

> Deployment plan for running the contrastive_reflection parameter sweep on AWS.
> This experiment is **CPU/API-bound** (no GPU). All model inference goes through OpenRouter API calls to `openrouter/qwen/qwen3-8b`.

---

## 1. EC2 Instance Selection

### Recommended Instance: `t3.medium` (spot)

| Attribute | Value | Rationale |
|-----------|-------|-----------|
| Instance type | `t3.medium` | 2 vCPU, 4 GB RAM. Sufficient for a Python process that spends 95%+ of wall-clock time waiting on OpenRouter API responses. |
| vCPUs | 2 | One for the experiment process, one for OS + checkpoint upload. No parallelism needed within a single run. |
| Memory | 4 GB | GEPA state, candidate pool, and contrastive train index all fit comfortably in <1 GB even at 7000+ rollouts. The reflective dataset and evaluation cache are small text structures. |
| Pricing model | **Spot** | See justification below. |
| AMI | Amazon Linux 2023 (arm64 or x86_64) | Minimal, fast boot. Python 3.12 available via `dnf`. |
| Architecture | `x86_64` preferred | Broader spot availability. `arm64` (`t4g.medium`) is ~20% cheaper but has occasional capacity issues in some AZs. Use `t4g.medium` if available. |
| EBS | 20 GB `gp3` | Code + venv + checkpoint data. GEPA state serializations are <50 MB per run. |

### Why not a larger instance?

The workload is **I/O-bound on network latency to OpenRouter**, not CPU-bound. Profiling the existing `ExperimentRunner` shows:
- ~95% of wall time is `litellm.completion()` calls (network I/O wait)
- ~4% is evaluation scoring (string matching, F1 computation)
- ~1% is contrastive search, candidate selection, state management

A `t3.large` (8 GB) or `c6i.large` would waste money. The extra CPU and memory sit idle while waiting for API responses.

### Why spot, not on-demand?

| Factor | Spot | On-demand |
|--------|------|-----------|
| `t3.medium` price (us-east-1) | ~$0.0125/hr | ~$0.0416/hr |
| Savings | **70%** | baseline |
| Interruption risk | 2-min warning, ~5% frequency for t3 | none |
| Mitigation | Checkpoint-resume (Section 2) handles this cleanly | N/A |

The 70% savings across a 36-run sweep outweighs the small risk of interruption. Each run checkpoints to S3 every 50 iterations, so a spot interruption loses at most ~15-20 minutes of work.

**Fallback**: If spot capacity is unavailable in the selected AZ, the launch script falls back to on-demand automatically (see Section 8).

### Parallel execution strategy

Run **up to 6 instances in parallel** (one per configuration within a benchmark-seed batch). Each instance handles one `(benchmark, config, seed)` tuple. This keeps the sweep wall-clock manageable without hitting OpenRouter rate limits.

| Parallelism | Instances | OpenRouter RPM impact | Wall-clock for Tier 1+2 |
|-------------|-----------|----------------------|------------------------|
| 1 serial | 1 | ~15 RPM | ~108 hours |
| 6 parallel | 6 | ~90 RPM | ~18 hours |
| 12 parallel | 12 | ~180 RPM | ~9 hours |

**Recommendation**: 6 parallel. OpenRouter's free tier allows 200 RPM for Qwen3-8B. At 6 instances, we use ~90 RPM, leaving 50% headroom for retries and bursts.

---

## 2. S3 Storage

### Bucket structure

Uses the existing `gepa-mutations-results` bucket (created by `scripts/aws_setup.py`). The `contrastive_reflection` experiment extends the existing path convention.

```
s3://gepa-mutations-results/
  runs/
    <benchmark>/
      contrastive_reflection/          # method = "contrastive_reflection"
        <config_id>/                   # e.g., "cr_snippet300" or "vanilla_control"
          <seed>/
            config.json                # Full experiment config snapshot
            result.json                # Final scores, best prompt, rollout count
            metrics.json               # Per-iteration diagnostics from MetricsCallback
            contrastive_metrics.json   # Contrastive-specific: injection count, snippet lengths, pool sizes
            checkpoints/
              checkpoint_iter_0050.json
              checkpoint_iter_0100.json
              ...                      # Incremental checkpoints every 50 iterations
            logs/
              experiment.log           # Full experiment log
```

### Config ID naming convention

| Config | `config_id` |
|--------|-------------|
| Vanilla GEPA control | `vanilla_control` |
| Contrastive, snippet_length=150 | `cr_snippet150` |
| Contrastive, snippet_length=300 | `cr_snippet300` |
| Contrastive, snippet_length=500 | `cr_snippet500` |

For secondary sweep configs:
- `cr_thresh030_gap010` (failing_threshold=0.3, min_contrastive_score_gap=0.1)
- `cr_source_random` (contrastive_source=random_solver)
- etc.

### Checkpointing for spot interruption recovery

**Checkpoint content** (JSON, ~1-5 MB per checkpoint):
```json
{
  "iteration": 150,
  "total_metric_calls": 2100,
  "gepa_state_hash": "sha256:abc123...",
  "contrastive_train_index": { ... },
  "best_val_score": 0.583,
  "best_candidate_idx": 7,
  "wall_clock_elapsed_seconds": 3420,
  "timestamp_utc": "2026-03-20T14:30:00Z"
}
```

**Checkpoint frequency**: Every 50 iterations (approximately every 10-15 minutes of wall time). This balances S3 PUT cost ($0.000005 per PUT) against lost work on interruption.

**Checkpoint upload**:
1. Write checkpoint JSON + GEPA state to local `/tmp/checkpoint/`
2. Upload to S3 with `aws s3 sync` (atomic per-file)
3. Write a `latest_checkpoint.json` pointer file last (acts as a commit marker)

**Resume flow**:
1. On instance boot, check for `s3://.../checkpoints/latest_checkpoint.json`
2. If found, download the referenced checkpoint + state files
3. Restore GEPA state, contrastive train index, and iteration counter
4. Resume from `iteration + 1`
5. If `latest_checkpoint.json` is missing or corrupt, start from scratch

---

## 3. Parameter Sweep Execution

### Primary sweep matrix

From the mutation plan, the primary sweep tests 2 parameters across these values:

| Parameter | Values | Count |
|-----------|--------|-------|
| `use_contrastive_reflection` | `[True, False]` | 2 |
| `contrastive_snippet_length` | `[150, 300, 500]` | 3 (only when `True`) |

This yields **4 distinct configurations** per benchmark (not 6, because `snippet_length` is irrelevant when `use_contrastive_reflection=False`):

| Config # | `use_contrastive_reflection` | `contrastive_snippet_length` | `config_id` |
|----------|------------------------------|------------------------------|-------------|
| 1 | `False` | N/A | `vanilla_control` |
| 2 | `True` | 150 | `cr_snippet150` |
| 3 | `True` | 300 | `cr_snippet300` |
| 4 | `True` | 500 | `cr_snippet500` |

### Seeds

From the mutation plan: `[42, 137, 2025]` (3 seeds per config).

### Benchmarks (tiered)

| Tier | Benchmark | Rollout budget | Priority |
|------|-----------|---------------|----------|
| 1 | HotpotQA | 6,871 | Run first |
| 2 | HoVer | 2,426 | Run second |
| 3 | IFBench | 3,593 | Run if Tiers 1-2 positive |
| 3 | PUPA | 3,936 | Run if Tiers 1-2 positive |
| 4 | AIME-2025 | 7,051 | Run last |
| 4 | LiveBench-Math | 1,839 | Run last |

### Total run counts

| Scope | Configs | Seeds | Benchmarks | Total runs |
|-------|---------|-------|------------|------------|
| Tier 1 only (HotpotQA) | 4 | 3 | 1 | **12** |
| Tier 1 + Tier 2 | 4 | 3 | 2 | **24** |
| Full primary sweep (Tiers 1-4) | 4 | 3 | 6 | **72** |
| Secondary sweep (per benchmark) | 27 | 3 | 1 | **81** |

**Recommended execution order**:
1. Tier 1: 12 runs on HotpotQA (validate the mutation works)
2. Tier 2: 12 runs on HoVer (validate generalization)
3. Analyze Tier 1+2 results before committing to Tier 3+4
4. Tier 3: 24 runs (IFBench + PUPA) only if positive signal
5. Tier 4: 24 runs (AIME + LiveBench) only if strong signal

### Execution batching

With 6 parallel instances:
- **Batch 1** (Tier 1): 12 runs / 6 parallel = 2 waves. Each HotpotQA run takes ~3 hours. Total: ~6 hours.
- **Batch 2** (Tier 2): 12 runs / 6 parallel = 2 waves. Each HoVer run takes ~1 hour. Total: ~2 hours.
- **Batch 3** (Tier 3): 24 runs / 6 parallel = 4 waves. Mix of IFBench (~1.5 hr) and PUPA (~1.5 hr). Total: ~6 hours.

---

## 4. Cost Optimization

### EC2 compute costs

| Item | Per-run | Tier 1 (12 runs) | Tier 1+2 (24 runs) | Full sweep (72 runs) |
|------|---------|-------------------|---------------------|----------------------|
| Instance hours (spot, $0.0125/hr) | | | | |
| - HotpotQA (~3 hr/run) | $0.038 | $0.45 | - | - |
| - HoVer (~1 hr/run) | $0.013 | - | $0.15 | - |
| - IFBench (~1.5 hr/run) | $0.019 | - | - | - |
| - PUPA (~1.5 hr/run) | $0.019 | - | - | - |
| - AIME (~3 hr/run) | $0.038 | - | - | - |
| - LiveBench (~0.8 hr/run) | $0.010 | - | - | - |
| **EC2 total** | - | **$0.45** | **$0.60** | **$1.50** |
| EBS (20 GB gp3, $0.08/GB/mo) | $0.001/hr | $0.04 | $0.08 | $0.25 |

EC2 costs are negligible. The real cost is OpenRouter API.

### OpenRouter API costs

**Qwen3-8B pricing** (OpenRouter):
- Input: $0.05 per 1M tokens
- Output: $0.40 per 1M tokens

**Per-iteration token estimates** (from existing runner profiling):
- Evaluation call: ~800 input tokens, ~200 output tokens
- Reflection call: ~2,000 input tokens (includes reflective dataset + contrastive snippet), ~500 output tokens
- Contrastive overhead: +75 input tokens per reflection call (the injected snippet)

**Per-run cost model**:

| Benchmark | Rollouts | Eval calls | Reflection calls | Input tokens (M) | Output tokens (M) | API cost |
|-----------|----------|------------|-------------------|-------------------|--------------------| ---------|
| HotpotQA | 6,871 | 6,871 | ~2,290 | ~10.1 | ~2.5 | **$1.51** |
| HoVer | 2,426 | 2,426 | ~809 | ~3.6 | ~0.8 | **$0.50** |
| IFBench | 3,593 | 3,593 | ~1,198 | ~5.3 | ~1.2 | **$0.75** |
| PUPA | 3,936 | 3,936 | ~1,312 | ~5.8 | ~1.3 | **$0.82** |
| AIME | 7,051 | 7,051 | ~2,350 | ~10.3 | ~2.6 | **$1.56** |
| LiveBench | 1,839 | 1,839 | ~613 | ~2.7 | ~0.6 | **$0.38** |

**Sweep cost totals**:

| Scope | Runs | API cost | EC2 cost | **Total** |
|-------|------|----------|----------|-----------|
| Tier 1 (HotpotQA) | 12 | $18.12 | $0.49 | **$18.61** |
| Tier 1+2 | 24 | $24.12 | $0.68 | **$24.80** |
| Full primary sweep | 72 | $63.72 | $1.75 | **$65.47** |

### Cost reduction strategies

1. **Tiered execution** (already planned): Run Tier 1 first. If no signal, stop. Saves $46.86 if Tier 1 shows no improvement.

2. **Share vanilla control runs**: The `vanilla_control` config is identical across all snippet_length experiments. Run it once per (benchmark, seed) and reuse results. Saves 25% of runs (1 in 4 configs).
   - Tier 1+2: 24 runs becomes 18 unique runs = **$6.20 API savings**.
   - Full sweep: 72 runs becomes 54 unique runs.

3. **Early stopping**: GEPA already stops when the score plateaus for 20 consecutive iterations. On average, runs terminate 10-15% below budget, saving ~$3-5 across the sweep.

4. **OpenRouter caching**: OpenRouter caches identical prompt completions. Across 3 seeds, some evaluation calls will hit the same (prompt, example) pair and return cached responses at reduced cost. Estimated 5-10% savings on evaluation calls.

5. **Off-peak scheduling**: Run during US nighttime hours (UTC 06:00-14:00) when spot prices are lowest and OpenRouter has lower load.

**Optimized Tier 1+2 cost estimate**: ~$18-20 (after shared controls + early stopping + caching).

---

## 5. Reliability

### Spot interruption handling

**Signal**: EC2 sends a 2-minute interruption warning via instance metadata (`http://169.254.169.254/latest/meta-data/spot/instance-action`).

**Handler script** (`scripts/spot_interrupt_handler.sh`):
```bash
#!/bin/bash
# Poll for spot interruption notice every 5 seconds
while true; do
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
        http://169.254.169.254/latest/meta-data/spot/instance-action)
    if [ "$RESPONSE" -eq 200 ]; then
        echo "SPOT INTERRUPTION DETECTED - initiating emergency checkpoint"
        # Signal the Python process to checkpoint immediately
        kill -SIGUSR1 $(cat /tmp/gepa_experiment.pid)
        # Wait for checkpoint upload (max 90 seconds)
        sleep 90
        # Upload any remaining logs
        aws s3 sync /tmp/experiment_logs/ \
            s3://gepa-mutations-results/runs/${BENCHMARK}/${METHOD}/${CONFIG_ID}/${SEED}/logs/
        exit 0
    fi
    sleep 5
done
```

**Python-side SIGUSR1 handler** (in the experiment runner):
```python
import signal

def _handle_spot_interrupt(signum, frame):
    """Emergency checkpoint on spot interruption warning."""
    logger.warning("SPOT INTERRUPTION - saving emergency checkpoint")
    save_checkpoint(state, contrastive_train_index, iteration, to_s3=True)
    logger.warning("Emergency checkpoint saved to S3")

signal.signal(signal.SIGUSR1, _handle_spot_interrupt)
```

### Checkpoint resume logic

```
START
  |
  v
Check S3 for latest_checkpoint.json
  |
  +--[Found]--> Download checkpoint + state files
  |               |
  |               v
  |             Validate checkpoint integrity (hash check)
  |               |
  |               +--[Valid]--> Resume from checkpoint iteration + 1
  |               |
  |               +--[Invalid]--> Log warning, start from scratch
  |
  +--[Not found]--> Start from scratch
  |
  v
Run experiment loop
  |
  +-- Every 50 iterations: save checkpoint to S3
  |
  +-- On SIGUSR1: emergency checkpoint to S3
  |
  +-- On completion: upload final results, delete checkpoints, self-terminate
```

### Self-termination

Every instance self-terminates after its run completes (or fails). This prevents orphan instances from accumulating charges.

```bash
# At the end of the experiment launch script
echo "Experiment complete. Self-terminating in 60 seconds..."
sleep 60
# Upload final status
aws s3 cp /tmp/final_status.json \
    s3://gepa-mutations-results/runs/${BENCHMARK}/${METHOD}/${CONFIG_ID}/${SEED}/final_status.json

# Self-terminate
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

**Safety net**: The CloudWatch alarm (from `scripts/aws_setup.py`) fires if any tagged instance runs longer than 36 hours, sending an SNS alert.

### Failure modes and mitigations

| Failure mode | Detection | Mitigation |
|-------------|-----------|------------|
| Spot interruption mid-run | Metadata polling (5s interval) | Emergency checkpoint + resume on new instance |
| OpenRouter API outage | LiteLLM raises `ServiceUnavailableError` | Exponential backoff (Section 6), checkpoint every 50 iters |
| OpenRouter rate limit | HTTP 429 response | Backoff + reduce parallelism |
| Instance crash (OOM) | No heartbeat in CloudWatch for 10 min | CloudWatch alarm, restart on new instance |
| Corrupted checkpoint | SHA-256 hash mismatch on download | Fall back to previous checkpoint or restart |
| S3 upload failure | boto3 exception during checkpoint | Retry 3x with backoff, keep local copy |
| Experiment code bug | Python exception | Catch at top level, upload logs + traceback to S3, notify, self-terminate |

---

## 6. Model API Configuration

### OpenRouter setup

```python
# Environment variables (loaded from SSM Parameter Store on instance boot)
OPENROUTER_API_KEY = ssm.get_parameter("/gepa-mutations/openrouter-api-key")

# LiteLLM configuration (used by existing LM wrapper in runner/experiment.py)
LITELLM_CONFIG = {
    "model": "openrouter/qwen/qwen3-8b",
    "api_base": "https://openrouter.ai/api/v1",
    "api_key": OPENROUTER_API_KEY,
    "drop_params": True,  # Drop unsupported params silently
}

# Model parameters (paper defaults)
MODEL_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 16384,
}
```

### Rate limiting strategy

OpenRouter Qwen3-8B limits:
- **RPM**: 200 requests/minute (free tier), higher on paid plans
- **TPM**: Varies, typically 200K+ tokens/minute

**Client-side rate limiting**:
```python
# Configure LiteLLM's built-in rate limiter
import litellm

litellm.set_verbose = False  # Reduce log noise on EC2

# Custom rate limiter: 25 RPM per instance (safe for 6 parallel instances = 150 RPM total)
RATE_LIMIT_RPM = 25
```

The existing `LM` class in `runner/experiment.py` already passes `num_retries=3` to `litellm.completion()`. We extend this with:
- Per-instance RPM cap of 25 (configurable via env var `GEPA_RATE_LIMIT_RPM`)
- Inter-request delay: `60 / RATE_LIMIT_RPM = 2.4 seconds` minimum between calls (applied as a floor, not a sleep -- most calls take longer than this due to generation time)

### Retry and error handling

```python
# Retry configuration for LiteLLM
RETRY_CONFIG = {
    "num_retries": 5,              # Up from default 3 for production runs
    "retry_after": 10,             # Wait 10 seconds before first retry
    "max_retry_wait": 120,         # Cap exponential backoff at 2 minutes
    "retry_on_status_codes": [
        429,  # Rate limit
        500,  # Internal server error
        502,  # Bad gateway
        503,  # Service unavailable
        504,  # Gateway timeout
    ],
}
```

**Escalation policy for persistent API failures**:
1. Retries 1-3: Exponential backoff (10s, 20s, 40s)
2. Retries 4-5: Extended backoff (80s, 120s)
3. After 5 retries: Save checkpoint to S3, send Telegram alert, sleep 10 minutes, then retry
4. After 3 sleep-retry cycles (30 minutes of failures): Checkpoint, notify, self-terminate with `exit_code=2` (retriable failure)

The orchestrator script (Section 8) can detect `exit_code=2` and re-launch the instance after a cooldown period.

### Request/response logging

Every LLM call is logged to a local JSONL file (`/tmp/experiment_logs/llm_calls.jsonl`) with:
- Timestamp
- Call type (evaluation vs. reflection)
- Token counts (input, output)
- Latency (ms)
- Response status (success, retry, final failure)

This file is uploaded to S3 on completion for cost reconciliation and debugging.

---

## 7. Monitoring

### CloudWatch metrics

Publish custom metrics to the `gepa-mutations` CloudWatch namespace:

| Metric | Unit | Frequency | Purpose |
|--------|------|-----------|---------|
| `ExperimentIteration` | Count | Per iteration | Track progress; detect stalls |
| `BestValScore` | None (0.0-1.0) | Per iteration | Track convergence curve |
| `RolloutCount` | Count | Per iteration | Track budget consumption |
| `ContrastiveInjectionCount` | Count | Per iteration | Track how often contrastive signal fires |
| `APICallLatencyMs` | Milliseconds | Per LLM call | Detect API degradation |
| `APIErrorCount` | Count | Per minute | Detect API outages |
| `CheckpointAge` | Seconds | Per minute | Detect stalled experiments |

**Dimensions** for all metrics: `Benchmark`, `ConfigId`, `Seed`, `InstanceId`.

```python
import boto3

cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")

def publish_iteration_metrics(benchmark, config_id, seed, iteration, val_score, rollouts, contrastive_count):
    cloudwatch.put_metric_data(
        Namespace="gepa-mutations",
        MetricData=[
            {
                "MetricName": "ExperimentIteration",
                "Value": iteration,
                "Unit": "Count",
                "Dimensions": [
                    {"Name": "Benchmark", "Value": benchmark},
                    {"Name": "ConfigId", "Value": config_id},
                    {"Name": "Seed", "Value": str(seed)},
                ],
            },
            {
                "MetricName": "BestValScore",
                "Value": val_score,
                "Unit": "None",
                "Dimensions": [
                    {"Name": "Benchmark", "Value": benchmark},
                    {"Name": "ConfigId", "Value": config_id},
                    {"Name": "Seed", "Value": str(seed)},
                ],
            },
            {
                "MetricName": "ContrastiveInjectionCount",
                "Value": contrastive_count,
                "Unit": "Count",
                "Dimensions": [
                    {"Name": "Benchmark", "Value": benchmark},
                    {"Name": "ConfigId", "Value": config_id},
                    {"Name": "Seed", "Value": str(seed)},
                ],
            },
        ],
    )
```

### CloudWatch alarms

| Alarm | Condition | Action |
|-------|-----------|--------|
| `ExperimentStalled` | No `ExperimentIteration` metric for 30 minutes | SNS -> Telegram alert |
| `HighAPIErrorRate` | `APIErrorCount` > 50 in 5 minutes | SNS -> Telegram alert |
| `OrphanInstance` | Instance running > 36 hours (from `aws_setup.py`) | SNS -> Telegram alert |
| `BudgetExceeded` | AWS Cost Explorer daily spend > $30 | SNS -> email alert |

### SNS/Telegram notifications

Uses the existing `Notifier` class from `src/gepa_mutations/notifications/notifier.py`. Extend with contrastive-specific events:

| Event | Channel | Content |
|-------|---------|---------|
| Experiment started | Telegram | Benchmark, config_id, seed, instance_id |
| Checkpoint saved | Telegram (every 5th checkpoint only) | Iteration, val_score, rollout count |
| Experiment completed | Telegram | Final test score, wall clock, API cost estimate |
| Experiment failed | Telegram + SNS | Error message, instance_id, last checkpoint |
| Spot interruption | Telegram | Instance_id, last checkpoint iteration |
| Sweep batch completed | Telegram | Summary: N/M runs done, mean scores by config |
| All Tier 1 complete | Telegram + SNS | Full Tier 1 results table, go/no-go recommendation |

**Telegram message format** (using existing `Notifier.send_telegram()`):
```
*GEPA Contrastive Reflection - Run Complete*
Benchmark: `hotpotqa` | Config: `cr_snippet300` | Seed: `42`
Test score: `64.17%` (baseline: `62.33%`, delta: `+1.84`)
Rollouts: `6871` | Wall clock: `2h 47m`
Contrastive injections: `1847/2290` iterations (80.7%)
Est. API cost: `$1.48`
```

### Dashboard

Create a CloudWatch dashboard `gepa-contrastive-sweep` with:
1. **Score convergence panel**: Line chart of `BestValScore` over `ExperimentIteration`, one line per config_id, for the currently running benchmark.
2. **Rollout budget panel**: Bar chart of `RolloutCount` per run, with paper budget as a reference line.
3. **API health panel**: `APICallLatencyMs` p50/p99 and `APIErrorCount` time series.
4. **Active instances panel**: Count of running instances with GEPA tags.
5. **Contrastive activity panel**: `ContrastiveInjectionCount` by config (should be 0 for vanilla, >0 for cr_* configs).

---

## 8. Runbook

### Prerequisites

1. AWS infrastructure is provisioned (run `scripts/aws_setup.py` if not already done).
2. SSM parameters are populated with real API keys:
   - `/gepa-mutations/openrouter-api-key`
   - `/gepa-mutations/hf-token`
   - `/gepa-mutations/telegram-bot-token`
   - `/gepa-mutations/telegram-chat-id`
3. Local AWS credentials have permissions to launch EC2, read SSM, write S3.
4. The `contrastive_reflection` experiment code is implemented and passes all unit/smoke tests.

### Step 1: Validate locally before deploying

```bash
# Run unit tests
cd /Users/ashwinchidambaram/dev/projects/gepa-mutations
uv run pytest tests/experiments/contrastive_reflection/ -v

# Run smoke test (3 iterations on HotpotQA, local)
uv run python -m gepa_mutations.experiments.contrastive_reflection.run \
    --benchmark hotpotqa \
    --config cr_snippet300 \
    --seed 42 \
    --max-iterations 3 \
    --dry-run
```

### Step 2: Build deployment package

```bash
# Create a deployment archive with the code and dependencies
DEPLOY_DIR="/tmp/gepa-deploy-$(date +%Y%m%d-%H%M%S)"
mkdir -p $DEPLOY_DIR

# Export locked dependencies
uv export --format requirements-txt > $DEPLOY_DIR/requirements.txt

# Copy source code (exclude .venv, .git, data/)
rsync -av --exclude='.venv' --exclude='.git' --exclude='data/' \
    --exclude='runs/' --exclude='notebooks/' --exclude='__pycache__' \
    . $DEPLOY_DIR/gepa-mutations/

# Upload to S3
aws s3 sync $DEPLOY_DIR s3://gepa-mutations-results/deploy/latest/
echo "Deployment package uploaded to s3://gepa-mutations-results/deploy/latest/"
```

### Step 3: Launch a single test run

```bash
# Launch one spot instance for a single run to validate the pipeline
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.medium \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --iam-instance-profile Name=gepa-mutations-ec2-profile \
    --security-group-ids $(aws ec2 describe-security-groups --group-names gepa-mutations-sg --query 'SecurityGroups[0].GroupId' --output text) \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=gepa-mutations},{Key=Experiment,Value=contrastive_reflection},{Key=Benchmark,Value=hotpotqa},{Key=ConfigId,Value=cr_snippet300},{Key=Seed,Value=42}]" \
    --user-data file://scripts/ec2_userdata.sh \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
    --query 'Instances[0].InstanceId' \
    --output text
```

**User data script** (`scripts/ec2_userdata.sh`):
```bash
#!/bin/bash
set -euo pipefail

# Configuration (passed via instance tags, read from metadata)
REGION="us-east-1"
PROJECT="gepa-mutations"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

# Read experiment config from instance tags
BENCHMARK=$(aws ec2 describe-tags --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=Benchmark" --query 'Tags[0].Value' --output text --region $REGION)
CONFIG_ID=$(aws ec2 describe-tags --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=ConfigId" --query 'Tags[0].Value' --output text --region $REGION)
SEED=$(aws ec2 describe-tags --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=Seed" --query 'Tags[0].Value' --output text --region $REGION)

echo "Starting experiment: benchmark=$BENCHMARK config=$CONFIG_ID seed=$SEED"

# Install dependencies
dnf install -y python3.12 python3.12-pip git
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Download deployment package
aws s3 sync s3://gepa-mutations-results/deploy/latest/ /opt/gepa-deploy/
cd /opt/gepa-deploy/gepa-mutations

# Install Python dependencies
uv venv --python 3.12
uv pip install -r /opt/gepa-deploy/requirements.txt
uv pip install -e .

# Load secrets from SSM Parameter Store
export OPENROUTER_API_KEY=$(aws ssm get-parameter --name "/$PROJECT/openrouter-api-key" --with-decryption --query 'Parameter.Value' --output text --region $REGION)
export HF_TOKEN=$(aws ssm get-parameter --name "/$PROJECT/hf-token" --with-decryption --query 'Parameter.Value' --output text --region $REGION)
export TELEGRAM_BOT_TOKEN=$(aws ssm get-parameter --name "/$PROJECT/telegram-bot-token" --with-decryption --query 'Parameter.Value' --output text --region $REGION)
export TELEGRAM_CHAT_ID=$(aws ssm get-parameter --name "/$PROJECT/telegram-chat-id" --with-decryption --query 'Parameter.Value' --output text --region $REGION)
export S3_BUCKET="gepa-mutations-results"

# Start spot interruption handler in background
bash scripts/spot_interrupt_handler.sh &
HANDLER_PID=$!

# Run the experiment
uv run python -m gepa_mutations.experiments.contrastive_reflection.run \
    --benchmark "$BENCHMARK" \
    --config "$CONFIG_ID" \
    --seed "$SEED" \
    --s3-checkpoint \
    --notify-telegram \
    2>&1 | tee /tmp/experiment.log

EXIT_CODE=${PIPESTATUS[0]}

# Upload logs
aws s3 cp /tmp/experiment.log \
    "s3://gepa-mutations-results/runs/$BENCHMARK/contrastive_reflection/$CONFIG_ID/$SEED/logs/experiment.log"

# Kill interrupt handler
kill $HANDLER_PID 2>/dev/null || true

# Self-terminate
echo "Experiment finished with exit code $EXIT_CODE. Self-terminating..."
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
```

### Step 4: Launch the full Tier 1 sweep

```bash
# Launch all 12 Tier 1 runs (4 configs x 3 seeds on HotpotQA)
# First 6 in parallel, then next 6

BENCHMARK="hotpotqa"
CONFIGS=("vanilla_control" "cr_snippet150" "cr_snippet300" "cr_snippet500")
SEEDS=(42 137 2025)
SG_ID=$(aws ec2 describe-security-groups --group-names gepa-mutations-sg \
    --query 'SecurityGroups[0].GroupId' --output text)

for CONFIG in "${CONFIGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Launching: $BENCHMARK / $CONFIG / seed=$SEED"
        aws ec2 run-instances \
            --image-id ami-0c02fb55956c7d316 \
            --instance-type t3.medium \
            --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
            --iam-instance-profile Name=gepa-mutations-ec2-profile \
            --security-group-ids $SG_ID \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=gepa-mutations},{Key=Experiment,Value=contrastive_reflection},{Key=Benchmark,Value=$BENCHMARK},{Key=ConfigId,Value=$CONFIG},{Key=Seed,Value=$SEED}]" \
            --user-data file://scripts/ec2_userdata.sh \
            --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
            --output text --query 'Instances[0].InstanceId'

        # Stagger launches by 30 seconds to avoid API spike
        sleep 30
    done
done
```

### Step 5: Monitor running experiments

```bash
# List all running experiment instances
aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=gepa-mutations" \
              "Name=tag:Experiment,Values=contrastive_reflection" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].{ID:InstanceId,Type:InstanceType,State:State.Name,Benchmark:Tags[?Key==`Benchmark`]|[0].Value,Config:Tags[?Key==`ConfigId`]|[0].Value,Seed:Tags[?Key==`Seed`]|[0].Value,LaunchTime:LaunchTime}' \
    --output table

# Check latest checkpoint for a specific run
aws s3 cp s3://gepa-mutations-results/runs/hotpotqa/contrastive_reflection/cr_snippet300/42/checkpoints/latest_checkpoint.json - | python3 -m json.tool

# View CloudWatch dashboard
echo "https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=gepa-contrastive-sweep"

# Tail CloudWatch logs for a specific run (if CloudWatch Logs agent is configured)
aws logs tail /gepa-mutations/experiments \
    --filter-pattern "benchmark=hotpotqa config=cr_snippet300 seed=42" \
    --follow
```

### Step 6: Collect results after sweep completion

```bash
# Download all Tier 1 results
mkdir -p results/contrastive_reflection/tier1
aws s3 sync s3://gepa-mutations-results/runs/hotpotqa/contrastive_reflection/ \
    results/contrastive_reflection/tier1/hotpotqa/ \
    --exclude "*/checkpoints/*" \
    --exclude "*/logs/*"

# Verify all 12 runs completed
echo "Expected: 12 result.json files"
find results/contrastive_reflection/tier1/ -name "result.json" | wc -l

# Quick results summary
uv run python -c "
import json
from pathlib import Path

results_dir = Path('results/contrastive_reflection/tier1/hotpotqa')
rows = []
for result_file in sorted(results_dir.rglob('result.json')):
    data = json.loads(result_file.read_text())
    config_id = result_file.parent.parent.name
    rows.append({
        'config': config_id,
        'seed': data['seed'],
        'test_score': data['test_score'],
        'rollouts': data['rollout_count'],
        'wall_clock_min': data['wall_clock_seconds'] / 60,
    })

# Print summary table
print(f\"{'Config':<20} {'Seed':<6} {'Test Score':>10} {'Rollouts':>8} {'Time (min)':>10}\")
print('-' * 60)
for r in rows:
    print(f\"{r['config']:<20} {r['seed']:<6} {r['test_score']:>10.4f} {r['rollouts']:>8} {r['wall_clock_min']:>10.1f}\")
"
```

### Step 7: Analyze and decide on Tier 2

```bash
# Run the analysis notebook / script to compare configs
uv run python -m gepa_mutations.analysis.compare_configs \
    --results-dir results/contrastive_reflection/tier1/ \
    --baseline-config vanilla_control \
    --output results/contrastive_reflection/tier1_analysis.json

# Decision criteria (from mutation_plan.md):
# - If cr_snippet300 >= 62.83 on HotpotQA (mean across 3 seeds): proceed to Tier 2
# - If cr_snippet300 < 62.83: stop, analyze why, consider secondary sweep params
```

### Step 8: Emergency procedures

**Kill all running experiments**:
```bash
# List instance IDs
INSTANCE_IDS=$(aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=gepa-mutations" \
              "Name=tag:Experiment,Values=contrastive_reflection" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].InstanceId' \
    --output text)

# Terminate all
if [ -n "$INSTANCE_IDS" ]; then
    aws ec2 terminate-instances --instance-ids $INSTANCE_IDS
    echo "Terminated: $INSTANCE_IDS"
else
    echo "No running instances found"
fi
```

**Re-launch a failed run** (resumes from checkpoint):
```bash
BENCHMARK="hotpotqa"
CONFIG="cr_snippet300"
SEED="42"
SG_ID=$(aws ec2 describe-security-groups --group-names gepa-mutations-sg \
    --query 'SecurityGroups[0].GroupId' --output text)

aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.medium \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --iam-instance-profile Name=gepa-mutations-ec2-profile \
    --security-group-ids $SG_ID \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=gepa-mutations},{Key=Experiment,Value=contrastive_reflection},{Key=Benchmark,Value=$BENCHMARK},{Key=ConfigId,Value=$CONFIG},{Key=Seed,Value=$SEED}]" \
    --user-data file://scripts/ec2_userdata.sh \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
    --output text --query 'Instances[0].InstanceId'

# The userdata script automatically checks for existing checkpoints and resumes
```

**Check for orphan instances** (instances that did not self-terminate):
```bash
aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=gepa-mutations" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].{ID:InstanceId,LaunchTime:LaunchTime}' \
    --output table

# Any instance running > 6 hours for HoVer or > 8 hours for HotpotQA is likely orphaned
```

**Estimate current API spend**:
```bash
# Count total LLM calls from uploaded logs
aws s3 ls s3://gepa-mutations-results/runs/ --recursive \
    | grep "llm_calls.jsonl" \
    | while read line; do
        FILE=$(echo $line | awk '{print $4}')
        aws s3 cp "s3://gepa-mutations-results/$FILE" - | wc -l
    done | paste -sd+ | bc

# Or check OpenRouter usage dashboard directly:
echo "https://openrouter.ai/settings/keys"
```

---

## Cost Summary

| Scenario | EC2 (spot) | OpenRouter API | S3 + misc | **Total** |
|----------|-----------|----------------|-----------|-----------|
| Tier 1 only (12 runs, HotpotQA) | $0.49 | $18.12 | $0.10 | **$18.71** |
| Tier 1+2 (24 runs, HotpotQA + HoVer) | $0.68 | $24.12 | $0.15 | **$24.95** |
| Optimized Tier 1+2 (shared controls) | $0.53 | $18.60 | $0.12 | **$19.25** |
| Full primary sweep (72 runs, all 6 benchmarks) | $1.75 | $63.72 | $0.40 | **$65.87** |

The dominant cost is OpenRouter API usage, not compute. EC2 is <3% of total cost. The entire Tier 1 sweep costs less than $20.
