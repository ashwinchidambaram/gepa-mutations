# AWS Deployment Plan: 90 GEPA Mutation Experiments

**Date:** 2026-03-20
**Status:** Ready for execution
**Estimated total cost:** ~$24 EC2 + ~$38 API = ~$62 total

---

## Table of Contents

1. [Experiment Matrix Summary](#1-experiment-matrix-summary)
2. [Infrastructure Design](#2-infrastructure-design)
3. [Experiment Orchestration](#3-experiment-orchestration)
4. [Logging and Monitoring](#4-logging-and-monitoring)
5. [Cost Estimate](#5-cost-estimate)
6. [Resumability](#6-resumability)
7. [Results Collection](#7-results-collection)
8. [Deployment Scripts](#8-deployment-scripts)
9. [Pre-Flight Checklist](#9-pre-flight-checklist)
10. [Runbook: Common Issues](#10-runbook-common-issues)

---

## 1. Experiment Matrix Summary

| Phase | Method                  | K values | Benchmarks              | Seeds                   | Runs | Dependency     |
|-------|-------------------------|----------|-------------------------|-------------------------|------|----------------|
| 1     | best_of_k (validate)    | 1, 3     | hotpotqa                | 42                      | 2    | None           |
| 2     | best_of_k               | 1, 3, 5  | hotpotqa, pupa, aime    | 42,123,456,789,1024     | 45   | Phase 1 passes |
| 3     | contrastive_reflection  | --       | hotpotqa, pupa, aime    | 42,123,456,789,1024     | 15   | Phase 1 passes (parallel with Phase 2) |
| 4     | failure_stratified_k    | 3, 5     | hotpotqa, pupa, aime    | 42,123,456,789,1024     | 30   | Phase 2 done   |
| **Total** |                     |          |                         |                         | **92** |              |

### Rollout budgets per benchmark (from PAPER_ROLLOUTS):

| Benchmark | Rollout Budget | Estimated API Calls | Est. Duration |
|-----------|----------------|---------------------|---------------|
| hotpotqa  | 6,871          | ~5,500              | 2-4 hrs       |
| pupa      | 3,936          | ~3,100              | 2-3 hrs       |
| aime      | 7,051          | ~7,100              | 6-12 hrs      |

---

## 2. Infrastructure Design

### 2.1 Instance Type Selection

**Choice: `t3.medium` spot instances ($0.0125/hr on-demand, ~$0.0042/hr spot)**

Rationale:
- Workload is CPU/API-bound, not compute-heavy (waiting on OpenRouter API responses)
- 2 vCPU + 4 GB RAM is sufficient for Python + GEPA + dspy
- t3.medium has generous burst credit baseline (24 CPU credits/hr)
- Spot pricing in us-east-1 is consistently ~$0.004-0.005/hr (67% savings)
- t3 instances have very low spot interruption rates (<5% historically)

### 2.2 Spot vs On-Demand Strategy

| Phase | Strategy   | Reason |
|-------|------------|--------|
| 1     | On-demand  | Only 2 validation runs; fast turnaround needed for go/no-go |
| 2     | Spot       | 45 runs, cost-sensitive, checkpointing handles interruptions |
| 3     | Spot       | 15 runs, same reasoning as Phase 2 |
| 4     | Spot       | 30 runs, same reasoning as Phase 2 |

### 2.3 Parallelism Strategy

**Maximum 10 concurrent instances** (balanced for cost vs throughput)

Rationale:
- OpenRouter rate limits: 10 concurrent requests is safe for qwen3-8b at this pricing tier
- 10 instances x ~$0.005/hr = $0.05/hr EC2 spend during peak
- Completes Phase 2 (45 runs) in ~5 waves of ~8-10 instances
- AIME runs are 3x slower, so we schedule them to overlap with faster runs

Concurrency plan by phase:
- **Phase 1:** 2 instances (sequential OK, takes ~4 hrs)
- **Phase 2 + 3:** 10 instances (60 runs across ~6 waves, ~36-48 hrs)
- **Phase 4:** 10 instances (30 runs across ~3 waves, ~18-24 hrs)

Total elapsed wall time: ~3-4 days

### 2.4 S3 Bucket Structure

```
s3://gepa-mutations-results/
  runs/
    hotpotqa/
      best_of_k_K1/
        42/
          result.json
          config.json
          metrics.json
          gepa_state/        # GEPA checkpoint directory
        123/
        ...
      best_of_k_K3/
      best_of_k_K5/
      contrastive_reflection/
      failure_stratified_k_K3/
      failure_stratified_k_K5/
    pupa/
      ...
    aime/
      ...
  logs/
    <instance-id>/
      cloud-init-output.log
      experiment.log
      heartbeat.json       # Updated every 5 minutes
  status/
    manifest.json          # Full experiment matrix with status tracking
    phase2_complete.flag   # Sentinel file for Phase 4 dependency
```

### 2.5 Self-Termination

Every instance self-terminates via a bash trap on ALL exit paths:
- Experiment success (exit 0)
- Experiment failure (non-zero exit)
- Unhandled signal (SIGTERM, etc.)
- Spot interruption (2-minute warning handler)

The existing `launch_experiment.py` already has this pattern. We enhance it with:
- Upload of partial results before termination
- Upload of log files to S3 before termination
- Status update to S3 manifest

### 2.6 Network and Security

The existing `aws_setup.py` creates:
- Egress-only security group (no SSH needed)
- IAM role with S3 write, SNS publish, CloudWatch, SSM read
- SSM Parameter Store for API keys

**Addition needed:** IAM policy must also allow `ec2:TerminateInstances` on self (for self-termination). Update the inline policy:

```json
{
    "Effect": "Allow",
    "Action": "ec2:TerminateInstances",
    "Resource": "*",
    "Condition": {
        "StringEquals": {
            "ec2:ResourceTag/Project": "gepa-mutations"
        }
    }
}
```

---

## 3. Experiment Orchestration

### 3.1 Architecture Overview

```
Local machine (orchestrator)
  |
  |--> scripts/orchestrate_experiments.py
  |      |
  |      |--> Reads manifest.json (experiment matrix + status)
  |      |--> Launches EC2 spot instances via boto3
  |      |--> Polls S3 for completion flags
  |      |--> Triggers next phase when dependencies met
  |
  |--> Each EC2 instance:
         |
         |--> user-data bootstrap script
         |      |--> Install uv, clone repo
         |      |--> Fetch secrets from SSM
         |      |--> Run single experiment
         |      |--> Upload results + logs to S3
         |      |--> Self-terminate
```

### 3.2 Manifest-Driven Job Queue

Instead of a message queue (SQS), we use a simple JSON manifest on S3. The orchestrator script reads it, finds pending jobs, and launches instances.

**`manifest.json` structure:**

```json
{
    "experiments": [
        {
            "id": "best_of_k_K1__hotpotqa__42",
            "phase": 1,
            "method": "best_of_k",
            "k": 1,
            "benchmark": "hotpotqa",
            "seed": 42,
            "status": "pending",
            "instance_id": null,
            "started_at": null,
            "completed_at": null,
            "test_score": null,
            "error": null
        }
    ],
    "phase_status": {
        "1": "pending",
        "2": "pending",
        "3": "pending",
        "4": "pending"
    }
}
```

Status values: `pending` -> `launched` -> `running` -> `completed` | `failed`

### 3.3 Config Management

Each instance receives its parameters through **instance tags + user-data arguments**. The user-data script reads tags to determine what to run:

```bash
# Inside user-data script
METHOD=$(curl -s http://169.254.169.254/latest/meta-data/tags/instance/Method)
BENCHMARK=$(curl -s http://169.254.169.254/latest/meta-data/tags/instance/Benchmark)
SEED=$(curl -s http://169.254.169.254/latest/meta-data/tags/instance/Seed)
K_VALUE=$(curl -s http://169.254.169.254/latest/meta-data/tags/instance/KValue)
```

Alternatively (and more reliably), parameters are baked into the user-data script at launch time. The existing `launch_experiment.py` already does this via f-string interpolation. We extend this pattern.

### 3.4 Phase 2 -> 4 Dependency Handling

The orchestrator script implements phase gating:

```python
def can_start_phase(phase: int, manifest: dict) -> bool:
    if phase <= 1:
        return True
    if phase in (2, 3):
        # Phase 2 and 3 require Phase 1 to pass
        phase1_exps = [e for e in manifest["experiments"] if e["phase"] == 1]
        return all(e["status"] == "completed" for e in phase1_exps)
    if phase == 4:
        # Phase 4 requires Phase 2 to complete (not Phase 3)
        phase2_exps = [e for e in manifest["experiments"] if e["phase"] == 2]
        return all(e["status"] in ("completed", "failed") for e in phase2_exps)
    return False
```

The orchestrator runs as a local cron job (every 15 minutes) or a manual polling loop:

```bash
# Run orchestrator in a loop
while true; do
    python scripts/orchestrate_experiments.py --poll
    sleep 900  # 15 minutes
done
```

### 3.5 Detailed User-Data Script

The enhanced user-data script for each instance (template filled at launch time):

```bash
#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/gepa-experiment.log) 2>&1

# ============================================================
# Parameters (injected at launch time)
# ============================================================
EXPERIMENT_ID="{experiment_id}"
METHOD="{method}"
BENCHMARK="{benchmark}"
SEED={seed}
K_VALUE={k_value}
BRANCH="{branch}"
REPO_URL="{repo_url}"
REGION="us-east-1"
PROJECT="gepa-mutations"
S3_BUCKET="gepa-mutations-results"

# ============================================================
# Self-termination trap (runs on ANY exit)
# ============================================================
INSTANCE_ID=$(TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600") && \
    curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id)

cleanup() {
    EXIT_CODE=$?
    echo "=== CLEANUP: exit code $EXIT_CODE at $(date -u) ==="

    # Upload experiment logs to S3
    aws s3 cp /var/log/gepa-experiment.log \
        "s3://$S3_BUCKET/logs/$INSTANCE_ID/experiment.log" || true

    # Upload any partial results
    if [ -d /home/ubuntu/gepa-mutations/runs ]; then
        cd /home/ubuntu/gepa-mutations
        aws s3 sync runs/ "s3://$S3_BUCKET/runs/" --quiet || true
    fi

    # Update manifest status
    if [ $EXIT_CODE -eq 0 ]; then
        STATUS="completed"
    else
        STATUS="failed"
    fi

    python3 -c "
import json, boto3, datetime
s3 = boto3.client('s3', region_name='$REGION')
try:
    obj = s3.get_object(Bucket='$S3_BUCKET', Key='status/manifest.json')
    manifest = json.loads(obj['Body'].read())
    for exp in manifest['experiments']:
        if exp['id'] == '$EXPERIMENT_ID':
            exp['status'] = '$STATUS'
            exp['completed_at'] = datetime.datetime.utcnow().isoformat()
            if '$STATUS' == 'failed':
                exp['error'] = 'Exit code $EXIT_CODE'
            break
    s3.put_object(Bucket='$S3_BUCKET', Key='status/manifest.json',
                  Body=json.dumps(manifest, indent=2))
except Exception as e:
    echo 'Failed to update manifest: {e}'
" 2>/dev/null || true

    # Self-terminate
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" || true
}
trap cleanup EXIT

# ============================================================
# Heartbeat: background process posts status every 5 min
# ============================================================
(
    while true; do
        HEARTBEAT=$(python3 -c "
import json, datetime, os
hb = {
    'instance_id': '$INSTANCE_ID',
    'experiment_id': '$EXPERIMENT_ID',
    'timestamp': datetime.datetime.utcnow().isoformat(),
    'uptime_seconds': float(open('/proc/uptime').read().split()[0]),
}
# Check if experiment process is still running
import subprocess
result = subprocess.run(['pgrep', '-f', 'gepa-mutations\\|best_of_k\\|contrastive_reflection\\|failure_stratified_k'],
                       capture_output=True)
hb['experiment_alive'] = result.returncode == 0

# Check run directory for progress
for d in ['runs']:
    run_path = f'/home/ubuntu/gepa-mutations/{d}'
    if os.path.exists(run_path):
        for root, dirs, files in os.walk(run_path):
            for f in files:
                if f == 'metrics.json':
                    try:
                        import json as j
                        with open(os.path.join(root, f)) as fh:
                            m = j.load(fh)
                            hb['iterations'] = m.get('total_iterations', 0)
                            hb['metric_calls'] = m.get('total_metric_calls', 0)
                            hb['acceptance_rate'] = m.get('acceptance_rate', 0)
                    except: pass
print(json.dumps(hb))
" 2>/dev/null)
        echo "$HEARTBEAT" | aws s3 cp - \
            "s3://$S3_BUCKET/logs/$INSTANCE_ID/heartbeat.json" --quiet 2>/dev/null || true
        sleep 300
    done
) &
HEARTBEAT_PID=$!

# ============================================================
# Spot interruption monitor
# ============================================================
(
    TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
    while true; do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            -H "X-aws-ec2-metadata-token: $TOKEN" \
            http://169.254.169.254/latest/meta-data/spot/instance-action)
        if [ "$HTTP_CODE" = "200" ]; then
            echo "SPOT INTERRUPTION DETECTED at $(date -u)"
            # Sync checkpoint immediately
            if [ -d /home/ubuntu/gepa-mutations/runs ]; then
                cd /home/ubuntu/gepa-mutations
                aws s3 sync runs/ "s3://$S3_BUCKET/runs/" --quiet || true
            fi
            echo "Checkpoint synced. Waiting for termination..."
            sleep 120
        fi
        sleep 5
    done
) &

# ============================================================
# Install dependencies
# ============================================================
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl python3-pip > /dev/null 2>&1

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# ============================================================
# Clone repo and setup
# ============================================================
cd /home/ubuntu
git clone --branch "$BRANCH" "$REPO_URL" gepa-mutations
cd gepa-mutations

# Initialize GEPA submodule
if [ ! -d "gepa/src" ]; then
    git submodule update --init --recursive 2>/dev/null || \
    git clone --branch v0.1.1 https://github.com/gepa-ai/gepa.git gepa
fi
rm -rf gepa/.venv gepa/uv.lock

# ============================================================
# Fetch secrets from SSM Parameter Store
# ============================================================
export OPENROUTER_API_KEY=$(aws ssm get-parameter \
    --name "/$PROJECT/openrouter-api-key" \
    --with-decryption --query "Parameter.Value" --output text --region "$REGION")
export HF_TOKEN=$(aws ssm get-parameter \
    --name "/$PROJECT/hf-token" \
    --with-decryption --query "Parameter.Value" --output text --region "$REGION")
export TELEGRAM_BOT_TOKEN=$(aws ssm get-parameter \
    --name "/$PROJECT/telegram-bot-token" \
    --with-decryption --query "Parameter.Value" --output text --region "$REGION" 2>/dev/null || echo "")
export TELEGRAM_CHAT_ID=$(aws ssm get-parameter \
    --name "/$PROJECT/telegram-chat-id" \
    --with-decryption --query "Parameter.Value" --output text --region "$REGION" 2>/dev/null || echo "")

# Validate critical secrets
if [ -z "$OPENROUTER_API_KEY" ] || [ "$OPENROUTER_API_KEY" = "REPLACE_WITH_YOUR_KEY" ]; then
    echo "FATAL: OpenRouter API key not configured in SSM"
    exit 1
fi

# ============================================================
# Install Python dependencies
# ============================================================
uv sync

# ============================================================
# Check for existing checkpoint (resume after spot interruption)
# ============================================================
RUN_DIR="runs/$BENCHMARK/$METHOD/$SEED"
aws s3 sync "s3://$S3_BUCKET/$RUN_DIR/" "$RUN_DIR/" --quiet 2>/dev/null || true
if [ -d "$RUN_DIR/gepa_state" ]; then
    echo "Found checkpoint in S3, resuming from existing state"
fi

# ============================================================
# Update manifest status to "running"
# ============================================================
python3 -c "
import json, boto3, datetime
s3 = boto3.client('s3', region_name='$REGION')
try:
    obj = s3.get_object(Bucket='$S3_BUCKET', Key='status/manifest.json')
    manifest = json.loads(obj['Body'].read())
    for exp in manifest['experiments']:
        if exp['id'] == '$EXPERIMENT_ID':
            exp['status'] = 'running'
            exp['instance_id'] = '$INSTANCE_ID'
            exp['started_at'] = datetime.datetime.utcnow().isoformat()
            break
    s3.put_object(Bucket='$S3_BUCKET', Key='status/manifest.json',
                  Body=json.dumps(manifest, indent=2))
except Exception as e:
    print(f'Warning: could not update manifest: {e}')
" 2>/dev/null || true

# ============================================================
# Run experiment (method-specific dispatch)
# ============================================================
case "$METHOD" in
    best_of_k_K*)
        # Extract K from method name (e.g., best_of_k_K3 -> 3)
        K=$(echo "$METHOD" | sed 's/best_of_k_K//')
        cd gepa-best-of-k
        uv run python -m best_of_k.sweep \
            --benchmarks "$BENCHMARK" \
            --k-values "$K" \
            --seeds "$SEED" \
            2>&1 | tee -a /var/log/gepa-experiment.log
        cd ..
        ;;
    contrastive_reflection)
        cd gepa-contrastive-reflection
        uv run python -m contrastive_reflection.runner \
            --benchmark "$BENCHMARK" \
            --seed "$SEED" \
            2>&1 | tee -a /var/log/gepa-experiment.log
        cd ..
        ;;
    failure_stratified_k_K*)
        K=$(echo "$METHOD" | sed 's/failure_stratified_k_K//')
        cd gepa-failure-stratified-k
        uv run python -c "
from failure_stratified_k.runner import run_failure_stratified_k
from gepa_mutations.base import MutationConfig

config = MutationConfig(
    mutation_name='failure_stratified_k_K$K',
    description='Failure-stratified K with K=$K',
    benchmark='$BENCHMARK',
    seed=$SEED,
    mutation_candidates=$K,
    use_failure_stratified_k=True,
)
run_failure_stratified_k(config=config, k=$K)
" 2>&1 | tee -a /var/log/gepa-experiment.log
        cd ..
        ;;
    *)
        echo "ERROR: Unknown method: $METHOD"
        exit 1
        ;;
esac

# ============================================================
# Upload final results to S3
# ============================================================
aws s3 sync runs/ "s3://$S3_BUCKET/runs/" --quiet

# Upload final log
aws s3 cp /var/log/gepa-experiment.log \
    "s3://$S3_BUCKET/logs/$INSTANCE_ID/experiment.log" || true

echo "=== Experiment $EXPERIMENT_ID completed successfully at $(date -u) ==="
```

---

## 4. Logging and Monitoring

### 4.1 Log Destinations

| Log Type | Destination | Retention |
|----------|-------------|-----------|
| Experiment stdout/stderr | S3: `logs/<instance-id>/experiment.log` | 180 days (lifecycle policy) |
| Heartbeat JSON | S3: `logs/<instance-id>/heartbeat.json` | Updated every 5 min, overwritten |
| GEPA run log | S3: `runs/<bm>/<method>/<seed>/gepa_state/run_log.txt` | With results |
| CloudWatch custom metrics | `GEPA/Experiments` namespace | 30 days |
| Manifest status | S3: `status/manifest.json` | Permanent |

### 4.2 Health Checks

The heartbeat file (updated every 5 minutes) contains:

```json
{
    "instance_id": "i-0abc123...",
    "experiment_id": "best_of_k_K3__hotpotqa__42",
    "timestamp": "2026-03-20T15:30:00Z",
    "uptime_seconds": 3600,
    "experiment_alive": true,
    "iterations": 12,
    "metric_calls": 1843,
    "acceptance_rate": 0.42
}
```

**Health check script** (run locally to check all running experiments):

```python
# scripts/check_health.py
import boto3, json, datetime

s3 = boto3.client("s3")
BUCKET = "gepa-mutations-results"
STALE_MINUTES = 15

# List all heartbeat files
response = s3.list_objects_v2(Bucket=BUCKET, Prefix="logs/", Delimiter="/")
for prefix in response.get("CommonPrefixes", []):
    instance_dir = prefix["Prefix"]
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"{instance_dir}heartbeat.json")
        hb = json.loads(obj["Body"].read())
        age = (datetime.datetime.utcnow() -
               datetime.datetime.fromisoformat(hb["timestamp"])).total_seconds() / 60

        status = "OK" if age < STALE_MINUTES and hb.get("experiment_alive") else "STALE"
        print(f"  {hb['experiment_id']:45s} | {status:6s} | "
              f"iter={hb.get('iterations', '?'):>4s} | "
              f"calls={hb.get('metric_calls', '?'):>5s} | "
              f"age={age:.0f}m")
    except Exception:
        pass
```

### 4.3 Alerting Rules

| Trigger | Detection | Action |
|---------|-----------|--------|
| Instance running > 18 hrs | CloudWatch alarm (existing 36hr, tightened to 18hr) | SNS -> email/Telegram |
| Heartbeat stale > 30 min | Health check script | Manual investigation |
| API error rate > 50% | Parse heartbeat acceptance_rate drop | SNS alert |
| Spot interruption | Instance metadata check, logged to S3 | Auto-resume via orchestrator |
| Phase completion | Orchestrator polls manifest | Launches next phase |
| Total spend > $45 EC2 | AWS Budgets alarm | SNS alert, pause launches |
| All experiments complete | Orchestrator detects all "completed" | Telegram summary |

### 4.4 Stuck Process Detection

An experiment is considered "stuck" if:
1. Heartbeat shows `experiment_alive: false` (process crashed but instance still running)
2. Heartbeat `metric_calls` unchanged for 3 consecutive checks (15 min)
3. Instance uptime exceeds 2x the expected duration for its benchmark

The orchestrator marks stuck experiments as `failed` and can auto-relaunch.

---

## 5. Cost Estimate

### 5.1 EC2 Cost Per Experiment

| Benchmark | Duration (hrs) | Spot Rate ($/hr) | EC2 Cost |
|-----------|----------------|-------------------|----------|
| hotpotqa  | 3              | $0.0042           | $0.013   |
| pupa      | 2.5            | $0.0042           | $0.011   |
| aime      | 9              | $0.0042           | $0.038   |

### 5.2 API Cost Per Experiment

Qwen3-8B via OpenRouter: $0.05/M input tokens, $0.40/M output tokens.

Estimated per-call token usage: ~800 input + ~400 output tokens (average across reflection and task calls).

| Benchmark | API Calls | Est. Input Tokens | Est. Output Tokens | API Cost |
|-----------|-----------|-------------------|--------------------|----------|
| hotpotqa  | ~5,500    | ~4.4M             | ~2.2M              | $1.10    |
| pupa      | ~3,100    | ~2.5M             | ~1.2M              | $0.61    |
| aime      | ~7,100    | ~5.7M             | ~2.8M              | $1.41    |

### 5.3 Total Cost Breakdown

**Phase 1 (2 runs):**
| Item | Runs | Per-Run | Total |
|------|------|---------|-------|
| EC2 (hotpotqa, on-demand) | 2 | $0.038 | $0.08 |
| API (hotpotqa) | 2 | $1.10 | $2.20 |
| **Phase 1 subtotal** | | | **$2.28** |

**Phase 2 (45 runs):**
| Item | Runs | Per-Run | Total |
|------|------|---------|-------|
| EC2 (15 hotpotqa) | 15 | $0.013 | $0.19 |
| EC2 (15 pupa) | 15 | $0.011 | $0.16 |
| EC2 (15 aime) | 15 | $0.038 | $0.57 |
| API (15 hotpotqa) | 15 | $1.10 | $16.50 |
| API (15 pupa) | 15 | $0.61 | $9.15 |
| API (15 aime) | 15 | $1.41 | $21.15 |
| **Phase 2 subtotal** | | | **$47.72** (EC2: $0.92, API: $46.80) |

**Phase 3 (15 runs):**
| Item | Runs | Per-Run | Total |
|------|------|---------|-------|
| EC2 (5 hotpotqa) | 5 | $0.013 | $0.07 |
| EC2 (5 pupa) | 5 | $0.011 | $0.06 |
| EC2 (5 aime) | 5 | $0.038 | $0.19 |
| API (5 hotpotqa) | 5 | $1.10 | $5.50 |
| API (5 pupa) | 5 | $0.61 | $3.05 |
| API (5 aime) | 5 | $1.41 | $7.05 |
| **Phase 3 subtotal** | | | **$15.92** (EC2: $0.32, API: $15.60) |

**Phase 4 (30 runs):**
| Item | Runs | Per-Run | Total |
|------|------|---------|-------|
| EC2 (10 hotpotqa) | 10 | $0.013 | $0.13 |
| EC2 (10 pupa) | 10 | $0.011 | $0.11 |
| EC2 (10 aime) | 10 | $0.038 | $0.38 |
| API (10 hotpotqa) | 10 | $1.10 | $11.00 |
| API (10 pupa) | 10 | $0.61 | $6.10 |
| API (10 aime) | 10 | $1.41 | $14.10 |
| **Phase 4 subtotal** | | | **$31.82** (EC2: $0.62, API: $31.20) |

### 5.4 Grand Total

| Category | Cost |
|----------|------|
| EC2 (spot + on-demand) | **$1.94** |
| OpenRouter API | **$95.80** |
| S3 storage (~50 GB) | ~$1.15/month |
| SNS notifications | ~$0.01 |
| **Grand total** | **~$99** |

> Note: The EC2 cost is well under the $50 constraint. The dominant cost is API calls.
> If API budget is a concern, consider reducing to 3 seeds (42, 123, 456) instead
> of 5, which cuts API cost to ~$57.

### 5.5 Cost Controls

```python
# Add to scripts/aws_setup.py

def create_budget_alarm():
    """Create AWS Budgets alarm at $50 total spend."""
    budgets = boto3.client("budgets")
    account_id = boto3.client("sts").get_caller_identity()["Account"]

    budgets.create_budget(
        AccountId=account_id,
        Budget={
            "BudgetName": "gepa-mutations-ec2-budget",
            "BudgetLimit": {"Amount": "50", "Unit": "USD"},
            "TimeUnit": "MONTHLY",
            "BudgetType": "COST",
            "CostFilters": {
                "TagKeyValue": ["user:Project$gepa-mutations"]
            },
        },
        NotificationsWithSubscribers=[{
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 80.0,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": [{
                "SubscriptionType": "SNS",
                "Address": "<SNS_TOPIC_ARN>",
            }],
        }],
    )
```

---

## 6. Resumability

### 6.1 GEPA's Built-in Checkpointing

GEPA saves its state to `run_dir/` after each iteration. The state includes:
- All candidate prompts and their scores
- Pareto front composition
- Evaluation cache
- RNG state
- Iteration counter and budget counter

When `GEPAEngine.run()` is called with a `run_dir` that already contains state, it automatically resumes from the last checkpoint.

### 6.2 S3 Checkpoint Strategy

**Periodic sync (not continuous):**
- Checkpoints are synced to S3 every 5 minutes by the heartbeat process (lightweight, only changed files)
- On spot interruption (2-min warning), immediate full sync
- On experiment completion, final sync

**Resume flow:**
1. Orchestrator detects a `failed` or `interrupted` experiment in manifest
2. Orchestrator launches a new instance with the same parameters
3. New instance's user-data script runs `aws s3 sync` to pull existing checkpoint
4. GEPA detects checkpoint in `run_dir/` and resumes

```bash
# In user-data (already included in Section 3.5):
RUN_DIR="runs/$BENCHMARK/$METHOD/$SEED"
aws s3 sync "s3://$S3_BUCKET/$RUN_DIR/" "$RUN_DIR/" --quiet 2>/dev/null || true
```

### 6.3 Enhanced Heartbeat with Checkpoint Sync

```bash
# Inside heartbeat loop (every 5 minutes)
# Sync checkpoint to S3 (only changed files)
if [ -d /home/ubuntu/gepa-mutations/runs ]; then
    cd /home/ubuntu/gepa-mutations
    aws s3 sync runs/ "s3://$S3_BUCKET/runs/" \
        --exclude "*.pyc" --exclude "__pycache__/*" --quiet || true
fi
```

### 6.4 Idempotency

The system is fully idempotent:
- Relaunching a completed experiment: the orchestrator checks `result.json` exists in S3 and skips it
- Relaunching a partially completed experiment: GEPA resumes from checkpoint, no duplicate work
- Duplicate launches: the orchestrator checks if an instance with the same experiment_id is already running (via manifest status)

---

## 7. Results Collection

### 7.1 Results Flow

```
EC2 Instance
  |-- runs/<bm>/<method>/<seed>/result.json
  |-- runs/<bm>/<method>/<seed>/config.json
  |-- runs/<bm>/<method>/<seed>/metrics.json
  |-- runs/<bm>/<method>/<seed>/gepa_state/
  |
  |--(aws s3 sync)-->  s3://gepa-mutations-results/runs/...
                            |
                            |--(download to local)
                            |
                        Local machine: runs/...
                            |
                            |--> gepa-mutations compare
                            |--> notebooks/analysis.ipynb
```

### 7.2 Batch Download Script

```python
# scripts/download_all_results.py
"""Download all experiment results from S3 to local runs/ directory."""

import boto3
from pathlib import Path

BUCKET = "gepa-mutations-results"
LOCAL_DIR = Path("runs")

def download_all():
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET, Prefix="runs/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            local_path = LOCAL_DIR / key[len("runs/"):]

            # Skip gepa_state directories (large, only needed for resume)
            if "gepa_state/" in key:
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists():
                s3.download_file(BUCKET, key, str(local_path))
                print(f"  Downloaded: {key}")

    print("Done. Run 'gepa-mutations compare' to see results.")

if __name__ == "__main__":
    download_all()
```

### 7.3 Data Validation

After downloading, validate that each result.json has the required fields:

```python
# scripts/validate_results.py
"""Validate all downloaded result.json files."""

import json
from pathlib import Path

REQUIRED_FIELDS = [
    "benchmark", "method", "seed", "test_score", "val_score",
    "best_prompt", "rollout_count", "wall_clock_seconds",
    "test_example_scores", "test_example_ids",
]

def validate():
    runs_dir = Path("runs")
    errors = []

    for result_file in runs_dir.rglob("result.json"):
        with open(result_file) as f:
            data = json.load(f)

        missing = [field for field in REQUIRED_FIELDS if field not in data]
        if missing:
            errors.append((str(result_file), f"Missing fields: {missing}"))

        # Validate test_score is reasonable
        score = data.get("test_score", -1)
        if not (0 <= score <= 1):
            errors.append((str(result_file), f"Invalid test_score: {score}"))

        # Validate per-example scores length matches test set
        example_scores = data.get("test_example_scores", [])
        if len(example_scores) == 0:
            errors.append((str(result_file), "Empty test_example_scores"))

    if errors:
        print(f"VALIDATION FAILED: {len(errors)} issues found")
        for path, error in errors:
            print(f"  {path}: {error}")
    else:
        count = len(list(runs_dir.rglob("result.json")))
        print(f"All {count} result files passed validation.")

if __name__ == "__main__":
    validate()
```

### 7.4 Aggregation for Analysis

The existing `gepa-mutations compare` CLI command and `src/gepa_mutations/analysis/` module handle aggregation. After downloading all results, the standard workflow is:

```bash
# Compare all best_of_k results across K values
for method in best_of_k_K1 best_of_k_K3 best_of_k_K5; do
    gepa-mutations compare --method $method
done

# Or use the analysis module directly from notebooks
```

---

## 8. Deployment Scripts

### 8.1 Enhanced aws_setup.py Additions

The existing `scripts/aws_setup.py` handles S3, SNS, IAM, security groups, SSM, and CloudWatch.

**Required additions:**
1. EC2 self-terminate permission in IAM policy
2. Budget alarm
3. Instance metadata tag access (needed for user-data to read its own tags)

### 8.2 Orchestrator Script

This is the main new script. Create as `scripts/orchestrate_experiments.py`:

```python
#!/usr/bin/env python3
"""Orchestrate GEPA experiments across EC2 spot instances.

Usage:
    # Generate manifest (one-time)
    python scripts/orchestrate_experiments.py --generate-manifest

    # Launch next batch of experiments
    python scripts/orchestrate_experiments.py --launch

    # Check status of all experiments
    python scripts/orchestrate_experiments.py --status

    # Poll loop (run in background)
    python scripts/orchestrate_experiments.py --poll

    # Auto-run: generate manifest, then poll until complete
    python scripts/orchestrate_experiments.py --auto
"""

import argparse
import json
import time
import datetime
import sys
from itertools import product

import boto3
from botocore.exceptions import ClientError

# =========================================================================
# Configuration
# =========================================================================
REGION = "us-east-1"
PROJECT = "gepa-mutations"
BUCKET = "gepa-mutations-results"
INSTANCE_TYPE = "t3.medium"
AMI_ID = "ami-0c7217cdde317cfec"  # Ubuntu 22.04 LTS us-east-1
INSTANCE_PROFILE = "gepa-mutations-ec2-profile"
SG_NAME = "gepa-mutations-sg"
MAX_CONCURRENT = 10
REPO_URL = "https://github.com/YOUR_ORG/gepa-mutations.git"  # UPDATE THIS
BRANCH = "master"

SEEDS = [42, 123, 456, 789, 1024]
BENCHMARKS = ["hotpotqa", "pupa", "aime"]

# =========================================================================
# Manifest Generation
# =========================================================================

def generate_manifest() -> dict:
    """Generate the full experiment manifest."""
    experiments = []

    # Phase 1: Validation (2 runs)
    for k in [1, 3]:
        experiments.append({
            "id": f"best_of_k_K{k}__hotpotqa__42",
            "phase": 1,
            "method": f"best_of_k_K{k}",
            "k": k,
            "benchmark": "hotpotqa",
            "seed": 42,
            "status": "pending",
            "instance_id": None,
            "started_at": None,
            "completed_at": None,
            "test_score": None,
            "error": None,
        })

    # Phase 2: best_of_k full sweep (45 runs)
    for k, bm, seed in product([1, 3, 5], BENCHMARKS, SEEDS):
        exp_id = f"best_of_k_K{k}__{bm}__{seed}"
        # Skip Phase 1 duplicates
        if exp_id in [e["id"] for e in experiments]:
            continue
        experiments.append({
            "id": exp_id,
            "phase": 2,
            "method": f"best_of_k_K{k}",
            "k": k,
            "benchmark": bm,
            "seed": seed,
            "status": "pending",
            "instance_id": None,
            "started_at": None,
            "completed_at": None,
            "test_score": None,
            "error": None,
        })

    # Phase 3: contrastive_reflection (15 runs)
    for bm, seed in product(BENCHMARKS, SEEDS):
        experiments.append({
            "id": f"contrastive_reflection__{bm}__{seed}",
            "phase": 3,
            "method": "contrastive_reflection",
            "k": 0,
            "benchmark": bm,
            "seed": seed,
            "status": "pending",
            "instance_id": None,
            "started_at": None,
            "completed_at": None,
            "test_score": None,
            "error": None,
        })

    # Phase 4: failure_stratified_k (30 runs)
    for k, bm, seed in product([3, 5], BENCHMARKS, SEEDS):
        experiments.append({
            "id": f"failure_stratified_k_K{k}__{bm}__{seed}",
            "phase": 4,
            "method": f"failure_stratified_k_K{k}",
            "k": k,
            "benchmark": bm,
            "seed": seed,
            "status": "pending",
            "instance_id": None,
            "started_at": None,
            "completed_at": None,
            "test_score": None,
            "error": None,
        })

    manifest = {
        "created_at": datetime.datetime.utcnow().isoformat(),
        "total_experiments": len(experiments),
        "experiments": experiments,
        "phase_status": {
            "1": "pending",
            "2": "pending",
            "3": "pending",
            "4": "pending",
        },
    }

    print(f"Generated manifest with {len(experiments)} experiments")
    for phase in [1, 2, 3, 4]:
        count = sum(1 for e in experiments if e["phase"] == phase)
        print(f"  Phase {phase}: {count} experiments")

    return manifest


def upload_manifest(manifest: dict):
    """Upload manifest to S3."""
    s3 = boto3.client("s3", region_name=REGION)
    s3.put_object(
        Bucket=BUCKET,
        Key="status/manifest.json",
        Body=json.dumps(manifest, indent=2),
    )
    print(f"Manifest uploaded to s3://{BUCKET}/status/manifest.json")


def download_manifest() -> dict:
    """Download current manifest from S3."""
    s3 = boto3.client("s3", region_name=REGION)
    try:
        obj = s3.get_object(Bucket=BUCKET, Key="status/manifest.json")
        return json.loads(obj["Body"].read())
    except ClientError:
        print("ERROR: No manifest found. Run with --generate-manifest first.")
        sys.exit(1)


# =========================================================================
# Phase Gating
# =========================================================================

def can_start_phase(phase: int, manifest: dict) -> bool:
    """Check if a phase's dependencies are met."""
    experiments = manifest["experiments"]

    if phase == 1:
        return True

    if phase in (2, 3):
        # Require Phase 1 to complete successfully
        phase1 = [e for e in experiments if e["phase"] == 1]
        return all(e["status"] == "completed" for e in phase1)

    if phase == 4:
        # Require ALL Phase 2 experiments to finish (completed or failed)
        phase2 = [e for e in experiments if e["phase"] == 2]
        return all(e["status"] in ("completed", "failed") for e in phase2)

    return False


# =========================================================================
# Instance Launching
# =========================================================================

def count_running_instances() -> int:
    """Count currently running GEPA experiment instances."""
    ec2 = boto3.client("ec2", region_name=REGION)
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Project", "Values": [PROJECT]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ]
    )
    count = sum(
        len(r["Instances"])
        for r in response["Reservations"]
    )
    return count


def launch_experiment(exp: dict, use_spot: bool = True) -> str | None:
    """Launch a single experiment on an EC2 instance."""
    ec2 = boto3.client("ec2", region_name=REGION)

    # Get security group
    sgs = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [SG_NAME]}]
    )
    if not sgs["SecurityGroups"]:
        print(f"  ERROR: Security group not found. Run aws_setup.py first.")
        return None
    sg_id = sgs["SecurityGroups"][0]["GroupId"]

    # Build user-data (the full script from Section 3.5)
    user_data = _build_user_data(exp)

    launch_params = {
        "ImageId": AMI_ID,
        "InstanceType": INSTANCE_TYPE,
        "MinCount": 1,
        "MaxCount": 1,
        "IamInstanceProfile": {"Name": INSTANCE_PROFILE},
        "SecurityGroupIds": [sg_id],
        "UserData": user_data,
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": f"{PROJECT}-{exp['id']}"},
                {"Key": "Project", "Value": PROJECT},
                {"Key": "ExperimentId", "Value": exp["id"]},
                {"Key": "Phase", "Value": str(exp["phase"])},
                {"Key": "Method", "Value": exp["method"]},
                {"Key": "Benchmark", "Value": exp["benchmark"]},
                {"Key": "Seed", "Value": str(exp["seed"])},
            ],
        }],
        # Enable instance metadata tags (needed for user-data to read tags)
        "MetadataOptions": {
            "HttpTokens": "required",
            "InstanceMetadataTags": "enabled",
        },
    }

    if use_spot:
        launch_params["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }

    try:
        response = ec2.run_instances(**launch_params)
        instance_id = response["Instances"][0]["InstanceId"]
        print(f"  Launched {instance_id} for {exp['id']} "
              f"({'spot' if use_spot else 'on-demand'})")
        return instance_id
    except ClientError as e:
        print(f"  ERROR launching {exp['id']}: {e}")
        return None


def _build_user_data(exp: dict) -> str:
    """Build the user-data bootstrap script for an experiment."""
    # This is the full script from Section 3.5 with parameters filled in
    return f"""#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/gepa-experiment.log) 2>&1

EXPERIMENT_ID="{exp['id']}"
METHOD="{exp['method']}"
BENCHMARK="{exp['benchmark']}"
SEED={exp['seed']}
K_VALUE={exp.get('k', 0)}
BRANCH="{BRANCH}"
REPO_URL="{REPO_URL}"
REGION="{REGION}"
PROJECT="{PROJECT}"
S3_BUCKET="{BUCKET}"

# ... (full script from Section 3.5 goes here)
# See Section 3.5 for the complete user-data template
"""
    # NOTE: In the actual implementation, the full user-data script from
    # Section 3.5 is templated here. Kept abbreviated for readability.


# =========================================================================
# Main Orchestration Logic
# =========================================================================

def launch_batch(manifest: dict) -> dict:
    """Launch the next batch of eligible experiments."""
    running = count_running_instances()
    available_slots = MAX_CONCURRENT - running

    if available_slots <= 0:
        print(f"At capacity: {running} instances running (max {MAX_CONCURRENT})")
        return manifest

    print(f"Running: {running}, Available slots: {available_slots}")

    launched = 0
    for exp in manifest["experiments"]:
        if launched >= available_slots:
            break

        if exp["status"] != "pending":
            continue

        if not can_start_phase(exp["phase"], manifest):
            continue

        # Check if result already exists in S3 (idempotency)
        s3 = boto3.client("s3", region_name=REGION)
        result_key = f"runs/{exp['benchmark']}/{exp['method']}/{exp['seed']}/result.json"
        try:
            s3.head_object(Bucket=BUCKET, Key=result_key)
            print(f"  Skipping {exp['id']} (result already in S3)")
            exp["status"] = "completed"
            continue
        except ClientError:
            pass  # Not found, proceed

        # Launch
        use_spot = exp["phase"] != 1  # On-demand for Phase 1
        instance_id = launch_experiment(exp, use_spot=use_spot)

        if instance_id:
            exp["status"] = "launched"
            exp["instance_id"] = instance_id
            exp["started_at"] = datetime.datetime.utcnow().isoformat()
            launched += 1
            time.sleep(2)  # Avoid API throttling

    if launched > 0:
        upload_manifest(manifest)
        print(f"Launched {launched} experiments")
    else:
        print("No experiments to launch")

    return manifest


def print_status(manifest: dict):
    """Print a status summary of all experiments."""
    by_phase = {}
    for exp in manifest["experiments"]:
        phase = exp["phase"]
        by_phase.setdefault(phase, {"pending": 0, "launched": 0,
                                     "running": 0, "completed": 0, "failed": 0})
        status = exp["status"]
        by_phase[phase][status] = by_phase[phase].get(status, 0) + 1

    print("\n=== Experiment Status ===")
    for phase in sorted(by_phase.keys()):
        counts = by_phase[phase]
        total = sum(counts.values())
        gate = "OPEN" if can_start_phase(phase, manifest) else "BLOCKED"
        print(f"  Phase {phase} [{gate}]: "
              f"{counts['completed']}/{total} done, "
              f"{counts['running']+counts['launched']} active, "
              f"{counts['failed']} failed, "
              f"{counts['pending']} pending")

    # Overall
    total = len(manifest["experiments"])
    completed = sum(1 for e in manifest["experiments"] if e["status"] == "completed")
    failed = sum(1 for e in manifest["experiments"] if e["status"] == "failed")
    print(f"\n  Overall: {completed}/{total} completed, {failed} failed")


def poll_loop(manifest: dict, interval: int = 900):
    """Continuously poll and launch experiments until all complete."""
    while True:
        manifest = download_manifest()  # Refresh from S3
        print_status(manifest)

        # Check if all done
        all_done = all(
            e["status"] in ("completed", "failed")
            for e in manifest["experiments"]
        )
        if all_done:
            print("\nAll experiments complete!")
            break

        # Launch next batch
        manifest = launch_batch(manifest)

        print(f"\nSleeping {interval}s until next check...")
        time.sleep(interval)


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="GEPA experiment orchestrator")
    parser.add_argument("--generate-manifest", action="store_true",
                        help="Generate and upload experiment manifest")
    parser.add_argument("--launch", action="store_true",
                        help="Launch next batch of experiments")
    parser.add_argument("--status", action="store_true",
                        help="Print status of all experiments")
    parser.add_argument("--poll", action="store_true",
                        help="Run poll loop until all experiments complete")
    parser.add_argument("--auto", action="store_true",
                        help="Generate manifest then poll until complete")
    parser.add_argument("--interval", type=int, default=900,
                        help="Poll interval in seconds (default: 900)")

    args = parser.parse_args()

    if args.generate_manifest or args.auto:
        manifest = generate_manifest()
        upload_manifest(manifest)
        if not args.auto:
            return

    if args.auto or args.poll:
        manifest = download_manifest()
        poll_loop(manifest, interval=args.interval)
    elif args.launch:
        manifest = download_manifest()
        launch_batch(manifest)
    elif args.status:
        manifest = download_manifest()
        print_status(manifest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

### 8.3 Health Check Script

Create as `scripts/check_health.py`:

```python
#!/usr/bin/env python3
"""Check health of all running GEPA experiments."""

import boto3
import json
import datetime

BUCKET = "gepa-mutations-results"
REGION = "us-east-1"
STALE_THRESHOLD_MINUTES = 15

def check_health():
    s3 = boto3.client("s3", region_name=REGION)
    ec2 = boto3.client("ec2", region_name=REGION)

    # List running instances
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Project", "Values": ["gepa-mutations"]},
            {"Name": "instance-state-name", "Values": ["running"]},
        ]
    )

    instances = {}
    for res in response["Reservations"]:
        for inst in res["Instances"]:
            iid = inst["InstanceId"]
            tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
            instances[iid] = {
                "experiment_id": tags.get("ExperimentId", "unknown"),
                "launch_time": inst["LaunchTime"].isoformat(),
                "uptime_hours": (datetime.datetime.now(datetime.timezone.utc) -
                                inst["LaunchTime"]).total_seconds() / 3600,
            }

    if not instances:
        print("No running instances found.")
        return

    print(f"\n{'Instance':20s} {'Experiment':45s} {'Uptime':>8s} {'Iters':>6s} "
          f"{'Calls':>7s} {'Status':>8s}")
    print("-" * 100)

    for iid, info in sorted(instances.items(), key=lambda x: x[1]["experiment_id"]):
        # Try to get heartbeat
        try:
            obj = s3.get_object(Bucket=BUCKET,
                               Key=f"logs/{iid}/heartbeat.json")
            hb = json.loads(obj["Body"].read())
            age_min = (datetime.datetime.utcnow() -
                      datetime.datetime.fromisoformat(hb["timestamp"])).total_seconds() / 60
            alive = hb.get("experiment_alive", False)

            if not alive:
                status = "DEAD"
            elif age_min > STALE_THRESHOLD_MINUTES:
                status = "STALE"
            else:
                status = "OK"

            print(f"{iid:20s} {info['experiment_id']:45s} "
                  f"{info['uptime_hours']:7.1f}h "
                  f"{hb.get('iterations', '?'):>6} "
                  f"{hb.get('metric_calls', '?'):>7} "
                  f"{status:>8s}")
        except Exception:
            print(f"{iid:20s} {info['experiment_id']:45s} "
                  f"{info['uptime_hours']:7.1f}h "
                  f"{'?':>6} {'?':>7} {'NO HB':>8s}")


if __name__ == "__main__":
    check_health()
```

### 8.4 Download & Validate Scripts

(See Section 7.2 and 7.3 for `download_all_results.py` and `validate_results.py`)

### 8.5 Emergency Cleanup Script

```python
# scripts/emergency_cleanup.py
"""Terminate all GEPA experiment instances (emergency use only)."""

import boto3

REGION = "us-east-1"
PROJECT = "gepa-mutations"

def cleanup():
    ec2 = boto3.client("ec2", region_name=REGION)
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Project", "Values": [PROJECT]},
            {"Name": "instance-state-name",
             "Values": ["pending", "running", "stopping"]},
        ]
    )

    instance_ids = []
    for res in response["Reservations"]:
        for inst in res["Instances"]:
            instance_ids.append(inst["InstanceId"])

    if not instance_ids:
        print("No running instances found.")
        return

    print(f"Terminating {len(instance_ids)} instances:")
    for iid in instance_ids:
        print(f"  {iid}")

    confirm = input("Type 'yes' to confirm: ")
    if confirm != "yes":
        print("Aborted.")
        return

    ec2.terminate_instances(InstanceIds=instance_ids)
    print("Termination initiated.")


if __name__ == "__main__":
    cleanup()
```

---

## 9. Pre-Flight Checklist

Run these steps **in order** before launching experiments:

```bash
# 1. Run AWS setup (idempotent)
python scripts/aws_setup.py

# 2. Store real API keys in SSM (if not already done)
aws ssm put-parameter \
    --name "/gepa-mutations/openrouter-api-key" \
    --value "sk-or-v1-YOUR_KEY" \
    --type SecureString \
    --overwrite \
    --region us-east-1

aws ssm put-parameter \
    --name "/gepa-mutations/hf-token" \
    --value "hf_YOUR_TOKEN" \
    --type SecureString \
    --overwrite \
    --region us-east-1

# 3. (Optional) Configure Telegram notifications
aws ssm put-parameter \
    --name "/gepa-mutations/telegram-bot-token" \
    --value "YOUR_BOT_TOKEN" \
    --type SecureString \
    --overwrite \
    --region us-east-1

aws ssm put-parameter \
    --name "/gepa-mutations/telegram-chat-id" \
    --value "YOUR_CHAT_ID" \
    --type SecureString \
    --overwrite \
    --region us-east-1

# 4. Verify SSM parameters are set
aws ssm get-parameters \
    --names "/gepa-mutations/openrouter-api-key" "/gepa-mutations/hf-token" \
    --with-decryption \
    --query "Parameters[].{Name:Name,Value:Value}" \
    --region us-east-1

# 5. Update REPO_URL in scripts/orchestrate_experiments.py
#    and scripts/launch_experiment.py

# 6. Push latest code to the repo (instances will clone it)
git push origin master

# 7. Verify AMI exists
aws ec2 describe-images \
    --image-ids ami-0c7217cdde317cfec \
    --region us-east-1 \
    --query "Images[0].{Name:Name,State:State}"

# 8. Test with a dry run (Phase 1 only)
python scripts/orchestrate_experiments.py --generate-manifest
python scripts/orchestrate_experiments.py --status

# 9. Launch Phase 1 (validation)
python scripts/orchestrate_experiments.py --launch

# 10. Monitor Phase 1
python scripts/check_health.py

# 11. After Phase 1 completes successfully, start the poll loop
python scripts/orchestrate_experiments.py --poll
```

---

## 10. Runbook: Common Issues

### Spot Interruption

**Symptoms:** Instance disappears, manifest shows "launched" but instance is terminated.
**Fix:** The orchestrator detects stale "launched" entries (no heartbeat for >30 min) and resets them to "pending" for relaunch. GEPA resumes from the S3 checkpoint.

```python
# In orchestrator, add to poll_loop:
for exp in manifest["experiments"]:
    if exp["status"] == "launched" and exp.get("started_at"):
        started = datetime.datetime.fromisoformat(exp["started_at"])
        age = (datetime.datetime.utcnow() - started).total_seconds() / 3600
        if age > 0.5:  # 30 min with no status update
            # Check if instance still exists
            try:
                ec2.describe_instances(InstanceIds=[exp["instance_id"]])
            except:
                exp["status"] = "pending"
                exp["instance_id"] = None
```

### API Rate Limiting

**Symptoms:** Heartbeat shows `acceptance_rate: 0` and `metric_calls` not increasing.
**Fix:** GEPA's built-in retry logic (via LiteLLM `num_retries=3`) handles transient rate limits. If persistent, reduce `MAX_CONCURRENT` to 6-8 instances.

### Experiment Stuck (No Progress)

**Symptoms:** `check_health.py` shows STALE status, iterations unchanged.
**Fix:**
1. Check the experiment log: `aws s3 cp s3://gepa-mutations-results/logs/<instance-id>/experiment.log -`
2. If stuck on a specific iteration, terminate the instance manually
3. Orchestrator will relaunch from checkpoint

### Out of Budget

**Symptoms:** AWS Budgets alarm fires.
**Fix:** Run `python scripts/emergency_cleanup.py` to terminate all instances. Review spend before resuming.

### Corrupted Checkpoint

**Symptoms:** Experiment fails immediately after resuming from checkpoint.
**Fix:** Delete the checkpoint in S3 and relaunch fresh:
```bash
aws s3 rm s3://gepa-mutations-results/runs/<bm>/<method>/<seed>/gepa_state/ --recursive
```

### Phase 1 Validation Fails

**Symptoms:** Phase 1 experiments produce nonsensical scores.
**Fix:** Do not proceed to Phase 2. Debug locally first:
```bash
cd gepa-best-of-k
uv run python -m best_of_k.sweep --benchmarks hotpotqa --k-values 1 --seeds 42 --subset 10
```

---

## Architecture Diagram

```
                                    +-----------------+
                                    |   SSM Params    |
                                    |  (API keys)     |
                                    +--------+--------+
                                             |
+-------------------+              +--------+--------+
|   Local Machine   |              |   EC2 Spot      |
|                   |   launch     |   Instance      |
|  orchestrate.py   +------------->+                 |
|  check_health.py  |              | user-data:      |
|  download.py      |              |  1. clone repo  |
|  validate.py      |              |  2. get secrets |
|                   |   poll S3    |  3. uv sync     |
|                   +<-------------+  4. run exper.  |
+-------------------+              |  5. upload S3   |
                                   |  6. self-term   |
                                   +--------+--------+
                                            |
                              +-------------+-------------+
                              |                           |
                    +---------v---------+     +-----------v---------+
                    |   S3 Bucket       |     |   OpenRouter API    |
                    |                   |     |   (qwen3-8b)        |
                    | /runs/            |     +-----------+---------+
                    | /logs/            |
                    | /status/manifest  |
                    +-------------------+
                              |
                    +---------v---------+
                    |   SNS Topic       |
                    |   (alerts)        |
                    +-------------------+
```

---

## Timeline

| Day | Activity |
|-----|----------|
| Day 0 | Run pre-flight checklist, launch Phase 1 (2 validation runs) |
| Day 0 (evening) | Verify Phase 1 results, start orchestrator poll loop |
| Day 1-2 | Phase 2 + 3 running (60 experiments, 10 concurrent) |
| Day 2-3 | Phase 2 completes, Phase 4 auto-starts |
| Day 3-4 | Phase 4 completes (30 experiments) |
| Day 4 | Download results, validate, begin analysis |

Total wall time: **~4 days** (can be reduced to ~2.5 days with MAX_CONCURRENT=15).
