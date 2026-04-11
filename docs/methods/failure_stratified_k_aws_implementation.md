# AWS Implementation: failure_stratified_k

**Mutation:** Failure-Stratified K -- partition failing examples across K mutation candidates
**Date:** 2026-03-19
**Dependency:** best_of_k must complete its sweep before failure_stratified_k new runs begin. The `stratification=False` control conditions are reused directly from best_of_k results.

---

## 1. EC2 Instance Selection

### Instance type: `t3.medium` (shared with best_of_k)

| Property | Value | Rationale |
|----------|-------|-----------|
| vCPUs | 2 | Sufficient for single-threaded GEPA optimization loop. The bottleneck is API latency (OpenRouter round-trips), not local compute. |
| RAM | 4 GB | Python process + dataset loading peaks at ~1.5 GB. 4 GB provides headroom for OS, logging, and checkpoint serialization. |
| Network | Up to 5 Gbps | More than enough for API calls (~10 KB/request). |
| Storage | 20 GB gp3 EBS | GEPA state files, checkpoints, and result JSONs. Each run produces ~5-15 MB. Full sweep fits in < 2 GB. |
| Architecture | x86_64 (Intel) | Broadest AMI availability. ARM (t4g) saves ~20% but complicates the Python 3.12 + uv toolchain. |

### Spot vs On-Demand: Spot with checkpoint recovery

**Use Spot instances.** This workload is API-bound, not latency-sensitive. A 2-minute interruption costs nothing except the time to restart from the last checkpoint.

| Factor | Spot | On-Demand | Decision |
|--------|------|-----------|----------|
| Price (us-east-1) | ~$0.0104/hr | ~$0.0416/hr | Spot is 75% cheaper |
| Interruption risk | 2-min warning via instance metadata | None | Acceptable with checkpointing (Section 5) |
| Total sweep savings | ~$4.50 saved over full sweep | Baseline | Spot wins |

**Spot configuration:**

```json
{
  "InstanceMarketOptions": {
    "MarketType": "spot",
    "SpotOptions": {
      "SpotInstanceType": "persistent",
      "InstanceInterruptionBehavior": "stop"
    }
  }
}
```

Setting `InstanceInterruptionBehavior` to `"stop"` (not `"terminate"`) preserves the EBS volume across interruptions. When capacity returns, the instance restarts and the sweep script resumes from the last checkpoint.

### Shared infrastructure with best_of_k

failure_stratified_k runs on the **same instance type and AMI** as best_of_k. If best_of_k is still running when failure_stratified_k is ready, they can share the same EC2 instance sequentially (best_of_k finishes, then failure_stratified_k starts). They should NOT run concurrently on the same instance because both saturate the OpenRouter rate limit.

If best_of_k has already completed and its instance was terminated, launch a new spot instance with the same AMI, pull the best_of_k results from S3, and proceed with the failure_stratified_k-only runs.

---

## 2. S3 Storage

### Bucket: `gepa-mutations-results` (existing, from `Settings.s3_bucket`)

### Key structure

The S3 key layout follows the existing convention in `storage/s3.py`:

```
s3://gepa-mutations-results/
  runs/
    {benchmark}/
      best_of_k/                         # best_of_k results (already uploaded)
        K{k}/
          {seed}/
            result.json
            config.json
            metrics.json
            gepa_state/                  # GEPA engine state for resume
              checkpoint.pkl
      failure_stratified_k/              # NEW: failure_stratified_k results
        K{k}/
          {seed}/
            result.json
            config.json
            metrics.json
            gepa_state/
              checkpoint.pkl
      gepa/                              # Phase 1 reproduction baselines
        {seed}/
          ...
```

### Relationship to best_of_k results

The `stratification=False` control conditions for failure_stratified_k are **identical** to the best_of_k runs at the same K and seed. These results live under `runs/{benchmark}/best_of_k/K{k}/{seed}/` and are referenced by the analysis scripts -- they are NOT duplicated under `failure_stratified_k/`.

The analysis pipeline joins results across these two prefixes:
- Treatment: `runs/{benchmark}/failure_stratified_k/K{k}/{seed}/result.json`
- Control: `runs/{benchmark}/best_of_k/K{k}/{seed}/result.json`

### Checkpointing for spot interruption recovery

Each run writes a checkpoint to local disk after every GEPA iteration. The checkpoint contains:

1. **GEPA engine state** (`gepa_state/checkpoint.pkl`): Frontier candidates, evaluation cache, iteration count, budget consumed. GEPA's `run_dir` parameter enables this natively via `optimize(run_dir=...)`.
2. **Sweep progress file** (`sweep_state.json`): Which (K, seed, benchmark) combinations have completed. Written atomically (write to temp file, then rename) to survive crashes.
3. **Per-run metrics** (`metrics.json`): Incremental metrics from `MetricsCallback`, including `stratification_applied` and `partition_sizes` per iteration.

**S3 sync cadence:** Every 10 minutes (via cron or background thread), sync the local `runs/` directory to S3:

```bash
aws s3 sync runs/ s3://gepa-mutations-results/runs/ \
  --exclude "*.pyc" --exclude "__pycache__/*"
```

This ensures at most 10 minutes of work is lost on spot interruption. Given each iteration takes 30-120 seconds (dominated by K API calls), this means losing at most 5-20 iterations -- easily recoverable from the GEPA checkpoint.

### S3 lifecycle policy

No lifecycle policy needed. Total storage for the full sweep is < 500 MB. Glacier transition would complicate result retrieval for negligible savings.

---

## 3. Parameter Sweep Execution

### Sweep matrix

From the mutation plan, the full sweep is:

| Parameter | Values | Count |
|-----------|--------|-------|
| `mutation_candidates` (K) | [3, 5, 7] | 3 |
| `use_failure_stratified_k` | [True, False] | 2 |
| Benchmarks (initial) | [hotpotqa, hover, ifbench, aime] | 4 |
| Seeds | [42, 123, 456] | 3 |

**Total conditions:** 3 K x 2 stratification x 4 benchmarks x 3 seeds = **72 runs**

### Runs reusable from best_of_k

The `use_failure_stratified_k=False` conditions are identical to best_of_k runs at the same K. These 36 runs (3 K x 4 benchmarks x 3 seeds) are already executed as part of the best_of_k sweep:

| Reusable (from best_of_k) | New (failure_stratified_k only) |
|----------------------------|---------------------------------|
| K=3, stratified=False, all benchmarks, all seeds | K=3, stratified=True, all benchmarks, all seeds |
| K=5, stratified=False, all benchmarks, all seeds | K=5, stratified=True, all benchmarks, all seeds |
| K=7, stratified=False, all benchmarks, all seeds | K=7, stratified=True, all benchmarks, all seeds |
| **36 runs reused** | **36 new runs** |

**Net new runs for failure_stratified_k: 36**

### Per-run iteration estimates

Using the paper rollout budgets and the budget formula from the mutation plan (`iterations ~ budget / (3 + 3K)`):

| Benchmark | Budget | Iter @ K=3 | Iter @ K=5 | Iter @ K=7 |
|-----------|--------|------------|------------|------------|
| HotpotQA | 6,871 | ~572 | ~382 | ~286 |
| HoVer | 2,426 | ~202 | ~135 | ~101 |
| IFBench | 3,593 | ~299 | ~200 | ~150 |
| AIME | 7,051 | ~587 | ~392 | ~294 |

Note: These are upper-bound estimates. Accepted candidates trigger full validation set evaluations that consume additional budget, reducing actual iterations.

### Execution order

The sweep follows the mutation plan's priority ordering and dependency on best_of_k:

```
Phase 0: Verify best_of_k results are available in S3 (prerequisite check)
          Download best_of_k result.json files for all (K, benchmark, seed) combos
          Abort if any are missing -- best_of_k must complete first

Phase 1: High-signal benchmarks (HotpotQA, HoVer) -- K=3 first
          hotpotqa_fsk_K3_seed42  -> hotpotqa_fsk_K3_seed123 -> hotpotqa_fsk_K3_seed456
          hover_fsk_K3_seed42     -> hover_fsk_K3_seed123    -> hover_fsk_K3_seed456
          [CHECKPOINT: Compare fsk_K3 vs bok_K3 on HotpotQA and HoVer]
          [GO/NO-GO: If fsk_K3 <= bok_K3 on both, stop here. Report null result.]

Phase 2: Extend K values on high-signal benchmarks
          hotpotqa_fsk_K5 (3 seeds) -> hotpotqa_fsk_K7 (3 seeds)
          hover_fsk_K5 (3 seeds)    -> hover_fsk_K7 (3 seeds)

Phase 3: Remaining benchmarks (IFBench, AIME as negative control)
          ifbench_fsk_K3 (3 seeds) -> ifbench_fsk_K5 (3 seeds) -> ifbench_fsk_K7 (3 seeds)
          aime_fsk_K3 (3 seeds)    -> aime_fsk_K5 (3 seeds)    -> aime_fsk_K7 (3 seeds)

Phase 4 (conditional): PUPA, LiveBench -- only if Phases 1-3 show signal
          pupa_fsk_K3 (3 seeds)     -> livebench_fsk_K3 (3 seeds)
```

### Sweep script invocation

Each run maps to a single `runner.py` call:

```bash
python -m gepa_mutations.experiments.failure_stratified_k.runner \
  --benchmark hotpotqa \
  --seed 42 \
  --mutation-candidates 3 \
  --use-failure-stratified-k
```

The sweep script iterates over the matrix and skips completed runs (checked via `sweep_state.json`).

---

## 4. Cost Optimization

### OpenRouter API costs (primary cost driver)

Model: `openrouter/qwen/qwen3-8b`
Pricing: $0.05 per 1M input tokens, $0.40 per 1M output tokens

Each GEPA iteration involves:
- **1 capture_traces call** (task LM): ~500 input + ~200 output tokens per example x 3 examples = ~1,500 input / ~600 output tokens
- **K propose_new_texts calls** (reflection LM): ~2,000 input + ~500 output tokens each
- **K minibatch evaluations** (task LM): ~500 input + ~200 output tokens per example x 3 examples x K = ~1,500K input / ~600K output tokens
- **Validation set evaluation** (on acceptance, task LM): ~500 input + ~200 output tokens per example x |valset| examples

Per-iteration token estimate at K=3 (rejected iteration, no val eval):

| Component | Input tokens | Output tokens |
|-----------|-------------|---------------|
| Capture traces (3 examples) | 1,500 | 600 |
| Reflection (K=3) | 6,000 | 1,500 |
| K evaluations (3x3 examples) | 4,500 | 1,800 |
| **Total** | **12,000** | **3,900** |

Per-iteration API cost at K=3: `(12,000 * $0.05 + 3,900 * $0.40) / 1,000,000 = $0.0006 + $0.00156 = $0.00216`

### Per-run cost estimates

| Benchmark | Budget | Approx iterations (K=3) | API cost per run |
|-----------|--------|------------------------|------------------|
| HotpotQA | 6,871 | ~572 | ~$1.24 |
| HoVer | 2,426 | ~202 | ~$0.44 |
| IFBench | 3,593 | ~299 | ~$0.65 |
| AIME | 7,051 | ~587 | ~$1.27 |

Note: These estimates exclude validation set evaluations on acceptance (which add ~$0.01-$0.10 per acceptance depending on valset size) and test set evaluation at the end (~$0.05-$0.20 depending on test set size). Actual costs will be ~20-40% higher due to acceptances and merge operations.

### Total sweep cost (new runs only)

| Benchmark | Runs (K=3,5,7 x 3 seeds) | Avg cost/run | Subtotal |
|-----------|---------------------------|-------------|----------|
| HotpotQA | 9 | ~$1.50 | ~$13.50 |
| HoVer | 9 | ~$0.55 | ~$4.95 |
| IFBench | 9 | ~$0.80 | ~$7.20 |
| AIME | 9 | ~$1.55 | ~$13.95 |
| **Total API cost (36 new runs)** | | | **~$39.60** |

### EC2 compute cost

Each run takes approximately 2-8 hours wall clock (dominated by API latency, not compute). Estimate ~5 hours average per run.

| Component | Hours | Spot $/hr | Cost |
|-----------|-------|-----------|------|
| 36 runs x ~5 hrs avg | 180 hrs | $0.0104 | ~$1.87 |
| Overhead (setup, restarts) | ~20 hrs | $0.0104 | ~$0.21 |
| **Total EC2 cost** | **200 hrs** | | **~$2.08** |

### Grand total

| Component | Cost |
|-----------|------|
| OpenRouter API (36 new runs) | ~$39.60 |
| EC2 spot (t3.medium, ~200 hrs) | ~$2.08 |
| S3 storage (< 500 MB) | ~$0.01 |
| Data transfer | ~$0.10 |
| **Grand total** | **~$41.79** |
| **Savings from reusing best_of_k controls** | **~$39.60 saved** (36 runs not re-executed) |
| **Cost if running with early stopping** (Phases 1-2 only) | **~$18.45** (HotpotQA + HoVer, 18 runs) |

### Cost guardrails

1. **OpenRouter spending limit:** Set a daily API spend cap of $15 via the OpenRouter dashboard. This prevents runaway costs from infinite retry loops.
2. **Per-run timeout:** Each run self-terminates after 12 hours wall clock. No single run should take longer than 8 hours; 12 hours indicates a hang.
3. **Sweep budget cap:** The sweep script aborts if cumulative OpenRouter spend exceeds $50 (tracked via the OpenRouter `/api/v1/credits` endpoint between runs).

---

## 5. Reliability

### Spot interruption handling

The EC2 instance metadata service provides a 2-minute warning before spot termination. A background monitoring process polls for this warning and triggers a graceful shutdown:

```bash
# Interrupt monitor (runs as a background process)
while true; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    http://169.254.169.254/latest/meta-data/spot/instance-action)
  if [ "$HTTP_CODE" -eq 200 ]; then
    echo "SPOT INTERRUPTION DETECTED -- syncing to S3"
    aws s3 sync runs/ s3://gepa-mutations-results/runs/ --quiet
    # Signal the sweep script to stop accepting new runs
    touch /tmp/spot_interrupt_signal
    # GEPA's run_dir checkpoint is already on disk from the last iteration
    break
  fi
  sleep 5
done
```

The sweep script checks for `/tmp/spot_interrupt_signal` between runs and exits cleanly. On instance restart (spot capacity restored), the script resumes from `sweep_state.json`.

### Checkpoint resume protocol

1. **GEPA-level resume:** GEPA's `optimize()` accepts `run_dir` which persists engine state (frontier, evaluation cache, iteration counter) to disk. On restart, passing the same `run_dir` resumes from the last completed iteration. This is built into the existing `run_mutation()` flow.

2. **Sweep-level resume:** `sweep_state.json` tracks which (K, benchmark, seed) combinations have completed:

```json
{
  "completed": [
    {"k": 3, "benchmark": "hotpotqa", "seed": 42, "result_s3_key": "runs/hotpotqa/failure_stratified_k/K3/42/result.json"},
    {"k": 3, "benchmark": "hotpotqa", "seed": 123, "result_s3_key": "..."}
  ],
  "in_progress": {
    "k": 3, "benchmark": "hotpotqa", "seed": 456,
    "started_at": "2026-03-19T14:32:00Z",
    "run_dir": "runs/hotpotqa/failure_stratified_k/K3/456/gepa_state"
  },
  "remaining": [
    {"k": 3, "benchmark": "hover", "seed": 42},
    ...
  ]
}
```

3. **Resume sequence on instance restart:**
   ```bash
   # 1. Pull latest sweep state from S3
   aws s3 cp s3://gepa-mutations-results/sweep_state_fsk.json ./sweep_state.json
   # 2. Pull any in-progress run's GEPA state
   aws s3 sync s3://gepa-mutations-results/runs/ ./runs/ --exclude "*" \
     --include "*/failure_stratified_k/*/gepa_state/*"
   # 3. Resume sweep
   python -m gepa_mutations.experiments.failure_stratified_k.sweep --resume
   ```

### Self-termination

The EC2 instance self-terminates when the sweep completes to avoid idle cost:

```bash
# At the end of sweep.py (after S3 upload)
if os.environ.get("AUTO_TERMINATE", "false") == "true":
    instance_id = requests.get(
        "http://169.254.169.254/latest/meta-data/instance-id"
    ).text
    boto3.client("ec2").terminate_instances(InstanceIds=[instance_id])
```

Self-termination is gated by the `AUTO_TERMINATE` env var (set in the launch script) to prevent accidental termination during development.

### Failure modes and mitigations

| Failure mode | Detection | Mitigation |
|--------------|-----------|------------|
| Spot interruption | Instance metadata polling | Checkpoint + S3 sync + auto-resume |
| OpenRouter 429 (rate limit) | HTTP status in LiteLLM | LiteLLM built-in retry with exponential backoff (Section 6) |
| OpenRouter 5xx (server error) | HTTP status | LiteLLM retries (num_retries=3); sweep script retries full run once |
| GEPA exception (bug in proposer) | `raise_on_exception=True` in `optimize()` | Sweep script logs error, skips run, continues with next |
| Disk full | `df` check before each run | 20 GB EBS is sufficient; alert if < 2 GB free |
| Network partition | API timeout (60s default) | LiteLLM timeout + retry; S3 sync retries automatically |
| Sweep script crash (OOM, segfault) | CloudWatch agent (Section 7) | systemd auto-restart; resume from checkpoint |

---

## 6. Model API Configuration

### OpenRouter setup

```bash
# .env file on EC2 instance
OPENROUTER_API_KEY=sk-or-v1-...
HF_TOKEN=hf_...
```

The `LM` wrapper in `runner/experiment.py` uses LiteLLM under the hood. LiteLLM handles OpenRouter's API format natively when the model string is prefixed with `openrouter/`.

### Model identifier

```python
model = "openrouter/qwen/qwen3-8b"
```

This routes through OpenRouter to Qwen3-8B. The model is used for both the task LM (evaluating examples) and the reflection LM (proposing mutations).

### Sampling parameters (paper defaults)

```python
temperature = 0.6
top_p = 0.95
top_k = 20
max_tokens = 16384
```

These are set in `Settings` and propagated to both the task LM and reflection LM via `build_reflection_lm()` and `build_task_lm()` / `build_qa_task_lm()`.

### Rate limiting

OpenRouter enforces per-model rate limits. For Qwen3-8B, the typical limits are:
- 200 requests/minute (may vary by plan)
- 200,000 tokens/minute

GEPA's iteration loop is naturally rate-limited by sequential execution (each API call waits for the previous to complete). At K=3 with 3 examples per minibatch, each iteration makes ~10-15 API calls, completing in ~30-60 seconds. This translates to ~15-30 requests/minute, well under the limit.

**If rate-limited:** LiteLLM handles 429 responses automatically with exponential backoff:

```python
# Already configured in the LM wrapper via num_retries=3
LM(
    model="openrouter/qwen/qwen3-8b",
    num_retries=3,  # LiteLLM retries with exponential backoff
    ...
)
```

LiteLLM's default retry behavior for 429s: wait 1s, 2s, 4s (exponential backoff). If all 3 retries fail, the exception propagates and GEPA's `raise_on_exception=True` will surface it.

### Additional LiteLLM configuration

```bash
# .env additions for LiteLLM tuning
LITELLM_LOG=WARNING              # Reduce verbose logging
OPENROUTER_TIMEOUT=120           # 2-minute timeout per request (some reflections are long)
LITELLM_DROP_PARAMS=true         # Drop unsupported params silently
```

### API cost tracking

Between runs, query the OpenRouter credits API to track cumulative spend:

```python
import requests

response = requests.get(
    "https://openrouter.ai/api/v1/credits",
    headers={"Authorization": f"Bearer {api_key}"}
)
credits_remaining = response.json()["data"]["total_credits"]
```

Log this value after each run and abort the sweep if the budget cap ($50) is reached.

---

## 7. Monitoring

### CloudWatch metrics

Install the CloudWatch agent on the EC2 instance to capture system-level metrics:

```json
{
  "metrics": {
    "namespace": "GEPA/FailureStratifiedK",
    "metrics_collected": {
      "cpu": { "measurement": ["cpu_usage_idle"], "totalcpu": true },
      "mem": { "measurement": ["mem_used_percent"] },
      "disk": { "measurement": ["disk_used_percent"], "resources": ["/"] }
    }
  }
}
```

### Custom CloudWatch metrics (emitted by sweep script)

| Metric | Unit | Emitted when |
|--------|------|-------------|
| `RunCompleted` | Count | After each (K, benchmark, seed) run finishes |
| `RunFailed` | Count | After a run fails with an exception |
| `SweepProgress` | Percent | `completed_runs / total_runs * 100` after each run |
| `StratificationActivationRate` | Percent | Fraction of iterations where stratification activated (from metrics) |
| `TestScore` | None (0-1) | Test accuracy of completed run |
| `WallClockSeconds` | Seconds | Duration of completed run |
| `OpenRouterSpend` | None (USD) | Cumulative API spend after each run |

Emit via boto3:

```python
cloudwatch = boto3.client("cloudwatch")
cloudwatch.put_metric_data(
    Namespace="GEPA/FailureStratifiedK",
    MetricData=[{
        "MetricName": "RunCompleted",
        "Value": 1,
        "Unit": "Count",
        "Dimensions": [
            {"Name": "Benchmark", "Value": benchmark},
            {"Name": "K", "Value": str(k)},
        ]
    }]
)
```

### CloudWatch alarms

| Alarm | Condition | Action |
|-------|-----------|--------|
| `FSK-RunFailed` | `RunFailed >= 2` in 30 min | SNS notification |
| `FSK-NoProgress` | `RunCompleted == 0` for 4 hours | SNS notification (likely hung or interrupted) |
| `FSK-HighSpend` | `OpenRouterSpend > 40` | SNS notification (approaching $50 cap) |
| `FSK-DiskFull` | `disk_used_percent > 85` | SNS notification |
| `FSK-SweepComplete` | `SweepProgress == 100` | SNS notification (all runs done) |

### SNS / Telegram notifications

Use the existing `Notifier` class from `src/gepa_mutations/notifications/notifier.py`. The sweep script sends notifications at:

1. **Sweep start:** List of runs to execute, estimated duration, dependency check result.
2. **Phase completion:** After each phase (e.g., "Phase 1 complete: HotpotQA K=3, 3 seeds. Mean fsk score: X vs bok score: Y").
3. **Go/no-go decision points:** After Phase 1, report whether stratification shows signal on HotpotQA/HoVer.
4. **Run failure:** Immediate notification with benchmark, seed, K, and error message.
5. **Sweep completion:** Final summary with all scores, S3 paths, total cost, total wall time.
6. **Spot interruption:** Notification that the instance was interrupted and work will resume on restart.

Telegram is preferred for real-time monitoring (push notifications to phone). SNS is used for alarm-driven notifications (CloudWatch alarms route to SNS topics).

### Log aggregation

The sweep script writes structured logs to `/var/log/gepa-fsk-sweep.log` with JSON formatting:

```json
{
  "timestamp": "2026-03-19T15:30:00Z",
  "level": "INFO",
  "event": "run_completed",
  "benchmark": "hotpotqa",
  "k": 3,
  "seed": 42,
  "stratified": true,
  "test_score": 0.6433,
  "stratification_activation_rate": 0.45,
  "wall_clock_seconds": 14523,
  "iterations": 487
}
```

This log is tailed via CloudWatch Logs agent for centralized access:

```json
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [{
          "file_path": "/var/log/gepa-fsk-sweep.log",
          "log_group_name": "/gepa/failure_stratified_k",
          "log_stream_name": "{instance_id}"
        }]
      }
    }
  }
}
```

---

## 8. Runbook

### Prerequisites

1. best_of_k sweep must be complete. Verify:
   ```bash
   # Check that all best_of_k control results exist in S3
   for benchmark in hotpotqa hover ifbench aime; do
     for k in 3 5 7; do
       for seed in 42 123 456; do
         aws s3 ls "s3://gepa-mutations-results/runs/${benchmark}/best_of_k/K${k}/${seed}/result.json" \
           || echo "MISSING: ${benchmark}/best_of_k/K${k}/${seed}"
       done
     done
   done
   ```
   If any are missing, **do not proceed**. Complete the best_of_k sweep first.

2. OpenRouter API key with sufficient credits (>= $50 remaining).

3. AWS credentials configured with permissions for: EC2, S3, CloudWatch, SNS.

### Step 1: Launch EC2 instance

```bash
# Launch spot instance (us-east-1, Ubuntu 22.04 AMI)
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type t3.medium \
  --key-name gepa-experiments \
  --security-group-ids sg-XXXXXXXXX \
  --iam-instance-profile Name=gepa-experiment-role \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=gepa-fsk-sweep},{Key=Project,Value=gepa-mutations},{Key=Experiment,Value=failure_stratified_k}]' \
  --user-data file://scripts/ec2_userdata_fsk.sh
```

### Step 2: Instance setup (in user-data or SSH)

```bash
#!/bin/bash
set -euo pipefail

# System packages
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv git awscli

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

# Clone repo
git clone https://github.com/<org>/gepa-mutations.git /home/ubuntu/gepa-mutations
cd /home/ubuntu/gepa-mutations
git submodule update --init --recursive

# Install dependencies
uv sync

# Copy secrets from SSM Parameter Store
aws ssm get-parameter --name /gepa/openrouter-api-key --with-decryption \
  --query 'Parameter.Value' --output text > /tmp/openrouter_key
aws ssm get-parameter --name /gepa/hf-token --with-decryption \
  --query 'Parameter.Value' --output text > /tmp/hf_token
aws ssm get-parameter --name /gepa/telegram-bot-token --with-decryption \
  --query 'Parameter.Value' --output text > /tmp/telegram_token
aws ssm get-parameter --name /gepa/telegram-chat-id --with-decryption \
  --query 'Parameter.Value' --output text > /tmp/telegram_chat_id

cat > /home/ubuntu/gepa-mutations/.env <<ENVEOF
OPENROUTER_API_KEY=$(cat /tmp/openrouter_key)
HF_TOKEN=$(cat /tmp/hf_token)
TELEGRAM_BOT_TOKEN=$(cat /tmp/telegram_token)
TELEGRAM_CHAT_ID=$(cat /tmp/telegram_chat_id)
S3_BUCKET=gepa-mutations-results
ENVEOF
rm /tmp/openrouter_key /tmp/hf_token /tmp/telegram_token /tmp/telegram_chat_id

# Pull best_of_k results (control condition)
aws s3 sync s3://gepa-mutations-results/runs/ /home/ubuntu/gepa-mutations/runs/ \
  --exclude "*" --include "*/best_of_k/*/result.json"

# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb
sudo amazon-cloudwatch-agent-ctl -a start -m ec2 -c file:/home/ubuntu/gepa-mutations/configs/cloudwatch_fsk.json

# Set up spot interrupt monitor
nohup /home/ubuntu/gepa-mutations/scripts/spot_interrupt_monitor.sh &

# Set environment variables
export AUTO_TERMINATE=true
```

### Step 3: Start the sweep

```bash
cd /home/ubuntu/gepa-mutations

# Start sweep in a tmux session (survives SSH disconnect)
tmux new-session -d -s fsk-sweep \
  'uv run python -m gepa_mutations.experiments.failure_stratified_k.sweep \
    --benchmarks hotpotqa hover ifbench aime \
    --k-values 3 5 7 \
    --seeds 42 123 456 \
    --resume \
    2>&1 | tee /var/log/gepa-fsk-sweep.log'
```

### Step 4: Monitor progress

```bash
# From local machine: check sweep progress
aws cloudwatch get-metric-data \
  --metric-data-queries '[{
    "Id": "progress",
    "MetricStat": {
      "Metric": {
        "Namespace": "GEPA/FailureStratifiedK",
        "MetricName": "SweepProgress"
      },
      "Period": 3600,
      "Stat": "Maximum"
    }
  }]' \
  --start-time $(date -u -v-24H +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ)

# Check CloudWatch logs
aws logs tail /gepa/failure_stratified_k --follow

# SSH into the instance and attach to tmux
ssh -i gepa-experiments.pem ubuntu@<instance-ip>
tmux attach -t fsk-sweep

# Quick check: how many runs completed?
aws s3 ls s3://gepa-mutations-results/runs/ --recursive \
  | grep "failure_stratified_k" | grep "result.json" | wc -l

# Check latest results for a specific benchmark
aws s3 cp s3://gepa-mutations-results/runs/hotpotqa/failure_stratified_k/K3/42/result.json - \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Test: {d[\"test_score\"]:.4f}')"
```

### Step 5: Go/No-Go after Phase 1

After HotpotQA and HoVer K=3 runs complete (6 runs, ~3 seeds each):

```bash
# Download Phase 1 results
aws s3 sync s3://gepa-mutations-results/runs/hotpotqa/ ./results/hotpotqa/ \
  --include "*/result.json" --exclude "*gepa_state*"
aws s3 sync s3://gepa-mutations-results/runs/hover/ ./results/hover/ \
  --include "*/result.json" --exclude "*gepa_state*"

# Compare fsk vs bok (paired by seed)
uv run python -m gepa_mutations.experiments.failure_stratified_k.analyze \
  --benchmarks hotpotqa hover \
  --k 3

# Decision:
#   fsk_K3 > bok_K3 on HotpotQA or HoVer? -> Continue to Phase 2
#   fsk_K3 <= bok_K3 on both?              -> Stop sweep, report null result
```

### Step 6: Collect final results

After sweep completes (or after early stopping):

```bash
# Download all failure_stratified_k results
aws s3 sync s3://gepa-mutations-results/runs/ ./results/ \
  --include "*/failure_stratified_k/*/result.json" \
  --include "*/failure_stratified_k/*/metrics.json" \
  --include "*/best_of_k/*/result.json" \
  --exclude "*gepa_state*"

# Generate comparison report
uv run python -m gepa_mutations.experiments.failure_stratified_k.analyze \
  --benchmarks hotpotqa hover ifbench aime \
  --k-values 3 5 7 \
  --output results/failure_stratified_k_report.json

# Verify instance self-terminated (or terminate manually)
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=gepa-fsk-sweep" \
  --query "Reservations[].Instances[].[InstanceId,State.Name]" --output table
```

### Step 7: Cleanup

```bash
# Verify all results are in S3 before cleanup
aws s3 ls s3://gepa-mutations-results/runs/ --recursive \
  | grep "failure_stratified_k" | grep "result.json" | wc -l
# Expected: 36 (if full sweep) or fewer (if early-stopped)

# Terminate instance if still running
aws ec2 terminate-instances --instance-ids i-XXXXXXXXX

# Keep S3 results indefinitely (no lifecycle policy)
# Keep CloudWatch logs for 30 days (set retention in console)
aws logs put-retention-policy \
  --log-group-name /gepa/failure_stratified_k \
  --retention-in-days 30
```

### Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Sweep not progressing | Check tmux session, CloudWatch logs | SSH in, check `/var/log/gepa-fsk-sweep.log` for errors |
| All runs show identical fsk/bok scores | Stratification never activating (all minibatch examples passing) | Check `stratification_activation_rate` in metrics. If < 0.05, the prompt is too good for the minibatch -- this is a valid null result. |
| API errors (429/5xx) | Rate limit or OpenRouter outage | LiteLLM retries handle transient issues. For sustained outages, pause the sweep and wait. |
| Instance terminated unexpectedly | Spot interruption or self-termination bug | Check CloudWatch for spot interruption event. Re-launch instance, resume from checkpoint. |
| Results missing from S3 | S3 sync failed before termination | Check local disk on EBS volume (still attached if `stop` behavior). Mount volume on new instance, sync manually. |
| GEPA raises exception mid-run | Bug in proposer patch | Check the error traceback in logs. Fix the code, push to git, pull on instance, resume sweep. The failed run will be retried. |
| Test scores are anomalously low | Wrong model, wrong benchmark data, or broken evaluator | Verify model string (`openrouter/qwen/qwen3-8b`), re-run smoke test with `--subset 5 --max-metric-calls 20`. |

---

## Appendix: Estimated Timeline

| Phase | Runs | Est. wall clock | Est. cost |
|-------|------|----------------|-----------|
| Phase 0: Prerequisite check | 0 | 5 min | $0 |
| Phase 1: HotpotQA + HoVer, K=3 | 6 | ~30 hrs | ~$5.40 |
| Go/no-go analysis | 0 | 1 hr | $0 |
| Phase 2: HotpotQA + HoVer, K=5,7 | 12 | ~50 hrs | ~$9.90 |
| Phase 3: IFBench + AIME, K=3,5,7 | 18 | ~75 hrs | ~$24.50 |
| Phase 4 (conditional): PUPA + LiveBench | 6-18 | ~25-60 hrs | ~$5-15 |
| **Total (Phases 0-3)** | **36** | **~155 hrs (~6.5 days)** | **~$41.79** |

Phases 1-3 can be executed sequentially on a single EC2 instance running 24/7. The entire sweep completes within ~7 days of continuous operation, including spot interruption recovery overhead.
