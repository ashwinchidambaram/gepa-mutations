# AWS Implementation: best_of_k Mutation Sweep

**Date:** 2026-03-19
**Status:** Draft
**Depends on:** `mutation_plan.md` (same directory)

---

## 1. EC2 Instance Selection

### Workload Profile

The best_of_k mutation is **CPU-idle, network-bound**. The instance spends >95% of wall-clock time waiting for OpenRouter API responses (Qwen3-8B inference). Local compute is limited to:

- Python control flow (proposer logic, deduplication hashing)
- Score aggregation and candidate comparison
- GEPA state serialization (~1-5 MB per checkpoint)
- Data loading at startup (~100 MB peak for HotpotQA)

There is zero GPU usage. The instance needs reliable networking, enough RAM to hold benchmark data and GEPA state, and nothing else.

### Recommended Instance

| Property | Value | Rationale |
|----------|-------|-----------|
| **Instance type** | `t3.small` | 2 vCPUs, 2 GiB RAM. CPU is idle 95%+ of the time; 2 GiB is sufficient for Python + benchmark data + GEPA state. Burstable is fine because the baseline CPU credit accumulation rate exceeds the tiny compute demand. |
| **Pricing model** | **Spot** | $0.0066/hr spot vs $0.0208/hr on-demand (us-east-1, 2026-03 pricing). 68% savings. Spot interruption risk is acceptable because GEPA has built-in state checkpointing via `run_dir` and experiments resume cleanly from the last iteration. |
| **AMI** | Amazon Linux 2023 (arm64: `t4g.small` is even cheaper at $0.0053/hr spot, but requires arm64 Python wheels; use x86 `t3.small` unless arm64 is validated) |
| **Storage** | 20 GiB gp3 EBS root volume. GEPA state files are small (<50 MB cumulative). Benchmark data downloads are cached locally. |
| **Region** | `us-east-1` — lowest spot pricing, closest to OpenRouter's US endpoints for minimal API latency. |

### Why not smaller?

- `t3.micro` (1 GiB RAM) risks OOM during HotpotQA data loading + GEPA state serialization simultaneously.
- `t3.nano` (0.5 GiB) is ruled out entirely.

### Why not larger?

- `t3.medium` (4 GiB) doubles cost for no benefit. The workload is network-bound, not compute-bound.
- `c`/`m`/`r` families are overkill for a workload that idles at <5% CPU.

### Spot Interruption Tolerance

A spot interruption loses at most the current iteration's in-progress API calls (typically 1-7 LLM calls depending on K). The experiment resumes from the last completed iteration via GEPA's `run_dir` checkpoint. At Qwen3-8B pricing, the lost API cost per interruption is <$0.01. The `t3.small` spot interruption frequency in us-east-1 is historically <5% over 24 hours, making this an acceptable risk.

---

## 2. S3 Storage

### Bucket

**Bucket name:** `gepa-mutations-results` (already configured in `Settings.s3_bucket`)

### Key Structure

```
s3://gepa-mutations-results/
  runs/
    {benchmark}/
      best_of_k_K{k}/          # method name encodes K value
        {seed}/
          config.json           # MutationConfig snapshot
          result.json           # final scores + best prompt
          metrics.json          # per-iteration diagnostics
          gepa_state/           # GEPA's run_dir checkpoint tree
            candidates/         # per-candidate data
            ...
          logs/
            stdout.log          # full experiment stdout
            run_meta.json       # instance ID, start/end times, spot price
```

Example paths:
```
s3://gepa-mutations-results/runs/aime/best_of_k_K3/42/result.json
s3://gepa-mutations-results/runs/hotpotqa/best_of_k_K1/123/metrics.json
```

The method name `best_of_k_K{k}` is used instead of a flat `best_of_k` so that each K value is a separate "method" in the existing `save_result()` / `upload_results()` infrastructure. This maps cleanly to:
```python
method = f"best_of_k_K{config.mutation_candidates}"
```

### Checkpointing Strategy

GEPA's `run_dir` parameter already enables automatic state persistence. After each iteration, `GEPAEngine` serializes the full optimizer state (Pareto front, candidate history, evaluation cache, RNG state) to the `run_dir` directory. On restart with the same `run_dir`, optimization resumes from exactly where it stopped.

**Checkpoint upload cadence:**

1. **After each completed run** (benchmark + seed + K): Upload the full `runs/{benchmark}/best_of_k_K{k}/{seed}/` directory to S3. This is the primary persistence path.

2. **Periodic mid-run backup** (every 30 minutes): Sync the `gepa_state/` subdirectory to S3. This limits data loss on spot interruption to at most 30 minutes of iteration progress (typically 5-15 iterations, worth <$0.05 in API calls).

3. **On spot interruption** (2-minute warning handler): Immediately sync current `gepa_state/` to S3 before the instance terminates.

**Implementation:**

```bash
# Periodic sync (run as background process)
while true; do
    aws s3 sync runs/ s3://gepa-mutations-results/runs/ \
        --exclude "*.pyc" --exclude "__pycache__/*"
    sleep 1800  # 30 minutes
done &
```

```python
# Spot interruption handler (in the experiment runner)
import signal
import subprocess

def handle_spot_interruption(signum, frame):
    """Upload checkpoint on spot termination notice."""
    subprocess.run([
        "aws", "s3", "sync", "runs/",
        "s3://gepa-mutations-results/runs/",
        "--exclude", "*.pyc",
    ], timeout=110)  # 110s of the 120s warning
    sys.exit(0)

# Register for SIGTERM (sent by EC2 spot interruption)
signal.signal(signal.SIGTERM, handle_spot_interruption)
```

### Resume from Checkpoint

When launching a run that may have been interrupted:

```bash
# Pull existing checkpoint from S3 (no-op if none exists)
aws s3 sync \
    "s3://gepa-mutations-results/runs/${BENCHMARK}/best_of_k_K${K}/${SEED}/" \
    "runs/${BENCHMARK}/best_of_k_K${K}/${SEED}/"

# GEPA's optimize() will detect the existing run_dir and resume
python -m gepa_mutations.experiments.best_of_k.runner \
    --benchmark "$BENCHMARK" --seed "$SEED" --k "$K"
```

---

## 3. Parameter Sweep Execution

### Sweep Dimensions

From the mutation plan:

| Dimension | Values | Count |
|-----------|--------|-------|
| K (mutation_candidates) | 1, 3, 5, 7 | 4 |
| Benchmarks (Tier 1) | AIME-2025, HotpotQA | 2 |
| Benchmarks (Tier 2) | HoVer, LiveBench-Math | 2 |
| Benchmarks (Tier 3) | IFBench, PUPA | 2 |
| Seeds | 42, 123, 456 | 3 |

### Total Run Count

| Tier | K values | Benchmarks | Seeds | Runs |
|------|----------|------------|-------|------|
| Tier 1 (full) | 4 | 2 | 3 | **24** |
| Tier 2 (K=1,3 only) | 2 | 2 | 3 | **12** |
| Tier 3 (conditional, K=1,3 only) | 2 | 2 | 3 | **12** |
| **Total (Tier 1+2)** | | | | **36** |
| **Total (all tiers)** | | | | **48** |

### Execution Strategy: Sequential on One Instance

**Recommended:** Run all experiments sequentially on a single `t3.small` spot instance.

**Why sequential, not parallel:**

1. **OpenRouter rate limits.** Qwen3-8B on OpenRouter has per-key rate limits (typically 60-200 RPM for free tier, higher for paid). Running multiple experiments in parallel from separate instances would hit rate limits faster, causing retries and wasted time. Sequential execution on one instance naturally throttles to a sustainable request rate.

2. **Cost equivalence.** Since the workload is API-bound (not CPU-bound), running N experiments in parallel on N instances takes the same total wall-clock API time and costs the same in API fees. The only savings from parallelism is reduced EC2 hours, but EC2 cost is negligible compared to API cost (see Section 4). Running 36 experiments sequentially on one instance vs 36 instances in parallel saves EC2 management complexity for ~$0.50 in EC2 cost difference.

3. **Simpler failure recovery.** One instance means one checkpoint state, one log stream, one monitoring target. Debugging a failed run on 1 instance is trivial; debugging 36 concurrent instances is not.

4. **Early stopping is trivial.** The mutation plan specifies a go/no-go checkpoint after Tier 1 K=1 and K=3. With sequential execution, the sweep script simply evaluates results after each block and decides whether to continue. With parallel execution, you'd need inter-instance coordination.

**Exception: Tier 1 and Tier 2 can run on separate instances** if wall-clock time is a concern. Tier 1 (AIME + HotpotQA, all K values) is independent from Tier 2 (HoVer + LiveBench, K=1 and K=3 only). Two instances with separate OpenRouter API keys would avoid rate limit contention.

### Execution Order

The sweep follows the mutation plan's tiered priority with early stopping:

```
# Phase A: Tier 1 confirmation (go/no-go gate)
AIME    K=1 seed=42   -> AIME    K=1 seed=123  -> AIME    K=1 seed=456
AIME    K=3 seed=42   -> AIME    K=3 seed=123  -> AIME    K=3 seed=456
HotpotQA K=1 seed=42  -> HotpotQA K=1 seed=123 -> HotpotQA K=1 seed=456
HotpotQA K=3 seed=42  -> HotpotQA K=3 seed=123 -> HotpotQA K=3 seed=456

>>> CHECKPOINT: Compare K=1 vs K=3 on AIME and HotpotQA.
>>> If K=3 mean <= K=1 mean on AIME: STOP. H1 falsified.
>>> If K=3 > K=1 on AIME: continue to Phase B.

# Phase B: Tier 1 scaling (K=5, K=7)
AIME    K=5 (3 seeds) -> AIME    K=7 (3 seeds)
HotpotQA K=5 (3 seeds) -> HotpotQA K=7 (3 seeds)

# Phase C: Tier 2 generalizability
HoVer    K=1 (3 seeds) -> HoVer    K=3 (3 seeds)
LiveBench K=1 (3 seeds) -> LiveBench K=3 (3 seeds)

# Phase D: Tier 3 (conditional on Tier 1-2 signal)
IFBench  K=1 (3 seeds) -> IFBench  K=3 (3 seeds)
PUPA     K=1 (3 seeds) -> PUPA     K=3 (3 seeds)
```

### Wall-Clock Time Estimates

Each GEPA run with paper-level budgets takes approximately 4-12 hours depending on benchmark budget size and API response latency. Estimates:

| Benchmark | Budget | Est. wall-clock per run | Runs (Tier 1+2) | Total hours |
|-----------|--------|------------------------|-----------------|-------------|
| AIME | 7,051 | ~10 hrs | 12 | ~120 hrs |
| HotpotQA | 6,871 | ~10 hrs | 12 | ~120 hrs |
| HoVer | 2,426 | ~4 hrs | 6 | ~24 hrs |
| LiveBench | 1,839 | ~3 hrs | 6 | ~18 hrs |
| **Tier 1+2 total** | | | **36** | **~282 hrs (~12 days)** |

These estimates assume ~100ms average API latency per call and include reflection LM calls (which scale with iterations, not metric calls). Actual time depends heavily on OpenRouter queue depth and rate limiting.

---

## 4. Cost Optimization

### Cost Breakdown

#### OpenRouter API Cost (Dominant)

Qwen3-8B pricing: **$0.05/M input tokens, $0.40/M output tokens**.

Per metric call (one task-model inference):
- Input: ~2,000 tokens (system prompt + task input) = $0.0001
- Output: ~500 tokens (model response) = $0.0002
- **Total per metric call: ~$0.0003**

Per reflection call (one propose_new_texts invocation):
- Input: ~4,000 tokens (parent prompt + reflective dataset + instruction) = $0.0002
- Output: ~1,500 tokens (new prompt text) = $0.0006
- **Total per reflection call: ~$0.0008**

| Component | Tier 1+2 Volume | Unit Cost | Total |
|-----------|----------------|-----------|-------|
| Metric calls (task model) | 192,654 calls | $0.0003/call | **$57.80** |
| Reflection calls (propose_new_texts) | ~25,000 calls* | $0.0008/call | **$20.00** |
| Test set evaluation | 36 runs x ~150 examples avg | $0.0003/call | **$1.62** |
| **API total** | | | **~$79.42** |

*Reflection call estimate: Each iteration makes 1 reflection call per component updated (typically 1) times K. At ~25,000 total iterations across all 36 runs, this is ~25,000 reflection calls. Higher K values reduce iteration count but increase reflection calls per iteration, roughly canceling out.

#### EC2 Cost (Negligible)

| Component | Hours | Rate | Total |
|-----------|-------|------|-------|
| `t3.small` spot (282 hrs) | 282 | $0.0066/hr | **$1.86** |
| EBS gp3 20 GiB (12 days) | 288 | $0.08/GB-month | **$0.02** |
| Data transfer (S3 sync) | — | Negligible | **~$0.05** |
| **EC2 total** | | | **~$1.93** |

#### Grand Total (Tier 1 + Tier 2)

| Category | Cost | % of Total |
|----------|------|------------|
| OpenRouter API | $79.42 | **97.6%** |
| EC2 + infra | $1.93 | 2.4% |
| **Total** | **~$81.35** | 100% |

### Cost Reduction Strategies

1. **Early stopping (highest impact).** If K=3 shows no gain on AIME after Phase A (12 runs, ~$22 API cost), stop the entire sweep. This caps the worst-case cost for a negative result at ~$24 total.

2. **Tier 3 is conditional.** Only run IFBench and PUPA if Tier 1-2 show signal. This saves ~$20 in API costs if the mutation is AIME-specific.

3. **Skip K=7 if K=5 plateaus.** If K=5 shows no improvement over K=3 on AIME, skip K=7 on all benchmarks. Saves 6 runs (~$12).

4. **Spot pricing.** Already using spot instances. On-demand would add only ~$4 more (EC2 is 2% of cost), so this is a minor optimization.

5. **OpenRouter caching.** OpenRouter caches identical requests. If the deduplication mechanism in the proposer produces identical reflection prompts across seeds (unlikely but possible), some reflection calls may hit cache and cost nothing. Do not rely on this.

6. **Do NOT reduce rollout budgets.** The mutation plan explicitly requires paper-matched budgets for fair comparison. Reducing budgets would invalidate the results. Cost reduction must come from skipping runs, not shrinking runs.

---

## 5. Reliability

### Spot Interruption Handling

EC2 spot instances receive a 2-minute warning before termination via the instance metadata endpoint and a SIGTERM signal. The experiment runner must handle both.

#### Signal Handler (Primary)

```python
# In the sweep runner entry point
import signal
import subprocess
import sys
import json
from datetime import datetime, timezone

def _sync_to_s3():
    """Best-effort sync of all run data to S3."""
    subprocess.run(
        ["aws", "s3", "sync", "runs/",
         "s3://gepa-mutations-results/runs/",
         "--exclude", "*.pyc", "--exclude", "__pycache__/*"],
        timeout=110,
        capture_output=True,
    )

def _save_sweep_progress(current_run_idx: int, sweep_manifest: list[dict]):
    """Save sweep progress so a new instance knows where to resume."""
    progress = {
        "last_completed_idx": current_run_idx - 1,
        "current_run_idx": current_run_idx,
        "total_runs": len(sweep_manifest),
        "interrupted_at": datetime.now(timezone.utc).isoformat(),
    }
    with open("sweep_progress.json", "w") as f:
        json.dump(progress, f)

def handle_spot_interruption(signum, frame):
    _save_sweep_progress(current_run_idx, sweep_manifest)
    _sync_to_s3()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_spot_interruption)
```

#### Metadata Polling (Backup)

For cases where SIGTERM is not delivered (rare but documented):

```python
import requests
import threading

def poll_spot_interruption():
    """Background thread that polls EC2 metadata for spot interruption notice."""
    while True:
        try:
            resp = requests.get(
                "http://169.254.169.254/latest/meta-data/spot/instance-action",
                timeout=2,
            )
            if resp.status_code == 200:
                # Interruption notice received
                handle_spot_interruption(None, None)
        except requests.exceptions.RequestException:
            pass  # No interruption notice (normal case: 404)
        time.sleep(5)

threading.Thread(target=poll_spot_interruption, daemon=True).start()
```

### Checkpoint Resume via GEPA's run_dir

GEPA's `optimize()` function accepts a `run_dir` parameter. When a `run_dir` contains a prior state checkpoint, `optimize()` loads it and resumes from the last completed iteration. This is GEPA's built-in mechanism for exactly this use case.

**Resume path for best_of_k:**

```
runs/{benchmark}/best_of_k_K{k}/{seed}/gepa_state/
```

The `run_mutation()` function in `base.py` already constructs this path:
```python
run_dir = f"runs/{config.benchmark}/{config.mutation_name}/{config.seed}/gepa_state"
```

For the best_of_k sweep, `config.mutation_name` should be set to `best_of_k_K{k}` so each K value gets its own state directory.

**Resume sequence after spot interruption:**

1. New spot instance launches (via Auto Scaling Group relaunch or manual re-run of the launch script).
2. `sweep_progress.json` is downloaded from S3 to determine which run was interrupted.
3. The interrupted run's `gepa_state/` is downloaded from S3.
4. `optimize()` is called with the same `run_dir`, which loads the checkpoint and resumes.
5. The sweep continues from the interrupted run onward.

### Self-Termination

The instance must terminate itself on both completion and unrecoverable failure to avoid paying for idle EC2 hours.

```python
import boto3
import requests

def get_instance_id() -> str:
    """Get this instance's ID from EC2 metadata."""
    token_resp = requests.put(
        "http://169.254.169.254/latest/api/token",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        timeout=2,
    )
    resp = requests.get(
        "http://169.254.169.254/latest/meta-data/instance-id",
        headers={"X-aws-ec2-metadata-token": token_resp.text},
        timeout=2,
    )
    return resp.text

def self_terminate(reason: str):
    """Terminate this EC2 instance after final S3 sync."""
    _sync_to_s3()
    instance_id = get_instance_id()
    ec2 = boto3.client("ec2", region_name="us-east-1")

    # Tag the instance with termination reason before killing it
    ec2.create_tags(
        Resources=[instance_id],
        Tags=[{"Key": "TerminationReason", "Value": reason[:255]}],
    )
    ec2.terminate_instances(InstanceIds=[instance_id])
```

**Invocation points:**

```python
try:
    run_sweep(manifest)
    self_terminate("sweep_complete")
except Exception as e:
    notify_error(str(e))
    self_terminate(f"sweep_failed: {e}")
```

### Unrecoverable Failure Detection

The sweep runner should distinguish between recoverable errors (API timeout, transient 5xx) and unrecoverable errors that warrant stopping:

| Error | Recoverable? | Action |
|-------|-------------|--------|
| OpenRouter 429 (rate limit) | Yes | Exponential backoff, retry |
| OpenRouter 5xx | Yes | Retry up to 5 times with backoff |
| OpenRouter 401 (auth) | **No** | Notify + self-terminate |
| OpenRouter balance exhausted | **No** | Notify + self-terminate |
| Python OOM | **No** | Notify + self-terminate |
| GEPA internal assertion | Depends | Save state, skip this run, continue sweep |
| Benchmark data download failure | Yes | Retry 3 times, then skip benchmark |

---

## 6. Model API

### OpenRouter Configuration

```python
# Environment variables (set in .env or EC2 user-data)
OPENROUTER_API_KEY=sk-or-v1-...

# LiteLLM model string (used by both task LM and reflection LM)
MODEL = "openrouter/qwen/qwen3-8b"
```

The existing `LM` class in `runner/experiment.py` and the helpers in `base.py` (`build_reflection_lm`, `build_task_lm`, `build_qa_task_lm`) already handle this correctly. They pass `temperature=0.6`, `top_p=0.95`, `top_k=20`, and `max_tokens=16384` to LiteLLM, which routes to OpenRouter.

### Rate Limiting Strategy

OpenRouter enforces per-key rate limits that vary by plan tier. The strategy is to stay under the limit proactively rather than relying on 429 retries.

**Approach: Adaptive rate limiting with token bucket.**

```python
import time
import threading

class RateLimiter:
    """Token-bucket rate limiter for OpenRouter API calls."""

    def __init__(self, requests_per_minute: int = 50):
        self.rpm = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.last_request_time = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_request_time = time.monotonic()
```

**Recommended starting RPM:** 50 RPM (conservative). Increase to 100-200 RPM if OpenRouter headers indicate higher limits. The `x-ratelimit-remaining` response header can be used to dynamically adjust.

**Note:** LiteLLM (used by both the `LM` wrapper and `dspy.LM`) has built-in retry logic with exponential backoff for 429 responses. The `num_retries=3` parameter in our `LM` class enables this. Additional rate limiting on our side is a defense-in-depth measure to avoid burning through retries.

### Retry Logic

LiteLLM handles retries for transient failures automatically. The configuration:

```python
# Already in our LM class
LM(
    model="openrouter/qwen/qwen3-8b",
    num_retries=3,           # LiteLLM retries on 429, 500, 502, 503, 504
    ...
)
```

For the best_of_k proposer specifically, each of the K calls to `propose_new_texts()` is independent. If one fails after all retries:

```python
for k in range(K):
    try:
        new_texts = self.propose_new_texts(...)
    except Exception as e:
        logger.warning(f"K={k} proposal failed: {e}. Skipping this candidate.")
        continue  # Other K candidates still proceed

    # ... evaluate ...
```

This is specified in the mutation plan (test case #5: "Empty proposal test").

### Error Handling for Mid-Experiment API Failures

| Failure mode | Detection | Response |
|-------------|-----------|----------|
| Single K proposal fails | Exception in `propose_new_texts()` | Skip that K, evaluate remaining K-1 candidates |
| All K proposals fail in one iteration | Zero unique candidates after K attempts | Log warning, return parent as the "best" candidate (no-improvement iteration). The engine continues to the next iteration. |
| Sustained API failure (>10 consecutive failures) | Counter in the sweep runner | Pause for 5 minutes, then retry. After 3 pauses, notify and self-terminate. |
| API key invalidated mid-run | 401 response | Immediate notification + self-terminate. No point retrying. |
| Context length exceeded | 400 response with context_length error | Log the offending prompt length, skip this iteration. This can happen if the evolved prompt grows very large. |

### Cost Tracking

LiteLLM tracks token usage per call. The sweep runner should aggregate this:

```python
# After each run completes
import litellm
usage = litellm._current_cost  # Cumulative cost tracked by litellm
# Log to run_meta.json
```

Additionally, set a **hard cost cap** via OpenRouter's dashboard or API:
- Set a monthly limit of $150 (2x the estimated sweep cost)
- This prevents runaway costs if a bug causes infinite retry loops

---

## 7. Monitoring

### CloudWatch Metrics

Publish custom metrics from the experiment runner to CloudWatch for real-time dashboard visibility.

**Namespace:** `GEPAMutations/BestOfK`

| Metric | Dimensions | Unit | Source |
|--------|-----------|------|--------|
| `IterationsCompleted` | Benchmark, K, Seed | Count | `on_iteration_end` callback |
| `MetricCallsUsed` | Benchmark, K, Seed | Count | `on_budget_updated` callback |
| `BestValScore` | Benchmark, K, Seed | None (0-1) | `on_valset_evaluated` callback |
| `WallClockPerIteration` | Benchmark, K, Seed | Seconds | `on_iteration_end` callback |
| `APIErrorCount` | Benchmark, K, Seed | Count | LiteLLM exception handler |
| `DeduplicationRate` | Benchmark, K, Seed | Percent | `BestOfKMetricsCallback` |
| `SweepProgress` | (none) | Count | Sweep runner (runs completed / total) |

**Publishing frequency:** Every 60 seconds (CloudWatch free tier allows 10 custom metrics with 1-minute granularity).

```python
import boto3

cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")

def publish_metric(metric_name: str, value: float, dimensions: dict[str, str], unit: str = "Count"):
    cloudwatch.put_metric_data(
        Namespace="GEPAMutations/BestOfK",
        MetricData=[{
            "MetricName": metric_name,
            "Value": value,
            "Unit": unit,
            "Dimensions": [
                {"Name": k, "Value": v} for k, v in dimensions.items()
            ],
        }],
    )
```

### SNS / Telegram Alerts

The existing `Notifier` class in `notifications/notifier.py` supports both SNS and Telegram. Extend it for sweep-level events.

**SNS Topic:** `arn:aws:sns:us-east-1:{account_id}:gepa-mutations-alerts`

**Alert events:**

| Event | Channel | Message |
|-------|---------|---------|
| Sweep started | Telegram | Instance ID, total runs, estimated completion time |
| Individual run completed | Telegram | Benchmark, K, seed, test score, wall-clock time |
| Tier checkpoint (go/no-go) | Telegram + SNS | K=1 vs K=3 comparison table, recommendation |
| Run failed (recoverable) | Telegram | Error message, retry count |
| Sweep failed (unrecoverable) | Telegram + SNS | Error, last successful run, instance will self-terminate |
| Sweep completed | Telegram + SNS | Full results summary table, total cost, S3 path |
| Spot interruption detected | Telegram | Checkpoint status, expected resume time |

**Telegram message format for run completion:**

```
GEPA best_of_k Run Complete
Benchmark: aime | K: 3 | Seed: 42
Test: 38.00% | Val: 40.12%
Budget: 7051/7051 metric calls
Wall clock: 9.3 hrs
Progress: 5/36 runs (14%)
```

**Telegram message format for tier checkpoint:**

```
GEPA best_of_k: Tier 1 Checkpoint

AIME-2025:
  K=1: 31.33% +/- 2.1% (seeds: 30, 32, 32)
  K=3: 38.67% +/- 3.4% (seeds: 36, 38, 42)
  Delta: +7.34pp (d=1.82)

HotpotQA:
  K=1: 62.00% +/- 1.5%
  K=3: 63.33% +/- 2.0%
  Delta: +1.33pp (d=0.54)

Recommendation: CONTINUE (H1 confirmed on AIME)
Next: Phase B (K=5, K=7 on AIME + HotpotQA)
```

### Progress Tracking

**`sweep_progress.json`** (persisted locally and synced to S3):

```json
{
    "sweep_id": "best_of_k_20260319_143000",
    "started_at": "2026-03-19T14:30:00Z",
    "instance_id": "i-0abc123def456",
    "manifest": [
        {"benchmark": "aime", "k": 1, "seed": 42, "status": "completed", "test_score": 0.32},
        {"benchmark": "aime", "k": 1, "seed": 123, "status": "completed", "test_score": 0.30},
        {"benchmark": "aime", "k": 1, "seed": 456, "status": "running", "test_score": null},
        {"benchmark": "aime", "k": 3, "seed": 42, "status": "pending", "test_score": null}
    ],
    "last_updated": "2026-03-19T22:45:00Z",
    "total_api_cost_usd": 12.34
}
```

This file serves dual purpose: (1) the sweep runner reads it on startup to determine where to resume after interruption, and (2) it can be polled from a local machine to check progress without SSH.

```bash
# Check progress from local machine
aws s3 cp s3://gepa-mutations-results/sweep_progress.json - | python -m json.tool
```

---

## 8. Runbook

### Prerequisites

1. **AWS CLI configured** with credentials that have permissions for: EC2 (launch, terminate, describe), S3 (read/write to `gepa-mutations-results`), CloudWatch (put metrics), SNS (publish).

2. **OpenRouter API key** with sufficient balance (~$100 for full Tier 1+2 sweep).

3. **SSH key pair** registered in us-east-1 (e.g., `gepa-mutations-key`).

4. **Security group** allowing SSH (port 22) from your IP and all outbound traffic (for OpenRouter API and S3).

5. **IAM instance profile** (`gepa-mutations-ec2-role`) with policies:
   - `AmazonS3FullAccess` (scoped to `gepa-mutations-results` bucket)
   - `CloudWatchFullAccess` (for custom metrics)
   - `AmazonSNSFullAccess` (for alerts)
   - `ec2:TerminateInstances` and `ec2:CreateTags` (for self-termination)

### Step 1: Create IAM Role and Instance Profile

```bash
# Create the IAM role
aws iam create-role \
    --role-name gepa-mutations-ec2-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

# Attach policies
aws iam attach-role-policy \
    --role-name gepa-mutations-ec2-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
    --role-name gepa-mutations-ec2-role \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchFullAccess

# Create instance profile
aws iam create-instance-profile \
    --instance-profile-name gepa-mutations-ec2-profile

aws iam add-role-to-instance-profile \
    --instance-profile-name gepa-mutations-ec2-profile \
    --role-name gepa-mutations-ec2-role
```

### Step 2: Create S3 Bucket

```bash
aws s3 mb s3://gepa-mutations-results --region us-east-1

# Enable versioning (protects against accidental overwrites)
aws s3api put-bucket-versioning \
    --bucket gepa-mutations-results \
    --versioning-configuration Status=Enabled

# Set lifecycle rule: transition old versions to Glacier after 30 days
aws s3api put-bucket-lifecycle-configuration \
    --bucket gepa-mutations-results \
    --lifecycle-configuration '{
        "Rules": [{
            "ID": "archive-old-versions",
            "Status": "Enabled",
            "NoncurrentVersionTransitions": [{
                "NoncurrentDays": 30,
                "StorageClass": "GLACIER"
            }],
            "NoncurrentVersionExpiration": {"NoncurrentDays": 90}
        }]
    }'
```

### Step 3: Create SNS Topic

```bash
aws sns create-topic --name gepa-mutations-alerts --region us-east-1
# Note the TopicArn from the output

# Subscribe your email
aws sns subscribe \
    --topic-arn arn:aws:sns:us-east-1:ACCOUNT_ID:gepa-mutations-alerts \
    --protocol email \
    --notification-endpoint your-email@example.com
```

### Step 4: Launch EC2 Spot Instance

```bash
# User data script (base64-encoded, runs on first boot)
cat > /tmp/userdata.sh << 'USERDATA'
#!/bin/bash
set -euo pipefail

# System setup
yum update -y
yum install -y git python3.12 python3.12-pip

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Clone the project
cd /home/ec2-user
git clone https://github.com/YOUR_ORG/gepa-mutations.git
cd gepa-mutations

# Clone GEPA at v0.1.1 (required for local install)
git clone https://github.com/gepa-ai/gepa.git
cd gepa && git checkout v0.1.1 && cd ..
# Remove gepa's own venv/lock to avoid uv resolution conflicts
rm -rf gepa/.venv gepa/uv.lock

# Install dependencies
uv sync

# Write environment variables
cat > .env << ENV
OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
ENV

# Download any existing sweep progress from S3
aws s3 cp s3://gepa-mutations-results/sweep_progress.json sweep_progress.json 2>/dev/null || true

# Start the sweep (nohup so it survives SSH disconnect)
nohup uv run python -m gepa_mutations.experiments.best_of_k.sweep \
    --tiers 1,2 \
    --resume \
    > sweep.log 2>&1 &

echo "Sweep started. PID: $!"
USERDATA

# Launch spot instance
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.small \
    --key-name gepa-mutations-key \
    --security-group-ids sg-XXXXXXXX \
    --iam-instance-profile Name=gepa-mutations-ec2-profile \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=gepa-best-of-k-sweep},{Key=Project,Value=gepa-mutations}]' \
    --user-data file:///tmp/userdata.sh \
    --region us-east-1
```

**Important:** Replace `${OPENROUTER_API_KEY}`, `${TELEGRAM_BOT_TOKEN}`, `${TELEGRAM_CHAT_ID}` with actual values. For production use, store secrets in AWS Secrets Manager and fetch them in the user-data script.

**Spot configuration note:** Using `"InstanceInterruptionBehavior":"stop"` instead of `"terminate"` means the instance is stopped (not destroyed) on interruption. The EBS volume persists, so on relaunch the checkpoint is already on disk without needing an S3 download. This is cheaper and faster than terminate+relaunch, but costs $0.08/GB-month for the idle EBS volume.

### Step 5: Monitor Progress

```bash
# SSH into the instance
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=gepa-best-of-k-sweep" "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].InstanceId" --output text)

INSTANCE_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

ssh -i ~/.ssh/gepa-mutations-key.pem ec2-user@$INSTANCE_IP

# On the instance:
tail -f /home/ec2-user/gepa-mutations/sweep.log
```

```bash
# Check progress from local machine (no SSH needed)
aws s3 cp s3://gepa-mutations-results/sweep_progress.json - | python3 -m json.tool

# Check CloudWatch metrics
aws cloudwatch get-metric-statistics \
    --namespace GEPAMutations/BestOfK \
    --metric-name SweepProgress \
    --start-time $(date -u -v-1H +%Y-%m-%dT%H:%M:%S) \
    --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
    --period 300 \
    --statistics Maximum \
    --region us-east-1
```

### Step 6: Collect Results

```bash
# After sweep completes (you'll get a Telegram/SNS notification)

# Download all results
aws s3 sync s3://gepa-mutations-results/runs/ ./results/runs/

# Quick summary
python3 -c "
import json
from pathlib import Path

for result_file in sorted(Path('results/runs').rglob('result.json')):
    r = json.loads(result_file.read_text())
    parts = result_file.parts
    # parts: results/runs/{benchmark}/{method}/{seed}/result.json
    benchmark, method, seed = parts[-4], parts[-3], parts[-2]
    print(f'{benchmark:12s} {method:18s} seed={seed:>4s}  test={r[\"test_score\"]*100:6.2f}%')
"
```

### Step 7: Post-Sweep Analysis

```bash
# Download sweep progress for cost accounting
aws s3 cp s3://gepa-mutations-results/sweep_progress.json ./results/

# Generate comparison tables (uses existing analysis module)
uv run python -c "
from gepa_mutations.analysis.statistics import compare_methods
from gepa_mutations.storage.local import load_result

# Load results for K=1 vs K=3 on AIME
k1_scores = [load_result('aime', seed, 'best_of_k_K1')['test_score'] for seed in [42, 123, 456]]
k3_scores = [load_result('aime', seed, 'best_of_k_K3')['test_score'] for seed in [42, 123, 456]]

print('AIME K=1 scores:', k1_scores)
print('AIME K=3 scores:', k3_scores)
# ... run statistical tests from analysis/statistics.py
"
```

### Step 8: Cleanup

```bash
# Terminate the instance (if it didn't self-terminate)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

# Verify no orphaned instances
aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=gepa-mutations" "Name=instance-state-name,Values=running" \
    --query "Reservations[].Instances[].{Id:InstanceId,Type:InstanceType,LaunchTime:LaunchTime}" \
    --output table

# Check final S3 storage usage
aws s3 ls s3://gepa-mutations-results/ --recursive --summarize | tail -2
```

### Emergency Procedures

**Runaway costs:**
```bash
# Immediately terminate all project instances
aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=gepa-mutations" "Name=instance-state-name,Values=running" \
    --query "Reservations[].Instances[].InstanceId" --output text \
    | xargs -r aws ec2 terminate-instances --instance-ids
```

**Corrupted checkpoint:**
```bash
# Delete the checkpoint for a specific run and restart from scratch
aws s3 rm s3://gepa-mutations-results/runs/aime/best_of_k_K3/42/gepa_state/ --recursive

# Re-run just that one experiment
ssh ec2-user@$INSTANCE_IP
cd gepa-mutations
uv run python -m gepa_mutations.experiments.best_of_k.runner \
    --benchmark aime --seed 42 --k 3
```

**Instance won't start (spot capacity):**
```bash
# Fall back to on-demand (costs ~$4 more for the full sweep)
# Modify the run-instances command: remove --instance-market-options entirely
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.small \
    --key-name gepa-mutations-key \
    --security-group-ids sg-XXXXXXXX \
    --iam-instance-profile Name=gepa-mutations-ec2-profile \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=gepa-best-of-k-sweep},{Key=Project,Value=gepa-mutations}]' \
    --user-data file:///tmp/userdata.sh \
    --region us-east-1
```

---

## Appendix: Cost Summary Table

| Scenario | API Cost | EC2 Cost | Total | Runs |
|----------|----------|----------|-------|------|
| H1 falsified (Tier 1 Phase A only) | ~$22 | ~$0.50 | **~$22.50** | 12 |
| H1 confirmed, H2 falsified (Tier 1 + 2) | ~$79 | ~$1.93 | **~$81** | 36 |
| Full sweep (all tiers) | ~$99 | ~$2.40 | **~$101** | 48 |
| Worst case (full sweep + 5 seed expansion on 2 benchmarks) | ~$115 | ~$2.80 | **~$118** | 60 |
