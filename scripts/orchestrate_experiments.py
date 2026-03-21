#!/usr/bin/env python3
"""Orchestrate 90 GEPA mutation experiments across EC2 spot instances.

6 methods x 3 benchmarks x 5 seeds = 90 runs.
Phases are benchmark-based (hotpotqa=1, pupa=2, aime=3), all eligible immediately.

Each instance writes per-experiment status files to S3 (no shared manifest race
condition). The orchestrator scans these files to track completion.

Usage:
    # Pre-flight checks (spot quota, manifest, security group)
    python scripts/orchestrate_experiments.py --preflight

    # Check spot vCPU quota
    python scripts/orchestrate_experiments.py --check-quota

    # One-time: generate and upload the experiment manifest
    python scripts/orchestrate_experiments.py --generate-manifest

    # Launch the next batch of eligible experiments
    python scripts/orchestrate_experiments.py --launch

    # Print current status of all experiments
    python scripts/orchestrate_experiments.py --status

    # Poll loop: launch batches and monitor until everything completes
    python scripts/orchestrate_experiments.py --poll

    # All-in-one: generate manifest then poll
    python scripts/orchestrate_experiments.py --auto

    # Retry all failed experiments (reset status to pending)
    python scripts/orchestrate_experiments.py --retry-failed
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import boto3
from botocore.exceptions import ClientError


# ============================================================================
# Telegram helper (for orchestrator-side notifications)
# ============================================================================

def _send_telegram(message: str) -> None:
    """Send a Telegram message using the bot token from .env or environment.

    Silent on any failure -- notifications are best-effort.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    # Fall back to .env file in the project root
    if not token or not chat_id:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TELEGRAM_BOT_TOKEN="):
                        token = line.split("=", 1)[1]
                    elif line.startswith("TELEGRAM_CHAT_ID="):
                        chat_id = line.split("=", 1)[1]

    if not token or not chat_id:
        return

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
        }).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # Best-effort

# ---------------------------------------------------------------------------
# Configuration — update REPO_URL before first use
# ---------------------------------------------------------------------------
REGION = "us-east-1"
PROJECT = "gepa-mutations"
BUCKET = "gepa-mutations-results"
INSTANCE_TYPE = "t3.medium"
AMI_ID = "ami-0c421724a94bba6d6"  # Amazon Linux 2023 us-east-1
INSTANCE_PROFILE = "gepa-mutations-ec2-profile"
SG_NAME = "gepa-mutations-sg"
MAX_CONCURRENT = 10
REPO_URL = "https://github.com/ashwinchidambaram/gepa-mutations.git"  # <-- UPDATE
BRANCH = "master"

SEEDS = [42, 123, 456, 789, 1024]
BENCHMARKS = ["hotpotqa", "pupa", "aime"]

# Expected max durations per benchmark (hours) — for stale detection
EXPECTED_DURATION = {"hotpotqa": 5, "pupa": 4, "aime": 14}
STALE_LAUNCHED_HOURS = 0.5  # Mark "launched" as stale after 30 min without heartbeat


# ============================================================================
# Manifest helpers
# ============================================================================

METHODS = [
    ("best_of_k_K1", 1),
    ("best_of_k_K3", 3),
    ("best_of_k_K5", 5),
    ("contrastive_reflection", 0),
    ("failure_stratified_k_K3", 3),
    ("failure_stratified_k_K5", 5),
]

# Benchmark → phase mapping (no blocking between phases)
BENCHMARK_PHASE = {"hotpotqa": 1, "pupa": 2, "aime": 3}


def generate_manifest() -> dict:
    """Build the full experiment matrix as a JSON manifest.

    6 methods × 3 benchmarks × 5 seeds = 90 runs.
    Phases are benchmark-based (hotpotqa=1, pupa=2, aime=3), all eligible immediately.
    """
    experiments: list[dict] = []

    for bm in BENCHMARKS:
        phase = BENCHMARK_PHASE[bm]
        for method, k in METHODS:
            for seed in SEEDS:
                eid = f"{method}__{bm}__{seed}"
                experiments.append({
                    "id": eid,
                    "phase": phase,
                    "method": method,
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
    }

    print(f"Generated manifest with {len(experiments)} experiments:")
    for phase in sorted({e["phase"] for e in experiments}):
        count = sum(1 for e in experiments if e["phase"] == phase)
        bm = [b for b, p in BENCHMARK_PHASE.items() if p == phase][0]
        print(f"  Phase {phase} ({bm}): {count} experiments")

    return manifest


def upload_manifest(manifest: dict) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    s3.put_object(
        Bucket=BUCKET,
        Key="status/manifest.json",
        Body=json.dumps(manifest, indent=2),
        ContentType="application/json",
    )
    print(f"Manifest uploaded to s3://{BUCKET}/status/manifest.json")


def download_manifest() -> dict:
    s3 = boto3.client("s3", region_name=REGION)
    try:
        obj = s3.get_object(Bucket=BUCKET, Key="status/manifest.json")
        return json.loads(obj["Body"].read())
    except ClientError:
        print("ERROR: No manifest found in S3. Run with --generate-manifest first.")
        sys.exit(1)


def _scan_status_files(manifest: dict) -> bool:
    """Scan per-experiment status files from S3 and update manifest.

    Fix 8: Each instance writes its own status file instead of updating a
    shared manifest.json, avoiding race conditions.
    """
    s3 = boto3.client("s3", region_name=REGION)
    changed = False

    try:
        paginator = s3.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=BUCKET, Prefix="status/"):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
    except ClientError:
        return False

    # Build lookup: which experiments have started/done files
    started_ids: set[str] = set()
    done_keys: dict[str, str] = {}
    for key in keys:
        basename = key.rsplit("/", 1)[-1]
        if basename.endswith("_started.json"):
            exp_id = basename.replace("_started.json", "")
            started_ids.add(exp_id)
        elif basename.endswith("_done.json"):
            exp_id = basename.replace("_done.json", "")
            done_keys[exp_id] = key

    for exp in manifest["experiments"]:
        eid = exp["id"]

        # Done status takes priority
        if eid in done_keys and exp["status"] not in ("completed", "failed"):
            try:
                obj = s3.get_object(Bucket=BUCKET, Key=done_keys[eid])
                status = json.loads(obj["Body"].read())
                exp["status"] = status.get("status", "completed")
                exp["completed_at"] = status.get("completed_at")
                exp["instance_id"] = status.get("instance_id")
                exp["test_score"] = status.get("test_score")
                changed = True
            except (ClientError, json.JSONDecodeError):
                pass

        # Started status (only if not already done or running)
        elif eid in started_ids and exp["status"] in ("pending", "launched"):
            exp["status"] = "running"
            changed = True

    return changed


# ============================================================================
# Phase gating
# ============================================================================

def can_start_phase(_phase: int, _manifest: dict) -> bool:
    """All phases eligible immediately (no blocking between benchmarks)."""
    return True


# ============================================================================
# EC2 helpers
# ============================================================================

def _running_instance_count() -> int:
    ec2 = boto3.client("ec2", region_name=REGION)
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Project", "Values": [PROJECT]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ],
    )
    return sum(len(r["Instances"]) for r in resp["Reservations"])


def _sg_id() -> str | None:
    ec2 = boto3.client("ec2", region_name=REGION)
    sgs = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [SG_NAME]}],
    )
    if not sgs["SecurityGroups"]:
        return None
    return sgs["SecurityGroups"][0]["GroupId"]


def _result_exists_in_s3(exp: dict) -> bool:
    s3 = boto3.client("s3", region_name=REGION)
    key = f"runs/{exp['benchmark']}/{exp['method']}/{exp['seed']}/result.json"
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError:
        return False


# ============================================================================
# User-data script builder
# ============================================================================

def _user_data(exp: dict) -> str:
    """Return the full cloud-init bash script for *exp*."""
    return f"""#!/bin/bash
LOG=/var/log/gepa-experiment.log
log() {{ echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }}

# Fix 7: IMDSv2 token TTL increased to 6 hours
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)

# Fix 5: Hard kill safety net — 18 hours max lifetime
shutdown -h +1080 &

# ------- Telegram notification helper -------
send_telegram() {{
    local MSG="$1"
    if [ -f /root/gepa-mutations/.env ]; then
        local TG_TOKEN=$(grep '^TELEGRAM_BOT_TOKEN=' /root/gepa-mutations/.env | cut -d= -f2-)
        local TG_CHAT=$(grep '^TELEGRAM_CHAT_ID=' /root/gepa-mutations/.env | cut -d= -f2-)
        if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT" ]; then
            curl -s -X POST "https://api.telegram.org/bot$TG_TOKEN/sendMessage" \
                -d "chat_id=$TG_CHAT" \
                --data-urlencode "text=$MSG" \
                -d "parse_mode=HTML" > /dev/null 2>&1 || true
        fi
    fi
}}

# ------- Fix 3+4: periodic checkpoint sync + heartbeat every 30 min -------
periodic_sync() {{
    while true; do
        sleep 1800
        if [ -f /tmp/gepa_exit_code ]; then
            break
        fi
        local UPTIME_HRS=$(awk '{{printf "%.1f", $1/3600}}' /proc/uptime)
        local TAIL_LOG=$(tail -5 $LOG 2>/dev/null | head -3)

        # Fix 3: Sync checkpoint to S3
        if [ -d /root/gepa-mutations/runs/{exp['benchmark']}/{exp['method']}/{exp['seed']}/gepa_state ]; then
            aws s3 sync /root/gepa-mutations/runs/{exp['benchmark']}/{exp['method']}/{exp['seed']}/gepa_state/ \
                s3://{BUCKET}/runs/{exp['benchmark']}/{exp['method']}/{exp['seed']}/gepa_state/ \
                --region us-east-1 >> $LOG 2>&1 || true
        fi

        # Fix 4: Write heartbeat to S3
        python3 -c "
import json, datetime, boto3
hb = {{
    'timestamp': datetime.datetime.utcnow().isoformat(),
    'experiment_id': '{exp['id']}',
    'instance_id': '$INSTANCE_ID',
    'uptime_hours': float('$UPTIME_HRS'),
    'experiment_alive': True,
}}
boto3.client('s3', region_name='us-east-1').put_object(
    Bucket='{BUCKET}',
    Key='logs/$INSTANCE_ID/heartbeat.json',
    Body=json.dumps(hb),
    ContentType='application/json',
)" 2>/dev/null || true

        send_telegram "<b>Progress Update</b>
Experiment: <code>{exp['id']}</code>
Instance: <code>$INSTANCE_ID</code>
Uptime: ${{UPTIME_HRS}}h
Recent log:
<pre>${{TAIL_LOG}}</pre>"
    done
}}
periodic_sync &
MONITOR_PID=$!

cleanup() {{
    log "Cleanup: uploading and terminating"
    kill $MONITOR_PID 2>/dev/null || true
    aws s3 cp $LOG s3://{BUCKET}/runs/{exp['benchmark']}/{exp['method']}/{exp['seed']}/ec2.log --region us-east-1 || true
    [ -d /root/gepa-mutations/runs ] && aws s3 sync /root/gepa-mutations/runs/ s3://{BUCKET}/runs/ --region us-east-1 || true

    # Fix 8: Write per-experiment status file (no shared manifest race condition)
    python3 -c "
import json, boto3, datetime, os
exit_code = int(open('/tmp/gepa_exit_code').read().strip()) if os.path.exists('/tmp/gepa_exit_code') else 1
status = {{
    'experiment_id': '{exp['id']}',
    'status': 'completed' if exit_code == 0 else 'failed',
    'instance_id': '$INSTANCE_ID',
    'completed_at': datetime.datetime.utcnow().isoformat(),
    'exit_code': exit_code,
    'benchmark': '{exp['benchmark']}',
    'method': '{exp['method']}',
    'seed': {exp['seed']},
}}
try:
    import glob
    files = glob.glob('/root/gepa-mutations/runs/{exp['benchmark']}/{exp['method']}/{exp['seed']}/result.json')
    if files and exit_code == 0:
        d = json.load(open(files[0]))
        status['test_score'] = d.get('test_score')
        status['wall_clock_seconds'] = d.get('wall_clock_seconds')
except Exception:
    pass
boto3.client('s3', region_name='us-east-1').put_object(
    Bucket='{BUCKET}',
    Key='status/{exp['id']}_done.json',
    Body=json.dumps(status, indent=2),
    ContentType='application/json',
)" 2>/dev/null || true

    # Final heartbeat (experiment_alive=False)
    python3 -c "
import json, datetime, boto3
hb = {{
    'timestamp': datetime.datetime.utcnow().isoformat(),
    'experiment_id': '{exp['id']}',
    'instance_id': '$INSTANCE_ID',
    'experiment_alive': False,
}}
boto3.client('s3', region_name='us-east-1').put_object(
    Bucket='{BUCKET}',
    Key='logs/$INSTANCE_ID/heartbeat.json',
    Body=json.dumps(hb),
    ContentType='application/json',
)" 2>/dev/null || true

    # ------- send completion notification -------
    EXIT_CODE=$(cat /tmp/gepa_exit_code 2>/dev/null || echo "1")
    if [ "$EXIT_CODE" = "0" ]; then
        TEST_SCORE=$(python3 -c "
import json, glob
files = glob.glob('/root/gepa-mutations/runs/{exp['benchmark']}/{exp['method']}/{exp['seed']}/result.json')
if files:
    d = json.load(open(files[0]))
    print(f\"{{d.get('test_score', 0)*100:.2f}}%\")
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
        WALL_CLOCK=$(python3 -c "
import json, glob
files = glob.glob('/root/gepa-mutations/runs/{exp['benchmark']}/{exp['method']}/{exp['seed']}/result.json')
if files:
    d = json.load(open(files[0]))
    wc = d.get('wall_clock_seconds', 0)
    h, m = int(wc//3600), int((wc%3600)//60)
    print(f'{{h}}h {{m}}m')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
        send_telegram "Experiment complete
Method: <code>{exp['method']}</code>
Benchmark: <code>{exp['benchmark']}</code>
Seed: <code>{exp['seed']}</code>
Test score: <b>${{TEST_SCORE}}</b>
Wall clock: ${{WALL_CLOCK}}
Instance: <code>${{INSTANCE_ID}}</code>"
    else
        send_telegram "Experiment FAILED
Method: <code>{exp['method']}</code>
Benchmark: <code>{exp['benchmark']}</code>
Seed: <code>{exp['seed']}</code>
Exit code: ${{EXIT_CODE}}
Instance: <code>${{INSTANCE_ID}}</code>
Last log line: <pre>$(tail -1 $LOG 2>/dev/null)</pre>"
    fi
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1 || true
}}
trap cleanup EXIT

log "=== GEPA Experiment: {exp['id']} ==="
yum install -y git >> $LOG 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh >> $LOG 2>&1
export PATH="/root/.local/bin:$PATH"

cd /root
git clone --depth 1 --branch {BRANCH} {REPO_URL} >> $LOG 2>&1
cd gepa-mutations
git clone --depth 1 --branch v0.1.1 https://github.com/gepa-ai/gepa.git gepa >> $LOG 2>&1
rm -rf gepa/.venv gepa/uv.lock

aws s3 cp s3://{BUCKET}/config/.env .env --region us-east-1 >> $LOG 2>&1
uv sync >> $LOG 2>&1

# Fix 2: Download existing checkpoint if resuming after spot interruption
aws s3 sync s3://{BUCKET}/runs/{exp['benchmark']}/{exp['method']}/{exp['seed']}/gepa_state/ \
    /root/gepa-mutations/runs/{exp['benchmark']}/{exp['method']}/{exp['seed']}/gepa_state/ \
    --region us-east-1 >> $LOG 2>&1 || true

# Fix 8: Write per-experiment started status file
python3 -c "
import json, boto3, datetime
status = {{
    'experiment_id': '{exp['id']}',
    'status': 'running',
    'instance_id': '$INSTANCE_ID',
    'started_at': datetime.datetime.utcnow().isoformat(),
    'benchmark': '{exp['benchmark']}',
    'method': '{exp['method']}',
    'seed': {exp['seed']},
}}
boto3.client('s3', region_name='us-east-1').put_object(
    Bucket='{BUCKET}',
    Key='status/{exp['id']}_started.json',
    Body=json.dumps(status, indent=2),
    ContentType='application/json',
)" 2>/dev/null || true

# ------- send start notification -------
send_telegram "Experiment started
Method: <code>{exp['method']}</code>
Benchmark: <code>{exp['benchmark']}</code>
Seed: <code>{exp['seed']}</code>
Phase: {exp['phase']}
Instance: <code>${{INSTANCE_ID}}</code>"

# ------- run experiment -------
case "{exp['method']}" in
    best_of_k_K*)
        K=$(echo "{exp['method']}" | sed 's/best_of_k_K//')
        log "Starting: best_of_k K=$K benchmark={exp['benchmark']} seed={exp['seed']}"
        uv run python -c "
from best_of_k.runner import run_best_of_k
from gepa_mutations.base import MutationConfig
config = MutationConfig(
    mutation_name='{exp['method']}',
    benchmark='{exp['benchmark']}',
    seed={exp['seed']},
    mutation_candidates={exp.get('k', 1)},
)
run_best_of_k(config=config, k={exp.get('k', 1)})
" >> $LOG 2>&1
        ;;
    contrastive_reflection)
        log "Starting: contrastive_reflection benchmark={exp['benchmark']} seed={exp['seed']}"
        uv run python -c "
from contrastive_reflection.runner import run_contrastive_reflection
run_contrastive_reflection(benchmark='{exp['benchmark']}', seed={exp['seed']})
" >> $LOG 2>&1
        ;;
    failure_stratified_k_K*)
        K=$(echo "{exp['method']}" | sed 's/failure_stratified_k_K//')
        log "Starting: failure_stratified_k K=$K benchmark={exp['benchmark']} seed={exp['seed']}"
        uv run python -c "
from failure_stratified_k.runner import run_failure_stratified_k
from gepa_mutations.base import MutationConfig
config = MutationConfig(
    mutation_name='{exp['method']}',
    benchmark='{exp['benchmark']}',
    seed={exp['seed']},
    mutation_candidates={exp.get('k', 3)},
)
run_failure_stratified_k(config=config, k={exp.get('k', 3)})
" >> $LOG 2>&1
        ;;
    *)
        log "ERROR: Unknown method {exp['method']}"
        exit 1
        ;;
esac

echo $? > /tmp/gepa_exit_code
log "Exit code: $(cat /tmp/gepa_exit_code)"
"""


# ============================================================================
# Launch helpers
# ============================================================================

def launch_experiment(exp: dict, sg: str, use_spot: bool = True) -> str | None:
    """Launch a single EC2 instance for *exp*. Returns instance-id or None."""
    ec2 = boto3.client("ec2", region_name=REGION)
    params: dict = {
        "ImageId": AMI_ID,
        "InstanceType": INSTANCE_TYPE,
        "MinCount": 1,
        "MaxCount": 1,
        "IamInstanceProfile": {"Name": INSTANCE_PROFILE},
        "SecurityGroupIds": [sg],
        "UserData": _user_data(exp),
        "MetadataOptions": {
            "HttpTokens": "required",
            "InstanceMetadataTags": "enabled",
        },
        "TagSpecifications": [
            {
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
            }
        ],
    }
    if use_spot:
        params["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }

    try:
        resp = ec2.run_instances(**params)
        iid = resp["Instances"][0]["InstanceId"]
        label = "spot" if use_spot else "on-demand"
        print(f"  Launched {iid} ({label}) -> {exp['id']}")
        return iid
    except ClientError as exc:
        print(f"  ERROR launching {exp['id']}: {exc}")
        return None


# ============================================================================
# Stale-instance recovery
# ============================================================================

def _recover_stale(manifest: dict) -> bool:
    """Reset experiments stuck in 'launched' with no running instance."""
    ec2 = boto3.client("ec2", region_name=REGION)
    changed = False
    now = datetime.datetime.utcnow()

    for exp in manifest["experiments"]:
        if exp["status"] not in ("launched", "running"):
            continue
        if not exp.get("started_at"):
            continue
        started = datetime.datetime.fromisoformat(exp["started_at"])
        age_hrs = (now - started).total_seconds() / 3600

        # For "launched" entries that never moved to "running"
        if exp["status"] == "launched" and age_hrs > STALE_LAUNCHED_HOURS:
            exp["status"] = "pending"
            exp["instance_id"] = None
            exp["started_at"] = None
            print(f"  Reset stale-launched: {exp['id']}")
            changed = True
            continue

        # For "running" entries that exceed 2x expected duration
        max_hrs = EXPECTED_DURATION.get(exp["benchmark"], 14) * 2
        if exp["status"] == "running" and age_hrs > max_hrs:
            # Check if instance still exists
            if exp.get("instance_id"):
                try:
                    resp = ec2.describe_instances(InstanceIds=[exp["instance_id"]])
                    state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
                    if state in ("terminated", "shutting-down"):
                        exp["status"] = "pending"
                        exp["instance_id"] = None
                        exp["started_at"] = None
                        print(f"  Reset terminated-but-running: {exp['id']}")
                        changed = True
                except (ClientError, IndexError, KeyError):
                    exp["status"] = "pending"
                    exp["instance_id"] = None
                    exp["started_at"] = None
                    print(f"  Reset orphaned: {exp['id']}")
                    changed = True

    return changed


# ============================================================================
# Core orchestration
# ============================================================================

def launch_batch(manifest: dict) -> dict:
    """Launch the next batch of eligible experiments."""
    running = _running_instance_count()
    available = MAX_CONCURRENT - running

    if available <= 0:
        print(f"At capacity ({running}/{MAX_CONCURRENT}). Waiting.")
        return manifest

    sg = _sg_id()
    if sg is None:
        print("ERROR: Security group not found. Run aws_setup.py first.")
        return manifest

    print(f"Slots: {available} available ({running} running)")

    # Recover any stale entries first
    if _recover_stale(manifest):
        upload_manifest(manifest)

    # Sync status files from S3 before deciding what to launch
    if _scan_status_files(manifest):
        upload_manifest(manifest)

    launched = 0
    # Sort: benchmark (hotpotqa → pupa → aime), then method, then seed
    bm_order = {"hotpotqa": 0, "pupa": 1, "aime": 2}
    pending = sorted(
        [e for e in manifest["experiments"] if e["status"] == "pending"],
        key=lambda e: (bm_order.get(e["benchmark"], 9), e["method"], e["seed"]),
    )

    for exp in pending:
        if launched >= available:
            break

        # Idempotency: skip if result already in S3
        if _result_exists_in_s3(exp):
            print(f"  Skip (result exists): {exp['id']}")
            exp["status"] = "completed"
            continue

        # All spot instances
        iid = launch_experiment(exp, sg, use_spot=True)
        if iid:
            exp["status"] = "launched"
            exp["instance_id"] = iid
            exp["started_at"] = datetime.datetime.utcnow().isoformat()
            launched += 1
            time.sleep(2)  # Gentle pacing

    upload_manifest(manifest)
    print(f"Launched {launched} experiments this batch.")
    return manifest


def print_status(manifest: dict) -> None:
    """Pretty-print the current experiment matrix status."""
    print(f"\n{'='*70}")
    print("  GEPA Experiment Orchestrator Status")
    print(f"{'='*70}")

    for phase in sorted({e["phase"] for e in manifest["experiments"]}):
        phase_exps = [e for e in manifest["experiments"] if e["phase"] == phase]
        counts: dict[str, int] = {}
        for e in phase_exps:
            counts[e["status"]] = counts.get(e["status"], 0) + 1
        total = len(phase_exps)
        gate = "OPEN" if can_start_phase(phase, manifest) else "BLOCKED"

        bm_label = {v: k for k, v in BENCHMARK_PHASE.items()}.get(phase, "?")
        print(f"\n  Phase {phase} — {bm_label} [{gate}] ({total} total):")
        for status in ("completed", "running", "launched", "pending", "failed"):
            n = counts.get(status, 0)
            if n > 0:
                print(f"    {status:12s}: {n}")

    total = len(manifest["experiments"])
    done = sum(1 for e in manifest["experiments"] if e["status"] == "completed")
    failed = sum(1 for e in manifest["experiments"] if e["status"] == "failed")
    active = sum(
        1 for e in manifest["experiments"]
        if e["status"] in ("launched", "running")
    )
    pend = sum(1 for e in manifest["experiments"] if e["status"] == "pending")
    print(f"\n  Overall: {done} completed, {active} active, "
          f"{pend} pending, {failed} failed  ({total} total)")
    print(f"{'='*70}\n")


def _manifest_summary_text(manifest: dict) -> str:
    """Build a compact HTML summary of manifest status for Telegram."""
    exps = manifest["experiments"]
    total = len(exps)
    done = sum(1 for e in exps if e["status"] == "completed")
    failed = sum(1 for e in exps if e["status"] == "failed")
    active = sum(1 for e in exps if e["status"] in ("launched", "running"))
    pend = sum(1 for e in exps if e["status"] == "pending")

    lines = [
        "<b>GEPA Orchestrator Status</b>",
        f"Completed: {done}/{total}  |  Active: {active}",
        f"Pending: {pend}  |  Failed: {failed}",
    ]

    # Per-phase breakdown
    for phase in sorted({e["phase"] for e in exps}):
        phase_exps = [e for e in exps if e["phase"] == phase]
        p_done = sum(1 for e in phase_exps if e["status"] == "completed")
        p_active = sum(1 for e in phase_exps if e["status"] in ("launched", "running"))
        p_fail = sum(1 for e in phase_exps if e["status"] == "failed")
        bm_label = {v: k for k, v in BENCHMARK_PHASE.items()}.get(phase, "?")
        lines.append(f"  P{phase}({bm_label}): {p_done}/{len(phase_exps)} done, {p_active} active, {p_fail} fail")

    return "\n".join(lines)


def poll_loop(interval: int = 900) -> None:
    """Continuously launch batches until all experiments finish."""
    _send_telegram("<b>GEPA Orchestrator started</b>\nPolling every "
                   f"{interval}s for experiment completion.")

    while True:
        manifest = download_manifest()

        # Sync per-experiment status files from S3
        if _scan_status_files(manifest):
            upload_manifest(manifest)

        print_status(manifest)

        all_done = all(
            e["status"] in ("completed", "failed")
            for e in manifest["experiments"]
        )
        if all_done:
            print("All experiments complete.")
            summary = _manifest_summary_text(manifest)
            _send_telegram(f"{summary}\n\n<b>All experiments finished.</b>")
            break

        manifest = launch_batch(manifest)

        # Send Telegram status update each poll cycle
        _send_telegram(_manifest_summary_text(manifest))

        print(f"Next check in {interval}s ...")
        time.sleep(interval)


def retry_failed() -> None:
    """Reset all failed experiments to pending for relaunch."""
    manifest = download_manifest()
    count = 0
    for exp in manifest["experiments"]:
        if exp["status"] == "failed":
            exp["status"] = "pending"
            exp["instance_id"] = None
            exp["started_at"] = None
            exp["completed_at"] = None
            exp["error"] = None
            count += 1
    upload_manifest(manifest)
    print(f"Reset {count} failed experiments to pending.")


def check_spot_quota() -> None:
    """Fix 6: Check spot vCPU quota before launching."""
    try:
        sq = boto3.client("service-quotas", region_name=REGION)
        resp = sq.get_service_quota(
            ServiceCode="ec2",
            QuotaCode="L-34B43A08",  # All Standard Spot Instance Requests
        )
        quota = resp["Quota"]["Value"]
        print(f"Spot vCPU quota: {int(quota)}")
        if quota < 20:
            print(f"  WARNING: Quota ({int(quota)}) < 20 vCPUs. "
                  f"MAX_CONCURRENT should be <= {int(quota // 2)}.")
            print("  Request increase: aws service-quotas request-service-quota-increase "
                  "--service-code ec2 --quota-code L-34B43A08 --desired-value 40 "
                  f"--region {REGION}")
        else:
            print(f"  OK: Sufficient for MAX_CONCURRENT={MAX_CONCURRENT} "
                  f"(2 vCPUs per t3.medium = {MAX_CONCURRENT * 2} needed)")
    except ClientError as exc:
        print(f"  Could not check quota: {exc}")
        print("  Run manually: aws service-quotas get-service-quota "
              f"--service-code ec2 --quota-code L-34B43A08 --region {REGION}")


def preflight_check() -> None:
    """Fix 10: Run K=1 equivalence validation and other pre-launch checks."""
    print("=" * 60)
    print("  Pre-flight checks")
    print("=" * 60)

    # 1. Check spot quota
    print("\n1. Checking spot vCPU quota...")
    check_spot_quota()

    # 2. Verify manifest generation
    print("\n2. Generating manifest...")
    m = generate_manifest()
    assert len(m["experiments"]) == 90, (
        f"Expected 90 experiments, got {len(m['experiments'])}"
    )
    print(f"  OK: {len(m['experiments'])} experiments")

    # 3. Verify security group exists
    print("\n3. Checking security group...")
    sg = _sg_id()
    if sg:
        print(f"  OK: {SG_NAME} -> {sg}")
    else:
        print(f"  FAIL: Security group '{SG_NAME}' not found")

    # 4. Verify S3 bucket accessible
    print("\n4. Checking S3 bucket...")
    try:
        s3 = boto3.client("s3", region_name=REGION)
        s3.head_bucket(Bucket=BUCKET)
        print(f"  OK: s3://{BUCKET}")
    except ClientError as exc:
        print(f"  FAIL: {exc}")

    # 5. Fix 11: Verify seed prompts exist for all benchmarks
    print("\n5. Verifying seed prompts for experiment benchmarks...")
    try:
        from gepa_mutations.runner.experiment import BENCHMARK_SEED_PROMPTS
        missing = []
        for bm in BENCHMARKS:
            prompt = BENCHMARK_SEED_PROMPTS.get(bm)
            if not prompt:
                missing.append(bm)
            else:
                print(f"  {bm}: \"{prompt[:60]}...\"")
        if missing:
            print(f"  WARNING: Missing seed prompts for: {missing}")
        else:
            print("  OK: All benchmarks have seed prompts")
    except ImportError as e:
        print(f"  SKIP: Could not import seed prompts ({e})")

    # 6. K=1 equivalence note
    print("\n6. K=1 equivalence validation:")
    print("  To validate K=1 matches vanilla GEPA, run:")
    print("    uv run python -c \"")
    print("    from best_of_k.runner import run_best_of_k")
    print("    from gepa_mutations.base import MutationConfig")
    print("    config = MutationConfig(mutation_name='best_of_k_K1', benchmark='hotpotqa', seed=42, mutation_candidates=1)")
    print("    run_best_of_k(config=config, k=1)\"")
    print("  Compare iteration-by-iteration val scores with:")
    print("    uv run gepa-mutations run hotpotqa --seed 42")

    print(f"\n{'=' * 60}")
    print("  Pre-flight complete")
    print(f"{'=' * 60}")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GEPA experiment orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate-manifest", action="store_true",
                       help="Generate and upload the experiment manifest")
    group.add_argument("--launch", action="store_true",
                       help="Launch the next batch of experiments")
    group.add_argument("--status", action="store_true",
                       help="Print current status")
    group.add_argument("--poll", action="store_true",
                       help="Poll loop until all experiments complete")
    group.add_argument("--auto", action="store_true",
                       help="Generate manifest then poll")
    group.add_argument("--retry-failed", action="store_true",
                       help="Reset all failed experiments to pending")
    group.add_argument("--preflight", action="store_true",
                       help="Run pre-flight checks (quota, manifest, K=1 validation)")
    group.add_argument("--check-quota", action="store_true",
                       help="Check spot vCPU quota")

    parser.add_argument("--interval", type=int, default=900,
                        help="Poll interval in seconds (default 900)")

    args = parser.parse_args()

    if args.generate_manifest:
        m = generate_manifest()
        upload_manifest(m)
    elif args.launch:
        m = download_manifest()
        launch_batch(m)
    elif args.status:
        m = download_manifest()
        print_status(m)
    elif args.retry_failed:
        retry_failed()
    elif args.preflight:
        preflight_check()
    elif args.check_quota:
        check_spot_quota()
    elif args.poll:
        poll_loop(interval=args.interval)
    elif args.auto:
        m = generate_manifest()
        upload_manifest(m)
        poll_loop(interval=args.interval)


if __name__ == "__main__":
    main()
