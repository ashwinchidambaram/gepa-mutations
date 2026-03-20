#!/usr/bin/env python3
"""Orchestrate ~90 GEPA mutation experiments across EC2 spot instances.

Manifest-driven job queue: all experiment definitions and status live in a
single JSON file on S3.  The orchestrator polls this manifest, launches EC2
instances for pending jobs (respecting phase gates and concurrency caps),
and detects completion/failure via heartbeat files and instance state.

Usage:
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
import sys
import time
from itertools import product

import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Configuration — update REPO_URL before first use
# ---------------------------------------------------------------------------
REGION = "us-east-1"
PROJECT = "gepa-mutations"
BUCKET = "gepa-mutations-results"
INSTANCE_TYPE = "t3.medium"
AMI_ID = "ami-0c421724a94bba6d6"  # Ubuntu 22.04 LTS us-east-1
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

def generate_manifest() -> dict:
    """Build the full experiment matrix as a JSON manifest."""
    experiments: list[dict] = []
    seen_ids: set[str] = set()

    def _add(phase: int, method: str, k: int, bm: str, seed: int) -> None:
        eid = f"{method}__{bm}__{seed}"
        if eid in seen_ids:
            return
        seen_ids.add(eid)
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

    # Phase 1 — validation (2 runs)
    _add(1, "best_of_k_K1", 1, "hotpotqa", 42)
    _add(1, "best_of_k_K3", 3, "hotpotqa", 42)

    # Phase 2 — best_of_k sweep (45 runs, minus 2 already in Phase 1)
    for k, bm, seed in product([1, 3, 5], BENCHMARKS, SEEDS):
        _add(2, f"best_of_k_K{k}", k, bm, seed)

    # Phase 3 — contrastive_reflection (15 runs)
    for bm, seed in product(BENCHMARKS, SEEDS):
        _add(3, "contrastive_reflection", 0, bm, seed)

    # Phase 4 — failure_stratified_k (30 runs)
    for k, bm, seed in product([3, 5], BENCHMARKS, SEEDS):
        _add(4, f"failure_stratified_k_K{k}", k, bm, seed)

    manifest = {
        "created_at": datetime.datetime.utcnow().isoformat(),
        "total_experiments": len(experiments),
        "experiments": experiments,
    }

    print(f"Generated manifest with {len(experiments)} experiments:")
    for phase in sorted({e["phase"] for e in experiments}):
        count = sum(1 for e in experiments if e["phase"] == phase)
        print(f"  Phase {phase}: {count} experiments")

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


# ============================================================================
# Phase gating
# ============================================================================

def can_start_phase(phase: int, manifest: dict) -> bool:
    exps = manifest["experiments"]
    if phase == 1:
        return True
    if phase in (2, 3):
        return all(
            e["status"] == "completed"
            for e in exps
            if e["phase"] == 1
        )
    if phase == 4:
        return all(
            e["status"] in ("completed", "failed")
            for e in exps
            if e["phase"] == 2
        )
    return False


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
set -euxo pipefail
exec > >(tee /var/log/gepa-experiment.log) 2>&1
echo "=== GEPA experiment {exp['id']} starting at $(date -u) ==="

# ------- parameters (injected at launch) -------
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

# ------- IMDSv2 token -------
IMDS_TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \\
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \\
    http://169.254.169.254/latest/meta-data/instance-id)

# ------- self-termination trap -------
cleanup() {{
    EXIT_CODE=$?
    echo "=== CLEANUP exit=$EXIT_CODE at $(date -u) ==="
    cd /home/ubuntu/gepa-mutations 2>/dev/null || true
    aws s3 sync runs/ "s3://$S3_BUCKET/runs/" --quiet 2>/dev/null || true
    aws s3 cp /var/log/gepa-experiment.log \\
        "s3://$S3_BUCKET/logs/$INSTANCE_ID/experiment.log" 2>/dev/null || true
    # update manifest
    python3 -c "
import json, boto3, datetime
s3 = boto3.client('s3', region_name='$REGION')
try:
    obj = s3.get_object(Bucket='$S3_BUCKET', Key='status/manifest.json')
    m = json.loads(obj['Body'].read())
    for e in m['experiments']:
        if e['id'] == '$EXPERIMENT_ID':
            e['status'] = 'completed' if $EXIT_CODE == 0 else 'failed'
            e['completed_at'] = datetime.datetime.utcnow().isoformat()
            if $EXIT_CODE != 0:
                e['error'] = 'exit $EXIT_CODE'
            break
    s3.put_object(Bucket='$S3_BUCKET', Key='status/manifest.json',
                  Body=json.dumps(m, indent=2))
except Exception as exc:
    print(f'manifest update failed: {{exc}}')
" 2>/dev/null || true
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" || true
}}
trap cleanup EXIT

# ------- heartbeat (every 5 min) + checkpoint sync -------
(
    while true; do
        sleep 300
        python3 -c "
import json, datetime, os, subprocess
hb = {{
    'instance_id': '$INSTANCE_ID',
    'experiment_id': '$EXPERIMENT_ID',
    'timestamp': datetime.datetime.utcnow().isoformat(),
    'uptime_seconds': float(open('/proc/uptime').read().split()[0]),
    'experiment_alive': subprocess.run(
        ['pgrep', '-f', 'best_of_k|contrastive_reflection|failure_stratified_k|gepa-mutations'],
        capture_output=True).returncode == 0,
}}
# parse latest metrics if available
import glob
for mf in glob.glob('/home/ubuntu/gepa-mutations/runs/**/metrics.json', recursive=True):
    try:
        with open(mf) as fh:
            md = json.load(fh)
            hb['iterations'] = md.get('total_iterations', 0)
            hb['metric_calls'] = md.get('total_metric_calls', 0)
            hb['acceptance_rate'] = md.get('acceptance_rate', 0)
    except: pass
print(json.dumps(hb))
" 2>/dev/null | aws s3 cp - "s3://$S3_BUCKET/logs/$INSTANCE_ID/heartbeat.json" --quiet 2>/dev/null || true
        # checkpoint sync
        cd /home/ubuntu/gepa-mutations 2>/dev/null && \\
            aws s3 sync runs/ "s3://$S3_BUCKET/runs/" --exclude "*.pyc" --quiet 2>/dev/null || true
    done
) &

# ------- spot interruption monitor -------
(
    while true; do
        HTTP=$(curl -s -o /dev/null -w "%{{http_code}}" \\
            -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \\
            http://169.254.169.254/latest/meta-data/spot/instance-action 2>/dev/null || echo 000)
        if [ "$HTTP" = "200" ]; then
            echo "SPOT INTERRUPTION at $(date -u)"
            cd /home/ubuntu/gepa-mutations 2>/dev/null && \\
                aws s3 sync runs/ "s3://$S3_BUCKET/runs/" --quiet 2>/dev/null || true
            sleep 120
        fi
        sleep 5
    done
) &

# ------- install deps -------
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl > /dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# ------- clone repo -------
cd /home/ubuntu
git clone --branch "$BRANCH" "$REPO_URL" gepa-mutations
cd gepa-mutations

if [ ! -d "gepa/src" ]; then
    git submodule update --init --recursive 2>/dev/null || \\
    git clone --branch v0.1.1 https://github.com/gepa-ai/gepa.git gepa
fi
rm -rf gepa/.venv gepa/uv.lock

# ------- secrets -------
export OPENROUTER_API_KEY=$(aws ssm get-parameter \\
    --name "/$PROJECT/openrouter-api-key" --with-decryption \\
    --query "Parameter.Value" --output text --region "$REGION")
export HF_TOKEN=$(aws ssm get-parameter \\
    --name "/$PROJECT/hf-token" --with-decryption \\
    --query "Parameter.Value" --output text --region "$REGION")
export TELEGRAM_BOT_TOKEN=$(aws ssm get-parameter \\
    --name "/$PROJECT/telegram-bot-token" --with-decryption \\
    --query "Parameter.Value" --output text --region "$REGION" 2>/dev/null || echo "")
export TELEGRAM_CHAT_ID=$(aws ssm get-parameter \\
    --name "/$PROJECT/telegram-chat-id" --with-decryption \\
    --query "Parameter.Value" --output text --region "$REGION" 2>/dev/null || echo "")

if [ -z "$OPENROUTER_API_KEY" ] || [ "$OPENROUTER_API_KEY" = "REPLACE_WITH_YOUR_KEY" ]; then
    echo "FATAL: OpenRouter API key not set in SSM"
    exit 1
fi

# ------- install python deps -------
uv sync

# ------- restore checkpoint from S3 (resume after interruption) -------
RUN_DIR="runs/$BENCHMARK/$METHOD/$SEED"
aws s3 sync "s3://$S3_BUCKET/$RUN_DIR/" "$RUN_DIR/" --quiet 2>/dev/null || true
if [ -d "$RUN_DIR/gepa_state" ]; then
    echo "Resuming from S3 checkpoint"
fi

# ------- update manifest to running -------
python3 -c "
import json, boto3, datetime
s3 = boto3.client('s3', region_name='$REGION')
try:
    obj = s3.get_object(Bucket='$S3_BUCKET', Key='status/manifest.json')
    m = json.loads(obj['Body'].read())
    for e in m['experiments']:
        if e['id'] == '$EXPERIMENT_ID':
            e['status'] = 'running'
            e['instance_id'] = '$INSTANCE_ID'
            e['started_at'] = datetime.datetime.utcnow().isoformat()
            break
    s3.put_object(Bucket='$S3_BUCKET', Key='status/manifest.json',
                  Body=json.dumps(m, indent=2))
except Exception as exc:
    print(f'Warning: {{exc}}')
" 2>/dev/null || true

# ------- run experiment -------
case "$METHOD" in
    best_of_k_K*)
        K=$(echo "$METHOD" | sed 's/best_of_k_K//')
        uv run python -c "
from best_of_k.runner import run_best_of_k
from gepa_mutations.base import MutationConfig
config = MutationConfig(
    mutation_name='$METHOD',
    description='Best-of-K with K=$K',
    benchmark='$BENCHMARK',
    seed=$SEED,
    mutation_candidates=int('$K'),
)
run_best_of_k(config=config, k=int('$K'))
"
        ;;
    contrastive_reflection)
        uv run python -c "
from contrastive_reflection.runner import run_contrastive_reflection
run_contrastive_reflection(benchmark='$BENCHMARK', seed=$SEED)
"
        ;;
    failure_stratified_k_K*)
        K=$(echo "$METHOD" | sed 's/failure_stratified_k_K//')
        uv run python -c "
from failure_stratified_k.runner import run_failure_stratified_k
from gepa_mutations.base import MutationConfig
config = MutationConfig(
    mutation_name='$METHOD',
    description='Failure-stratified K with K=$K',
    benchmark='$BENCHMARK',
    seed=$SEED,
    mutation_candidates=int('$K'),
    use_failure_stratified_k=True,
)
run_failure_stratified_k(config=config, k=int('$K'))
"
        ;;
    *)
        echo "ERROR: Unknown method $METHOD"
        exit 1
        ;;
esac

# ------- final upload -------
aws s3 sync runs/ "s3://$S3_BUCKET/runs/" --quiet
echo "=== Experiment $EXPERIMENT_ID completed at $(date -u) ==="
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

    launched = 0
    # Sort: Phase 1 first, then shorter benchmarks first (hotpotqa < pupa < aime)
    bm_order = {"hotpotqa": 0, "pupa": 1, "aime": 2}
    pending = sorted(
        [e for e in manifest["experiments"] if e["status"] == "pending"],
        key=lambda e: (e["phase"], bm_order.get(e["benchmark"], 9)),
    )

    for exp in pending:
        if launched >= available:
            break
        if not can_start_phase(exp["phase"], manifest):
            continue

        # Idempotency: skip if result already in S3
        if _result_exists_in_s3(exp):
            print(f"  Skip (result exists): {exp['id']}")
            exp["status"] = "completed"
            continue

        use_spot = exp["phase"] != 1  # On-demand for validation phase
        iid = launch_experiment(exp, sg, use_spot=use_spot)
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

        print(f"\n  Phase {phase} [{gate}] ({total} total):")
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


def poll_loop(interval: int = 900) -> None:
    """Continuously launch batches until all experiments finish."""
    while True:
        manifest = download_manifest()
        print_status(manifest)

        all_done = all(
            e["status"] in ("completed", "failed")
            for e in manifest["experiments"]
        )
        if all_done:
            print("All experiments complete.")
            break

        manifest = launch_batch(manifest)

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
    elif args.poll:
        poll_loop(interval=args.interval)
    elif args.auto:
        m = generate_manifest()
        upload_manifest(m)
        poll_loop(interval=args.interval)


if __name__ == "__main__":
    main()
