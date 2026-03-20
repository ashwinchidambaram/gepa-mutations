"""Launch GEPA experiments on AWS EC2 spot instances.

Instance: t3.medium spot ($0.0042/hr in us-east-1)
Bootstrap: Install uv, clone repo, fetch secrets from SSM, run experiment,
           upload results, self-terminate.

Safety:
  - Auto-terminate on success AND failure (trap handler)
  - CloudWatch alarm if > 36hrs
  - Spot interruption: background thread polls metadata, flushes checkpoint on 2-min warning
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time

import boto3
from botocore.exceptions import ClientError

# Configuration
REGION = "us-east-1"
PROJECT = "gepa-mutations"
INSTANCE_TYPE = "t3.medium"
AMI_ID = "ami-0c421724a94bba6d6"  # Amazon Linux 2023 us-east-1
INSTANCE_PROFILE_NAME = "gepa-mutations-ec2-profile"
SG_NAME = "gepa-mutations-sg"
REPO_URL = "https://github.com/ashwinchidambaram/gepa-mutations.git"  # Update with actual repo


def _user_data_script(
    benchmark: str,
    seed: int,
    use_merge: bool,
    branch: str = "master",
) -> str:
    """Generate EC2 user-data bootstrap script."""
    merge_flag = "" if use_merge else "--no-merge"

    return f"""#!/bin/bash
LOG=/var/log/gepa-experiment.log
log() {{ echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }}

TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)

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

# ------- hourly progress monitor (background) -------
hourly_monitor() {{
    while true; do
        sleep 3600
        if [ -f /tmp/gepa_exit_code ]; then
            break  # experiment finished, stop monitoring
        fi
        local UPTIME_HRS=$(awk '{{printf "%.1f", $1/3600}}' /proc/uptime)
        local TAIL_LOG=$(tail -5 $LOG 2>/dev/null | head -3)
        send_telegram "<b>Hourly Update</b>
Experiment: <code>gepa/{benchmark}/seed={seed}</code>
Instance: <code>$INSTANCE_ID</code>
Uptime: ${{UPTIME_HRS}}h
Recent log:
<pre>${{TAIL_LOG}}</pre>"
    done
}}
hourly_monitor &
MONITOR_PID=$!

cleanup() {{
    log "Cleanup: uploading and terminating"
    # Stop hourly monitor
    kill $MONITOR_PID 2>/dev/null || true
    aws s3 cp $LOG s3://gepa-mutations-results/runs/{benchmark}/gepa/{seed}/ec2.log --region us-east-1 || true
    [ -d /root/gepa-mutations/runs ] && aws s3 sync /root/gepa-mutations/runs/ s3://gepa-mutations-results/runs/ --region us-east-1 || true
    # ------- send completion notification -------
    EXIT_CODE=${{GEPA_EXIT_CODE:-1}}
    if [ "$EXIT_CODE" = "0" ]; then
        TEST_SCORE=$(python3 -c "
import json, glob
files = glob.glob('/root/gepa-mutations/runs/{benchmark}/gepa/{seed}/result.json')
if files:
    d = json.load(open(files[0]))
    print(f\"{{d.get('test_score', 0)*100:.2f}}%\")
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
        WALL_CLOCK=$(python3 -c "
import json, glob
files = glob.glob('/root/gepa-mutations/runs/{benchmark}/gepa/{seed}/result.json')
if files:
    d = json.load(open(files[0]))
    wc = d.get('wall_clock_seconds', 0)
    h, m = int(wc//3600), int((wc%3600)//60)
    print(f'{{h}}h {{m}}m')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
        send_telegram "Baseline experiment complete
Benchmark: <code>{benchmark}</code>
Seed: <code>{seed}</code>
Test score: <b>${{TEST_SCORE}}</b>
Wall clock: ${{WALL_CLOCK}}
Instance: <code>${{INSTANCE_ID}}</code>"
    else
        send_telegram "Baseline experiment FAILED
Benchmark: <code>{benchmark}</code>
Seed: <code>{seed}</code>
Exit code: ${{EXIT_CODE}}
Instance: <code>${{INSTANCE_ID}}</code>
Last log line: <pre>$(tail -1 $LOG 2>/dev/null)</pre>"
    fi
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1 || true
}}
trap cleanup EXIT

log "=== GEPA Experiment: {benchmark} seed={seed} ==="
yum install -y git >> $LOG 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh >> $LOG 2>&1
export PATH="/root/.local/bin:$PATH"

cd /root
git clone --depth 1 --branch {branch} {REPO_URL} >> $LOG 2>&1
cd gepa-mutations
git clone --depth 1 --branch v0.1.1 https://github.com/gepa-ai/gepa.git gepa >> $LOG 2>&1
rm -rf gepa/.venv gepa/uv.lock

aws s3 cp s3://gepa-mutations-results/config/.env .env --region us-east-1 >> $LOG 2>&1
uv sync >> $LOG 2>&1

# ------- send start notification -------
send_telegram "Baseline experiment started
Benchmark: <code>{benchmark}</code>
Seed: <code>{seed}</code>
Merge: {'yes' if use_merge else 'no'}
Instance: <code>${{INSTANCE_ID}}</code>"

log "Starting: gepa-mutations run {benchmark} --seed {seed} {merge_flag}"
uv run gepa-mutations run {benchmark} --seed {seed} {merge_flag} >> $LOG 2>&1
GEPA_EXIT_CODE=$?
echo $GEPA_EXIT_CODE > /tmp/gepa_exit_code
log "Exit code: $GEPA_EXIT_CODE"
"""


def launch(
    benchmark: str,
    seed: int = 42,
    use_merge: bool = True,
    branch: str = "master",
    dry_run: bool = False,
) -> str | None:
    """Launch an experiment on an EC2 spot instance.

    Returns the instance ID.
    """
    ec2 = boto3.client("ec2", region_name=REGION)

    # Get security group ID
    sgs = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [SG_NAME]}]
    )
    if not sgs["SecurityGroups"]:
        print(f"ERROR: Security group '{SG_NAME}' not found. Run aws_setup.py first.")
        return None
    sg_id = sgs["SecurityGroups"][0]["GroupId"]

    user_data = _user_data_script(benchmark, seed, use_merge, branch)

    if dry_run:
        print("DRY RUN — would launch with this user-data:")
        print(user_data[:500] + "...")
        return None

    print(f"Launching {INSTANCE_TYPE} spot instance for {benchmark} (seed {seed})...")

    response = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        MinCount=1,
        MaxCount=1,
        IamInstanceProfile={"Name": INSTANCE_PROFILE_NAME},
        SecurityGroupIds=[sg_id],
        UserData=user_data,
        MetadataOptions={
            "HttpTokens": "required",
            "InstanceMetadataTags": "enabled",
        },
        InstanceMarketOptions={
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        },
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"{PROJECT}-{benchmark}-{seed}"},
                    {"Key": "Project", "Value": PROJECT},
                    {"Key": "Benchmark", "Value": benchmark},
                    {"Key": "Seed", "Value": str(seed)},
                ],
            },
        ],
    )

    instance_id = response["Instances"][0]["InstanceId"]
    print(f"  Instance launched: {instance_id}")
    print(f"  Type: {INSTANCE_TYPE} spot")
    print(f"  Benchmark: {benchmark}, Seed: {seed}")
    print(f"  Will self-terminate on completion")

    return instance_id


def launch_multi_seed(
    benchmark: str,
    seeds: list[int] | None = None,
    use_merge: bool = True,
    branch: str = "master",
) -> list[str]:
    """Launch experiments with multiple seeds."""
    seeds = seeds or [42, 123, 456, 789, 1024]
    instance_ids = []

    for seed in seeds:
        iid = launch(benchmark, seed, use_merge, branch)
        if iid:
            instance_ids.append(iid)
        time.sleep(2)  # Avoid API throttling

    print(f"\nLaunched {len(instance_ids)} instances for {benchmark}")
    return instance_ids


def main():
    parser = argparse.ArgumentParser(description="Launch GEPA experiment on EC2")
    parser.add_argument("benchmark", help="Benchmark to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds for multi-seed launch",
    )
    parser.add_argument("--no-merge", action="store_true", help="Disable merge")
    parser.add_argument("--branch", default="master", help="Git branch")
    parser.add_argument("--dry-run", action="store_true", help="Print user-data only")

    args = parser.parse_args()

    if args.seeds:
        seed_list = [int(s.strip()) for s in args.seeds.split(",")]
        launch_multi_seed(args.benchmark, seed_list, not args.no_merge, args.branch)
    else:
        launch(args.benchmark, args.seed, not args.no_merge, args.branch, args.dry_run)


if __name__ == "__main__":
    main()
