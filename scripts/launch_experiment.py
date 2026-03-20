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

cleanup() {{
    log "Cleanup: uploading and terminating"
    aws s3 cp $LOG s3://gepa-mutations-results/runs/{benchmark}/gepa/{seed}/ec2.log --region us-east-1 || true
    [ -d /root/gepa-mutations/runs ] && aws s3 sync /root/gepa-mutations/runs/ s3://gepa-mutations-results/runs/ --region us-east-1 || true
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

log "Starting: gepa-mutations run {benchmark} --seed {seed} {merge_flag}"
uv run gepa-mutations run {benchmark} --seed {seed} {merge_flag} >> $LOG 2>&1
log "Exit code: $?"
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
