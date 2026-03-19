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
AMI_ID = "ami-0c7217cdde317cfec"  # Ubuntu 22.04 LTS us-east-1 (update as needed)
INSTANCE_PROFILE_NAME = "gepa-mutations-ec2-profile"
SG_NAME = "gepa-mutations-sg"
REPO_URL = "https://github.com/YOUR_ORG/gepa-mutations.git"  # Update with actual repo


def _user_data_script(
    benchmark: str,
    seed: int,
    use_merge: bool,
    branch: str = "master",
) -> str:
    """Generate EC2 user-data bootstrap script."""
    merge_flag = "" if use_merge else "--no-merge"

    return f"""#!/bin/bash
set -euxo pipefail

# Trap: auto-terminate on exit (success or failure)
cleanup() {{
    echo "Experiment finished (exit code $?). Self-terminating..."
    # Upload any partial results
    cd /home/ubuntu/gepa-mutations && \\
        uv run gepa-mutations upload runs/ 2>/dev/null || true
    # Self-terminate
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region {REGION}
}}
trap cleanup EXIT

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone repo
cd /home/ubuntu
git clone --branch {branch} {REPO_URL}
cd gepa-mutations

# Clone GEPA source (v0.1.1) if not already in the repo
if [ ! -d "gepa/src" ]; then
    git clone --branch v0.1.1 https://github.com/gepa-ai/gepa.git gepa
fi
# Remove any nested venv/lock that would confuse uv
rm -rf gepa/.venv gepa/uv.lock

# Fetch secrets from SSM
export OPENROUTER_API_KEY=$(aws ssm get-parameter --name "/{PROJECT}/openrouter-api-key" \\
    --with-decryption --query "Parameter.Value" --output text --region {REGION})
export HF_TOKEN=$(aws ssm get-parameter --name "/{PROJECT}/hf-token" \\
    --with-decryption --query "Parameter.Value" --output text --region {REGION})
export TELEGRAM_BOT_TOKEN=$(aws ssm get-parameter --name "/{PROJECT}/telegram-bot-token" \\
    --with-decryption --query "Parameter.Value" --output text --region {REGION})
export TELEGRAM_CHAT_ID=$(aws ssm get-parameter --name "/{PROJECT}/telegram-chat-id" \\
    --with-decryption --query "Parameter.Value" --output text --region {REGION})

# Install dependencies
uv sync

# Start spot interruption monitor in background
python3 -c "
import threading, time, urllib.request, sys, signal
def monitor():
    while True:
        try:
            req = urllib.request.Request('http://169.254.169.254/latest/meta-data/spot/instance-action')
            urllib.request.urlopen(req, timeout=2)
            print('SPOT INTERRUPTION DETECTED - flushing checkpoint...')
            # Signal main process to save state
            import os; os.kill(os.getppid(), signal.SIGUSR1)
            time.sleep(120)
        except (urllib.error.HTTPError, urllib.error.URLError):
            pass
        time.sleep(5)
t = threading.Thread(target=monitor, daemon=True)
t.start()
while True: time.sleep(3600)
" &

# Run experiment
uv run gepa-mutations run {benchmark} --seed {seed} {merge_flag}

# Upload results to S3
uv run gepa-mutations upload runs/

echo "Experiment completed successfully!"
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
