#!/usr/bin/env python3
"""Check health of all running GEPA experiment instances.

Reads heartbeat files from S3 and cross-references with live EC2 state.
Flags stale, dead, or long-running experiments.

Usage:
    python scripts/check_health.py
"""

from __future__ import annotations

import datetime
import json

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
BUCKET = "gepa-mutations-results"
PROJECT = "gepa-mutations"
STALE_MINUTES = 15  # Heartbeat older than this = STALE


def check_health() -> None:
    ec2 = boto3.client("ec2", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    # Gather running instances
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Project", "Values": [PROJECT]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ],
    )

    instances: list[dict] = []
    for res in resp["Reservations"]:
        for inst in res["Instances"]:
            tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
            instances.append({
                "instance_id": inst["InstanceId"],
                "experiment_id": tags.get("ExperimentId", "?"),
                "benchmark": tags.get("Benchmark", "?"),
                "phase": tags.get("Phase", "?"),
                "state": inst["State"]["Name"],
                "launch_time": inst["LaunchTime"],
                "uptime_hrs": (
                    datetime.datetime.now(datetime.timezone.utc) - inst["LaunchTime"]
                ).total_seconds() / 3600,
            })

    if not instances:
        print("No running GEPA instances found.")
        return

    # Header
    hdr = (f"{'Instance':20s} {'Experiment':45s} {'Up':>6s} {'Iter':>5s} "
           f"{'Calls':>6s} {'Acc%':>5s} {'Status':>7s}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for inst in sorted(instances, key=lambda i: i["experiment_id"]):
        iid = inst["instance_id"]
        eid = inst["experiment_id"]
        up = f"{inst['uptime_hrs']:.1f}h"

        # Try to fetch heartbeat
        try:
            obj = s3.get_object(
                Bucket=BUCKET, Key=f"logs/{iid}/heartbeat.json"
            )
            hb = json.loads(obj["Body"].read())
            ts = datetime.datetime.fromisoformat(hb["timestamp"])
            age_min = (datetime.datetime.utcnow() - ts).total_seconds() / 60
            alive = hb.get("experiment_alive", False)

            iters = str(hb.get("iterations", "?"))
            calls = str(hb.get("metric_calls", "?"))
            acc = hb.get("acceptance_rate")
            acc_str = f"{acc:.0%}" if acc is not None else "?"

            if not alive:
                status = "DEAD"
            elif age_min > STALE_MINUTES:
                status = "STALE"
            else:
                status = "OK"

        except (ClientError, KeyError, json.JSONDecodeError):
            iters = "?"
            calls = "?"
            acc_str = "?"
            status = "NO_HB"

        print(f"{iid:20s} {eid:45s} {up:>6s} {iters:>5s} "
              f"{calls:>6s} {acc_str:>5s} {status:>7s}")

    print()


if __name__ == "__main__":
    check_health()
