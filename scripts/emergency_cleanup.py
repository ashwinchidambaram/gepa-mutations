#!/usr/bin/env python3
"""Terminate ALL running GEPA experiment instances. Emergency use only.

Lists all EC2 instances tagged Project=gepa-mutations that are in a
pending/running state, then asks for confirmation before terminating.

Usage:
    python scripts/emergency_cleanup.py
"""

from __future__ import annotations

import boto3

REGION = "us-east-1"
PROJECT = "gepa-mutations"


def cleanup() -> None:
    ec2 = boto3.client("ec2", region_name=REGION)
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Project", "Values": [PROJECT]},
            {"Name": "instance-state-name",
             "Values": ["pending", "running", "stopping"]},
        ],
    )

    instance_ids: list[str] = []
    for res in resp["Reservations"]:
        for inst in res["Instances"]:
            iid = inst["InstanceId"]
            tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
            eid = tags.get("ExperimentId", "?")
            state = inst["State"]["Name"]
            instance_ids.append(iid)
            print(f"  {iid}  {state:10s}  {eid}")

    if not instance_ids:
        print("No active GEPA instances found.")
        return

    print(f"\nWill terminate {len(instance_ids)} instance(s).")
    answer = input("Type 'yes' to confirm: ").strip()
    if answer != "yes":
        print("Aborted.")
        return

    ec2.terminate_instances(InstanceIds=instance_ids)
    print(f"Termination initiated for {len(instance_ids)} instances.")


if __name__ == "__main__":
    cleanup()
