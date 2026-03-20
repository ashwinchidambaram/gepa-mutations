#!/usr/bin/env python3
"""Download all experiment results from S3 to the local runs/ directory.

Skips gepa_state/ checkpoint directories by default (large, only needed for
resume).  Pass --include-checkpoints to download everything.

Usage:
    python scripts/download_results.py
    python scripts/download_results.py --include-checkpoints
    python scripts/download_results.py --benchmark hotpotqa --method best_of_k_K3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3

BUCKET = "gepa-mutations-results"
REGION = "us-east-1"
LOCAL_DIR = Path("runs")


def download_results(
    benchmark: str | None = None,
    method: str | None = None,
    include_checkpoints: bool = False,
) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    paginator = s3.get_paginator("list_objects_v2")

    prefix = "runs/"
    if benchmark and method:
        prefix = f"runs/{benchmark}/{method}/"
    elif benchmark:
        prefix = f"runs/{benchmark}/"

    downloaded = 0
    skipped = 0

    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            # Strip the leading "runs/" to get the relative path
            rel = key[len("runs/"):]
            if not rel:
                continue

            # Skip checkpoints unless requested
            if not include_checkpoints and "gepa_state/" in key:
                skipped += 1
                continue

            local_path = LOCAL_DIR / rel
            if local_path.exists():
                continue  # Already have it

            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, key, str(local_path))
            downloaded += 1
            if downloaded % 20 == 0:
                print(f"  Downloaded {downloaded} files...")

    print(f"Done. Downloaded {downloaded} new files, "
          f"skipped {skipped} checkpoint files.")
    if not include_checkpoints and skipped > 0:
        print("  (pass --include-checkpoints to also download gepa_state/)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GEPA results from S3")
    parser.add_argument("--benchmark", default=None,
                        help="Filter by benchmark name")
    parser.add_argument("--method", default=None,
                        help="Filter by method name")
    parser.add_argument("--include-checkpoints", action="store_true",
                        help="Also download gepa_state/ directories")
    args = parser.parse_args()

    download_results(
        benchmark=args.benchmark,
        method=args.method,
        include_checkpoints=args.include_checkpoints,
    )


if __name__ == "__main__":
    main()
