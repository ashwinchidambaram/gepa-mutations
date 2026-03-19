"""S3 storage for experiment results.

Bucket structure: s3://<bucket>/runs/<benchmark>/<method>/<seed>/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import boto3

from gepa_mutations.config import Settings


def _get_client():
    return boto3.client("s3")


def _get_bucket(bucket: str | None = None) -> str:
    if bucket:
        return bucket
    return Settings().s3_bucket


def upload_results(local_dir: str, bucket: str | None = None) -> None:
    """Upload a local results directory to S3.

    Recursively uploads all files in local_dir, preserving directory structure.
    """
    s3 = _get_client()
    bucket_name = _get_bucket(bucket)
    local_path = Path(local_dir)

    if not local_path.exists():
        raise FileNotFoundError(f"Directory not found: {local_dir}")

    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            key = str(file_path.relative_to(local_path.parent))
            s3.upload_file(str(file_path), bucket_name, key)
            print(f"  Uploaded: s3://{bucket_name}/{key}")


def download_results(
    benchmark: str,
    seed: int,
    method: str = "gepa",
    bucket: str | None = None,
    local_dir: str = "runs",
) -> Path:
    """Download results from S3 to local filesystem."""
    s3 = _get_client()
    bucket_name = _get_bucket(bucket)
    prefix = f"runs/{benchmark}/{method}/{seed}/"

    local_path = Path(local_dir) / benchmark / method / str(seed)
    local_path.mkdir(parents=True, exist_ok=True)

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    for obj in response.get("Contents", []):
        key = obj["Key"]
        filename = key[len(prefix) :]
        if filename:
            target = local_path / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket_name, key, str(target))

    return local_path


def list_results(
    benchmark: str | None = None,
    method: str = "gepa",
    bucket: str | None = None,
) -> list[dict[str, str]]:
    """List available results in S3."""
    s3 = _get_client()
    bucket_name = _get_bucket(bucket)

    prefix = "runs/"
    if benchmark:
        prefix = f"runs/{benchmark}/{method}/"

    results = []
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter="/")

    for common_prefix in response.get("CommonPrefixes", []):
        path = common_prefix["Prefix"]
        parts = path.strip("/").split("/")
        if len(parts) >= 4:
            results.append({
                "benchmark": parts[1],
                "method": parts[2],
                "seed": parts[3],
                "s3_path": f"s3://{bucket_name}/{path}",
            })

    return results
