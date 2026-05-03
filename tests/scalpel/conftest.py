"""Shared fixtures for SCALPEL tests.

Markers gated here:

* ``live`` — hits the raycluster vLLM endpoint (http://10.0.10.66:8123/v1,
  VPN required).  Enable with ``SCALPEL_LIVE_TESTS=1``.
* ``local_only`` — exercises real benchmark loaders / Hugging Face downloads.
  Enable with ``SCALPEL_LOCAL_TESTS=1``.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    _ = config
    live_enabled = os.environ.get("SCALPEL_LIVE_TESTS") == "1"
    local_enabled = os.environ.get("SCALPEL_LOCAL_TESTS") == "1"
    skip_live = pytest.mark.skip(reason="live tests disabled (set SCALPEL_LIVE_TESTS=1)")
    skip_local = pytest.mark.skip(
        reason="local_only tests disabled (set SCALPEL_LOCAL_TESTS=1)"
    )
    for item in items:
        if not live_enabled and "live" in item.keywords:
            item.add_marker(skip_live)
        if not local_enabled and "local_only" in item.keywords:
            item.add_marker(skip_local)
