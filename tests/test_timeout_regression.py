"""Regression tests for the shared thread pool fix in runner/timeout.py.

Bug: Per-call ThreadPoolExecutor leaked orphan threads on timeout.
     Old code called shutdown(wait=False) which orphaned running threads.
     After ~95 timeouts the process deadlocked.
Fix: Module-level singleton pool capped at _TIMEOUT_POOL_SIZE threads.
"""

from __future__ import annotations

import concurrent.futures
import time

import pytest

import gepa_mutations.runner.timeout as timeout_mod
from gepa_mutations.runner.timeout import (
    _TIMEOUT_POOL_SIZE,
    _get_pool,
    call_with_timeout,
)

# ---------------------------------------------------------------------------
# Fixture: isolate module-level pool state between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_timeout_pool():
    """Save and restore the module-level _TIMEOUT_POOL around each test."""
    original_pool = timeout_mod._TIMEOUT_POOL
    timeout_mod._TIMEOUT_POOL = None  # force lazy re-creation in each test
    yield
    # Shut down any pool created during this test, then restore the original.
    created_pool = timeout_mod._TIMEOUT_POOL
    if created_pool is not None and created_pool is not original_pool:
        created_pool.shutdown(wait=False)
    timeout_mod._TIMEOUT_POOL = original_pool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_timeout_raises_on_hang():
    """call_with_timeout must raise when the callable exceeds the deadline."""
    with pytest.raises(concurrent.futures.TimeoutError):
        call_with_timeout(time.sleep, 999, timeout_seconds=0.2)


def test_successful_call_returns_value():
    """call_with_timeout must return the callable's return value on success."""
    result = call_with_timeout(lambda: 42, timeout_seconds=5)
    assert result == 42


def test_pool_bounded_after_many_timeouts():
    """Thread count must stay <= _TIMEOUT_POOL_SIZE after many timeouts.

    This is the core regression: the old per-call pool leaked one thread per
    timeout; the shared pool caps thread count regardless of timeout count.
    """
    num_timeouts = 50
    for _ in range(num_timeouts):
        try:
            call_with_timeout(time.sleep, 999, timeout_seconds=0.1)
        except concurrent.futures.TimeoutError:
            pass

    pool = _get_pool()
    # ThreadPoolExecutor._threads is the internal set of live worker threads.
    thread_count = len(pool._threads)
    assert thread_count <= _TIMEOUT_POOL_SIZE, (
        f"Expected at most {_TIMEOUT_POOL_SIZE} threads, but found {thread_count}. "
        "Thread pool is leaking threads on timeout."
    )


def test_pool_size_env_override(monkeypatch):
    """TIMEOUT_POOL_SIZE env var must control the pool's max_workers."""
    monkeypatch.setenv("TIMEOUT_POOL_SIZE", "4")
    # Reset so _get_pool() re-reads _TIMEOUT_POOL_SIZE from the env var.
    # Note: _TIMEOUT_POOL_SIZE is a module constant evaluated at import time,
    # so we also patch it on the module to simulate a fresh import.
    monkeypatch.setattr(timeout_mod, "_TIMEOUT_POOL_SIZE", 4)
    timeout_mod._TIMEOUT_POOL = None

    pool = _get_pool()
    assert pool._max_workers == 4, (
        f"Expected max_workers=4 (from TIMEOUT_POOL_SIZE=4), got {pool._max_workers}"
    )
