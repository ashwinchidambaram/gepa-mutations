"""Hard wall-clock timeout wrapper using a shared ThreadPoolExecutor.

litellm/httpx timeouts don't reliably fire when TCP connections go idle
without triggering a read timeout. This provides a hard deadline.

IMPORTANT: uses a module-level SHARED pool (max 8 threads) instead of creating
a new pool per call. The old per-call pattern leaked a thread on every timeout
(ThreadPoolExecutor.shutdown(wait=False) orphans the running thread). After ~95
timed-out calls (one per GEPA iteration), the process accumulated 95+ orphan
threads and deadlocked. A shared pool caps thread count at 8 regardless of how
many timeouts occur — hung threads eventually free when the underlying httpx
connection times out.
"""

from __future__ import annotations

import atexit
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

_TIMEOUT_POOL_SIZE = int(os.environ.get("TIMEOUT_POOL_SIZE", 8))
_TIMEOUT_POOL: ThreadPoolExecutor | None = None


def _get_pool() -> ThreadPoolExecutor:
    """Lazily create the shared pool (avoids import-time side effects)."""
    global _TIMEOUT_POOL
    if _TIMEOUT_POOL is None:
        _TIMEOUT_POOL = ThreadPoolExecutor(max_workers=_TIMEOUT_POOL_SIZE)
        atexit.register(_TIMEOUT_POOL.shutdown, wait=False, cancel_futures=True)
    return _TIMEOUT_POOL


def call_with_timeout(fn, *args, timeout_seconds: int = 150, **kwargs) -> Any:
    """Call *fn* with a hard wall-clock timeout.

    Uses a shared pool of ``_TIMEOUT_POOL_SIZE`` worker threads (default 8).
    If a call times out, the underlying thread stays occupied until the httpx
    connection eventually times out — but total thread count is bounded.

    Args:
        fn: Callable to invoke.
        *args: Positional arguments forwarded to *fn*.
        timeout_seconds: Maximum seconds to wait (default 150 = litellm 120s + 30s buffer).
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn(*args, **kwargs)*.

    Raises:
        TimeoutError: If *fn* doesn't return within *timeout_seconds*.
    """
    pool = _get_pool()
    future: Future = pool.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    except Exception:
        # Mark the future as cancelled so the pool can reclaim the worker
        # once the underlying call finishes (cancel() is best-effort — if
        # the call is already running, it won't actually stop, but the
        # thread IS returned to the pool when the call eventually completes
        # or the process exits).
        future.cancel()
        raise
