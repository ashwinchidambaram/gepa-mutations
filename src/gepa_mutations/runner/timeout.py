"""Hard wall-clock timeout wrapper using ThreadPoolExecutor.

litellm/httpx timeouts don't reliably fire when TCP connections go idle
without triggering a read timeout. This provides a hard deadline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any


def call_with_timeout(fn, *args, timeout_seconds: int = 150, **kwargs) -> Any:
    """Call *fn* with a hard wall-clock timeout.

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
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
