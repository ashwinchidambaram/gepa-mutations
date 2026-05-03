"""Rung schedule and halving helper for SCALPEL Phase 6 successive halving.

See ``docs/scalpel/SCALPEL.md`` §3.C and §5.6.  The default schedule is
``(8, 16, 32, 64)`` rollouts-per-candidate at rungs 0..3; ``eta=2`` halves
the alive set after each rung.
"""

from __future__ import annotations

__all__ = ["DEFAULT_ETA", "DEFAULT_RUNGS", "halve"]


DEFAULT_RUNGS: tuple[int, ...] = (8, 16, 32, 64)
DEFAULT_ETA: int = 2  # halving factor.


def halve(n_alive: int, eta: int = DEFAULT_ETA) -> int:
    """Return the number of survivors after halving ``n_alive`` by ``eta``.

    Always at least 1 unless the input is 0 (or negative).
    """
    if n_alive <= 0:
        return 0
    return max(1, n_alive // eta)
