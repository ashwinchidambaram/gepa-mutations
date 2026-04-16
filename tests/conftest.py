"""conftest.py — ensure the real gepa_mutations package is importable before tests run.

Some test files (test_failure_matrix.py, test_donor_selection.py) stub out
gepa_mutations submodules so they can import colony.py standalone.  Those stubs
must NOT overwrite the real MetricsCollector used by test_trajectory_point.py.

We solve this by eagerly loading the real package here, before any test module
runs.  The stub-installing test files check whether the real class is already
present and skip the stub installation in that case.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path (mirrors what test_trajectory_point.py does)
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Pre-load the real collector so stub tests can't clobber it
try:
    import gepa_mutations.metrics.collector  # noqa: F401
except Exception:
    pass  # If the real package isn't importable, stub tests will install stubs as usual
