"""SCALPEL benchmark adapter.

Phase 2 of SCALPEL.  A thin shim over ``gepa_mutations.benchmarks`` that
exposes a uniform ``Benchmark`` Protocol (per ``docs/scalpel/SCALPEL.md`` Q6)
without duplicating loaders or scorers.

Public surface:

* :data:`SUPPORTED_BENCHMARKS` — tuple of benchmark names recognised by
  :func:`load`.  ``livebench`` is intentionally excluded for SCALPEL Phase 1.
* :class:`Benchmark` — a runtime-checkable Protocol with the attributes
  ``name``, ``trainset``, ``valset``, ``testset`` and the methods
  ``metric(gold, pred)`` and ``feedback(gold, pred, trace=None)``.
* :class:`BenchmarkAdapter` — a concrete implementation that wraps a
  ``BenchmarkData`` plus a GEPA-style adapter (the value returned by
  :func:`gepa_mutations.benchmarks.evaluators.get_adapter`).
* :func:`load` — factory that loads a benchmark by name and seed and
  returns a fully constructed :class:`BenchmarkAdapter`.

Note on ``task_lm``: ``get_adapter`` accepts ``task_lm=None`` for every
benchmark *except* ``aime``, which raises ``ValueError`` if no LM is given.
The SCALPEL adapter does not expose ``evaluate()`` rollouts at this phase,
so a ``None`` ``task_lm`` is harmless for the QA-style benchmarks we wrap;
``aime`` requires a non-``None`` ``task_lm`` to construct successfully.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from gepa_mutations.benchmarks.evaluators import get_adapter
from gepa_mutations.benchmarks.loader import BenchmarkData, load_benchmark

__all__ = [
    "SUPPORTED_BENCHMARKS",
    "Benchmark",
    "BenchmarkAdapter",
    "load",
]


# Order matters — locked by ``test_supported_benchmark_constant``.
SUPPORTED_BENCHMARKS: tuple[str, ...] = (
    "hotpotqa",
    "hover",
    "ifbench",
    "pupa",
    "aime",
)


@runtime_checkable
class Benchmark(Protocol):
    """Uniform benchmark interface consumed by the SCALPEL racing loop."""

    name: str
    trainset: list[Any]
    valset: list[Any]
    testset: list[Any]

    def metric(self, gold: Any, pred: Any) -> float: ...

    def feedback(self, gold: Any, pred: Any, trace: Any | None = None) -> str: ...


def _coerce_pred(pred: Any) -> Any:
    """Normalise a prediction for the underlying GEPA ``_score`` call.

    The GEPA scorers expect a *string* response (the model's free-form
    answer text).  SCALPEL callers may hand us a ``dspy.Prediction``-like
    object or a ``SimpleNamespace`` carrying ``.answer``; in those cases we
    extract the ``answer`` field.  Anything else is stringified.
    """
    if isinstance(pred, str):
        return pred
    if hasattr(pred, "answer"):
        return str(pred.answer)
    return str(pred)


class BenchmarkAdapter:
    """Concrete :class:`Benchmark` wrapping ``gepa_mutations.benchmarks`` artifacts."""

    name: str
    trainset: list[Any]
    valset: list[Any]
    testset: list[Any]
    metadata: dict[str, Any]
    _adapter: Any

    def __init__(self, name: str, data: BenchmarkData, gepa_adapter: Any) -> None:
        self.name = name
        self.trainset = data.train
        self.valset = data.val
        self.testset = data.test
        self.metadata = dict(data.metadata)
        self._adapter = gepa_adapter

    def metric(self, gold: Any, pred: Any) -> float:
        """Score a prediction against ``gold``, returning only the float."""
        score, _ = self._adapter._score(gold, _coerce_pred(pred))
        return float(score)

    def feedback(self, gold: Any, pred: Any, trace: Any | None = None) -> str:
        """Return the per-example feedback string from the underlying scorer.

        ``trace`` is accepted for Protocol compatibility but ignored; GEPA's
        ``_score`` produces feedback from ``(gold, response)`` alone.
        """
        _ = trace
        _, fb = self._adapter._score(gold, _coerce_pred(pred))
        return str(fb)


def load(name: str, seed: int = 0, task_lm: Any = None) -> BenchmarkAdapter:
    """Load a benchmark by name and wrap it in a :class:`BenchmarkAdapter`.

    Args:
        name: One of :data:`SUPPORTED_BENCHMARKS`.
        seed: Random seed forwarded to :func:`load_benchmark`.
        task_lm: Optional LM callable forwarded to :func:`get_adapter`.
            All benchmarks except ``aime`` accept ``None`` here; ``aime``
            requires a non-``None`` ``task_lm`` (see module docstring).

    Returns:
        A constructed :class:`BenchmarkAdapter`.

    Raises:
        ValueError: if ``name`` is not in :data:`SUPPORTED_BENCHMARKS`, or
            if ``name == "aime"`` and ``task_lm is None`` (re-raised from
            ``get_adapter``).
    """
    if name not in SUPPORTED_BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark '{name}'. Choose from: {list(SUPPORTED_BENCHMARKS)}"
        )

    data = load_benchmark(name, seed=seed)
    gepa_adapter = get_adapter(name, task_lm=task_lm, parallel_workers=1)
    return BenchmarkAdapter(name, data, gepa_adapter)
