"""SCALPEL Phase 9 — integrating optimizer.

Wires Phases 1-8 into a single ``SCALPEL`` class that satisfies
``iso_harness.experiment.protocols.Optimizer`` and (optionally)
``Checkpointable``.  The implementation follows ``docs/scalpel/SCALPEL.md``
§3.G (algorithm pseudocode), §5.10 (class spec), and the addenda Q1
(one cluster per iter), Q3 (no auto-promote), Q9 (system-level Pareto
frontier), and Q10 (AIME excluded from headline).

Public surface:

* :class:`Candidate` — pydantic model for a Pareto-pool candidate (one entry
  per system, holding the prompt for every module).
* :class:`IterationLog` — pydantic model for per-iteration telemetry.
* :class:`SCALPEL` — the optimizer class.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from iso_harness.experiment.context import run_context
from scalpel.clustering.kmeans import ClusterState, FailureClusterer
from scalpel.clustering.targeting import TargetingHistory, select_target_cluster
from scalpel.edits.apply import LengthCapExceeded, apply
from scalpel.edits.grammar import EDIT_LIST_SCHEMA, Edit, StructuredPrompt
from scalpel.edits.span_index import materialize, parse
from scalpel.lesson_book.retrieval import top_m_lessons
from scalpel.lesson_book.store import LessonBook
from scalpel.racing.successive_halving import SuccessiveHalving
from scalpel.reflection.parser import parse_reflection_response
from scalpel.reflection.prompt_builder import build_reflection_prompt
from scalpel.surrogate.skip_policy import SkipPolicy

__all__ = ["Candidate", "IterationLog", "SCALPEL"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- models


class Candidate(BaseModel):
    """A single candidate in the system-level Pareto pool (addendum Q9)."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    id: str
    parent_id: Optional[str] = None
    prompts: dict[str, StructuredPrompt]
    edits_applied: list[Edit] = Field(default_factory=list)
    pareto_score: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IterationLog(BaseModel):
    """One iteration's telemetry."""

    iter_num: int
    target_cluster_id: int
    target_module: str
    n_proposals: int
    n_candidates_after_apply: int
    race_total_rollouts: int
    race_total_skipped: int
    survivor_score: float
    accepted: bool
    cluster_k: int
    cluster_silhouette: float
    n_active_lessons: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ----------------------------------------------------------- internal helpers


class _StubEmbedder:
    """Tiny deterministic stand-in for :class:`scalpel.clustering.embeddings.BGEEmbedder`.

    Used internally when ``embedder=None`` is passed and the caller has not
    triggered a real BGE load.  Hashes text into a fixed 384-d float vector.
    The Lesson Book and clusterer call ``embed_one`` only; the dense BGE
    surface is not required for Phase 9's mock-test path.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed_one(self, text: str):
        # Deterministic pseudo-random vector keyed on the text.
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        n = float(np.linalg.norm(v)) or 1.0
        return v / n

    def embed(self, texts: list[str]):
        return np.stack([self.embed_one(t) for t in texts]) if texts else np.zeros(
            (0, self.dim), dtype=np.float32
        )


def _coerce_prompt(value: str | StructuredPrompt) -> StructuredPrompt:
    """Normalize a seed prompt into a :class:`StructuredPrompt`."""
    if isinstance(value, StructuredPrompt):
        return value
    return parse(value)


def _instance_input(instance: Any) -> Any:
    """Extract the model-facing input from a dataset instance."""
    if hasattr(instance, "input"):
        return getattr(instance, "input")
    if isinstance(instance, dict):
        if "input" in instance:
            return instance["input"]
        if "question" in instance:
            return instance["question"]
        return instance
    return instance


def _instance_gold(instance: Any) -> Any:
    """Extract the gold label from a dataset instance."""
    if hasattr(instance, "gold"):
        return getattr(instance, "gold")
    if isinstance(instance, dict):
        if "gold" in instance:
            return instance["gold"]
        if "answer" in instance:
            return instance["answer"]
    return None


# --------------------------------------------------------------- SCALPEL class


class SCALPEL:
    """SCALPEL prompt optimizer.

    Conforms to :class:`iso_harness.experiment.protocols.Optimizer` and
    optionally :class:`iso_harness.experiment.protocols.Checkpointable`.
    """

    def __init__(
        self,
        task_lm: Callable[..., Any],
        reflect_lm: Callable[..., Any],
        K: int = 8,
        rungs: tuple[int, ...] = (8, 16, 32, 64),
        eta: int = 2,
        alpha: float = 0.15,
        max_iters: int = 60,
        budget_tokens: int | None = None,
        embedder: Any = None,
        clusterer: FailureClusterer | None = None,
        targeting_history: TargetingHistory | None = None,
        race: SuccessiveHalving | None = None,
        surrogate: SkipPolicy | None = None,
        lesson_book: LessonBook | None = None,
        seed: int = 0,
    ) -> None:
        self.task_lm = task_lm
        self.reflect_lm = reflect_lm
        self.K = K
        self.rungs = rungs
        self.eta = eta
        self.alpha = alpha
        self.max_iters = max_iters
        self.budget_tokens = budget_tokens
        self.seed = seed

        # Auto-instantiate any subsystem the caller did not supply.  We use
        # _StubEmbedder by default to keep construction cheap (no BGE
        # download).  Callers in production should pass a real BGEEmbedder.
        self.embedder = embedder if embedder is not None else _StubEmbedder()
        self.clusterer = (
            clusterer if clusterer is not None else FailureClusterer(seed=seed)
        )
        self.targeting_history = (
            targeting_history if targeting_history is not None else TargetingHistory()
        )
        self.race = (
            race
            if race is not None
            else SuccessiveHalving(rungs=rungs, eta=eta, rng_seed=seed)
        )
        self.surrogate = (
            surrogate if surrogate is not None else SkipPolicy(rng_seed=seed)
        )
        self.lesson_book = (
            lesson_book
            if lesson_book is not None
            else LessonBook(embedder=self.embedder)
        )

        # Wire the surrogate into the race driver if the caller did not
        # pre-supply a custom race.
        if race is None and surrogate is not None:
            self.race.surrogate = surrogate

        # Internal mutable state.
        self._rng = random.Random(seed)
        self._candidates: list[Candidate] = []
        self._iteration_logs: list[IterationLog] = []
        self._module_round_robin_idx: int = 0
        self._run_id: str = str(uuid4())
        self._module_names: list[str] = []
        self._failure_pool: list[Any] = []  # raw failure instance objects
        # Tokens consumed by both LMs (best-effort, via _last_usage).
        self._cumulative_tokens: int = 0

    # ===================================================== helpers / state ====
    @property
    def iteration_logs(self) -> list[IterationLog]:
        return list(self._iteration_logs)

    @property
    def candidates(self) -> list[Candidate]:
        return list(self._candidates)

    @property
    def best_candidate(self) -> Candidate | None:
        if not self._candidates:
            return None
        return max(
            self._candidates, key=lambda c: (c.pareto_score, -c.created_at.timestamp())
        )

    # ===================================================== Optimizer Protocol
    def compile(
        self,
        student: dict[str, str | StructuredPrompt],
        trainset: list[Any],
        valset: list[Any],
        metric: Callable[[Any, Any], float] | None = None,
        feedback: Callable[..., str] | None = None,
    ) -> dict[str, StructuredPrompt]:
        """Run the optimizer and return the best candidate's prompts."""
        if not student:
            raise ValueError("student dict is empty; need at least one module")
        if metric is None:
            raise ValueError("metric is required (no Benchmark auto-detection here)")

        # Default feedback is "no feedback" — the failure pool stays empty
        # in that case but the rest of the pipeline still functions.
        feedback_fn: Callable[..., str] = feedback if feedback is not None else (
            lambda gold, pred, trace=None: ""
        )

        seed_prompts: dict[str, StructuredPrompt] = {
            name: _coerce_prompt(value) for name, value in student.items()
        }
        self._module_names = list(seed_prompts.keys())

        # The seed candidate goes into the pool with a real valset score so
        # it has something for children to beat.
        seed = Candidate(
            id=str(uuid4()),
            parent_id=None,
            prompts=seed_prompts,
            edits_applied=[],
        )
        with run_context(
            run_id=self._run_id,
            round_num=0,
            candidate_id=seed.id,
            phase="seed_eval",
        ):
            seed.pareto_score = self._evaluate_on_valset(seed, valset, metric)
        self._candidates.append(seed)

        # ---------------------------------------------------------- main loop
        for iter_num in range(1, self.max_iters + 1):
            if (
                self.budget_tokens is not None
                and self._cumulative_tokens >= self.budget_tokens
            ):
                logger.info(
                    "SCALPEL stopping: budget_tokens=%d reached at iter=%d",
                    self.budget_tokens,
                    iter_num - 1,
                )
                break

            self._run_one_iter(
                iter_num=iter_num,
                trainset=trainset,
                valset=valset,
                metric=metric,
                feedback=feedback_fn,
            )

        best = self.best_candidate
        assert best is not None
        return best.prompts

    # ==================================================== iteration internals
    def _run_one_iter(
        self,
        *,
        iter_num: int,
        trainset: list[Any],
        valset: list[Any],
        metric: Callable[[Any, Any], float],
        feedback: Callable[..., str],
    ) -> None:
        # 1. Recluster check.
        self.clusterer.step_iteration()
        if self.clusterer.should_recluster():
            self.clusterer.recluster()
        cluster_states = self.clusterer.cluster_states()
        if not cluster_states:
            cluster_states = [
                ClusterState(id=0, centroid=[0.0] * 384, failure_count=1)
            ]

        # 2. Target cluster.
        target_cluster_id = select_target_cluster(
            cluster_states, self.targeting_history
        )
        self.targeting_history = self.targeting_history.record(target_cluster_id)
        target_cluster = next(
            c for c in cluster_states if c.id == target_cluster_id
        )

        # 3. Parent selection (highest pareto_score; tie-break = older first).
        parent = max(
            self._candidates,
            key=lambda c: (c.pareto_score, -c.created_at.timestamp()),
        )

        # 4. Module selection (round-robin across declared modules).
        target_module = self._module_names[
            self._module_round_robin_idx % len(self._module_names)
        ]
        self._module_round_robin_idx += 1
        parent_prompt = parent.prompts[target_module]

        # 5. Build the reflection prompt.
        active_lessons = top_m_lessons(self.lesson_book, m=12)
        lesson_book_text = self.lesson_book.render(top_m=12) if active_lessons else ""
        parent_token_count = max(1, parent_prompt.token_count or len(
            materialize(parent_prompt).split()
        ))
        alpha_budget = max(1, int(self.alpha * parent_token_count))
        system_prompt, user_prompt = build_reflection_prompt(
            parent_prompt=parent_prompt,
            target_module=target_module,
            cluster_id=target_cluster.id,
            cluster_summary=target_cluster.summary or f"cluster-{target_cluster.id}",
            representative_trace="(none)",
            lesson_book_text=lesson_book_text,
            alpha_token_budget=alpha_budget,
        )
        reflect_prompt = f"{system_prompt}\n\n{user_prompt}"

        # 6. K reflection calls -> K edit lists -> K children (with skip on
        # length-cap exceedance, plus up to 3 resamples for the whole batch).
        children: list[Candidate] = []
        accumulated_lessons: list[str] = []
        n_proposals = 0
        attempts = 0
        max_attempts = self.K * 3
        while len(children) < self.K and attempts < max_attempts:
            attempts += 1
            n_proposals += 1
            with run_context(
                run_id=self._run_id,
                round_num=iter_num,
                candidate_id=parent.id,
                phase="reflect",
            ):
                raw = self._call_reflect(reflect_prompt)
            self._account_tokens(self.reflect_lm)

            edit_list, _errors = parse_reflection_response(raw)
            try:
                new_prompt = apply(parent_prompt, edit_list.edits, alpha=self.alpha)
            except LengthCapExceeded as exc:
                logger.warning(
                    "iter=%d: length-cap exceeded for proposal #%d (%s); skipping",
                    iter_num,
                    n_proposals,
                    exc,
                )
                continue
            except (IndexError, ValueError) as exc:
                logger.warning(
                    "iter=%d: edit application failed for proposal #%d (%s); skipping",
                    iter_num,
                    n_proposals,
                    exc,
                )
                continue

            child_prompts = dict(parent.prompts)
            child_prompts[target_module] = new_prompt
            child = Candidate(
                id=str(uuid4()),
                parent_id=parent.id,
                prompts=child_prompts,
                edits_applied=list(edit_list.edits),
                pareto_score=0.0,
            )
            children.append(child)
            accumulated_lessons.extend(edit_list.lessons)

        n_after_apply = len(children)

        # 7. Race the children (only if we have at least one survivor).
        survivor: Candidate | None = None
        survivor_score = 0.0
        race_total_rollouts = 0
        race_total_skipped = 0
        if children:
            child_by_id = {c.id: c for c in children}

            def _race_eval_fn(candidate_id: str, instance: Any) -> float:
                cand = child_by_id[candidate_id]
                with run_context(
                    run_id=self._run_id,
                    round_num=iter_num,
                    candidate_id=candidate_id,
                    phase="race",
                ):
                    score, _fb = self._evaluate_candidate(
                        cand, instance, metric, feedback
                    )
                return float(score)

            race_result = self.race.race(
                candidates=[c.id for c in children],
                eval_fn=_race_eval_fn,
                instance_pool=valset or trainset or [{}],
            )
            race_total_rollouts = race_result.total_rollouts
            race_total_skipped = race_result.total_skipped
            survivor = child_by_id[race_result.survivor_id]
            survivor_score = race_result.survivor_score

        # 8. Acceptance.
        accepted = False
        if survivor is not None and survivor_score > parent.pareto_score:
            with run_context(
                run_id=self._run_id,
                round_num=iter_num,
                candidate_id=survivor.id,
                phase="valset_eval",
            ):
                survivor.pareto_score = self._evaluate_on_valset(
                    survivor, valset, metric
                )
            self._candidates.append(survivor)
            accepted = True

            # 9. Lesson book updates (only on accept — Q3: lessons live in
            # the reflection context, never auto-mutate prompts).
            for lesson_text in accumulated_lessons:
                if lesson_text and lesson_text.strip():
                    self.lesson_book.add(
                        lesson_text, cluster_origin=target_cluster.id
                    )

        # 10. Failure-pool update via embedded feedback strings (best-effort).
        # Only embed when feedback() returns something non-empty.
        for instance in (valset or [])[: max(1, len(valset) // 4)]:
            try:
                gold = _instance_gold(instance)
                pred = self._call_task_for_instance(parent, instance)
                fb = feedback(gold, pred, None)
            except Exception:  # pragma: no cover -- defensive
                continue
            if fb:
                vec = self.embedder.embed_one(fb)
                arr = np.asarray(vec).reshape(1, -1)
                if arr.shape[1] == 384:
                    self.clusterer.add(arr.astype(np.float32))

        # Increment lesson age + run TTL eviction once per iter.
        self.lesson_book.increment_age_and_evict()

        # 11. Telemetry.
        self._iteration_logs.append(
            IterationLog(
                iter_num=iter_num,
                target_cluster_id=target_cluster.id,
                target_module=target_module,
                n_proposals=n_proposals,
                n_candidates_after_apply=n_after_apply,
                race_total_rollouts=race_total_rollouts,
                race_total_skipped=race_total_skipped,
                survivor_score=float(survivor_score),
                accepted=accepted,
                cluster_k=self.clusterer.last_k or len(cluster_states),
                cluster_silhouette=float(self.clusterer.last_silhouette),
                n_active_lessons=len(
                    [le for le in self.lesson_book.lessons if le.status == "active"]
                ),
            )
        )

    # ---------------------------------------------------------- eval helpers
    def _call_reflect(self, prompt: str) -> str:
        """Call ``reflect_lm`` preferring its ``.reflect`` API when available."""
        if hasattr(self.reflect_lm, "reflect"):
            try:
                return self.reflect_lm.reflect(
                    prompt, guided_json_schema=EDIT_LIST_SCHEMA
                )
            except TypeError:
                return self.reflect_lm.reflect(prompt)
        return self.reflect_lm(prompt)

    def _call_task_for_instance(
        self, candidate: Candidate, instance: Any
    ) -> Any:
        """Materialize prompts and run task LM on one instance."""
        # Single-module systems use the only module's prompt; multi-module
        # systems concatenate (Phase 9 mock has only one module in practice).
        materialized = "\n\n".join(
            materialize(candidate.prompts[m]) for m in self._module_names
        )
        user_input = _instance_input(instance)
        full_prompt = f"{materialized}\n\nInput: {user_input}"
        try:
            pred = self.task_lm(full_prompt)
        except Exception as exc:  # pragma: no cover -- defensive
            logger.warning("task_lm call raised %s; returning empty pred", exc)
            pred = ""
        # Some LMs (dspy.LM) return list[str]; collapse to first.
        if isinstance(pred, list) and pred:
            pred = pred[0]
        self._account_tokens(self.task_lm)
        return pred

    def _evaluate_candidate(
        self,
        candidate: Candidate,
        instance: Any,
        metric: Callable[[Any, Any], float],
        feedback: Callable[..., str],
    ) -> tuple[float, str]:
        """Run candidate on a single instance; return ``(score, feedback)``."""
        gold = _instance_gold(instance)
        pred = self._call_task_for_instance(candidate, instance)
        try:
            score = float(metric(gold, pred))
        except Exception as exc:  # pragma: no cover -- defensive
            logger.warning("metric raised %s; treating as zero", exc)
            score = 0.0
        try:
            fb = feedback(gold, pred, None)
        except Exception:  # pragma: no cover
            fb = ""
        return score, str(fb)

    def _evaluate_on_valset(
        self,
        candidate: Candidate,
        valset: list[Any],
        metric: Callable[[Any, Any], float],
    ) -> float:
        if not valset:
            return 0.0
        scores: list[float] = []
        for instance in valset:
            gold = _instance_gold(instance)
            pred = self._call_task_for_instance(candidate, instance)
            try:
                scores.append(float(metric(gold, pred)))
            except Exception:  # pragma: no cover
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def _account_tokens(self, lm: Any) -> None:
        usage = getattr(lm, "_last_usage", None)
        if usage is None:
            return
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        self._cumulative_tokens += prompt_tokens + completion_tokens

    # ============================================ Checkpointable Protocol ==
    def save_state(self, path) -> None:
        """Persist optimizer state — delegates to :func:`scalpel.checkpoint.save_state`."""
        from scalpel.checkpoint import save_state as _save

        _save(self, path)

    def load_state(self, path) -> None:
        """Restore optimizer state — delegates to :func:`scalpel.checkpoint.load_state`."""
        from scalpel.checkpoint import load_state as _load

        _load(self, path)
