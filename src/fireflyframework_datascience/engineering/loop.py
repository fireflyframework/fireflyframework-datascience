# Copyright 2026 Firefly Software Foundation.
"""The agentic ML-engineering loop engine and its proposers/verifier.

``AgenticAutoML`` seeds a population (each trainer at defaults), then repeatedly asks the proposer to
reflect on the attempt history and suggest a better candidate. Every candidate is trained and
cross-validated by the classical engine, then judged by a :class:`Verifier` that requires it to beat a
trivial baseline (not merely to run). The loop is greedy with a patience budget.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.engineering import (
    AttemptRecord,
    CandidateProposer,
    EngineeringRun,
    SolutionCandidate,
    Verdict,
)
from fireflyframework_datascience.evaluation import MetricsEvaluatorPort
from fireflyframework_datascience.models import Model, TrainerPort
from fireflyframework_datascience.preprocessing import build_pipeline

logger = logging.getLogger(__name__)
_CLASSIFICATION = {TaskType.BINARY, TaskType.MULTICLASS, TaskType.CLASSIFICATION}

_PROPOSER_INSTRUCTIONS = (
    "You are an ML engineer running an AutoML search. Given a dataset description and the history of "
    "tried (trainer, params, cv_score) attempts, propose the single most promising next candidate to "
    "try. Choose a trainer from the allowed list and suggest hyperparameters as a JSON object. Prefer "
    "tweaking the best trainer so far, or trying an untried one. Return trainer, params_json, rationale."
)


class DeterministicVerifier:
    """Verifies a candidate by requiring a finite score that beats the trivial baseline by ``margin``."""

    name = "deterministic"

    def __init__(self, margin: float = 0.0) -> None:
        self._margin = margin

    def verify(self, dataset: Dataset, candidate: SolutionCandidate, score: float, baseline: float) -> Verdict:
        if not math.isfinite(score):
            return Verdict(False, "training failed or produced a non-finite score", score)
        if score <= baseline + self._margin:
            return Verdict(False, f"does not beat the trivial baseline ({score:.4f} <= {baseline:.4f})", score)
        return Verdict(True, f"beats trivial baseline by {score - baseline:+.4f}", score)


class AgenticAutoML:
    """LLM-proposes / classical-engine-decides AutoML loop with a distinct verification stage."""

    name = "agentic_automl"

    def __init__(
        self,
        proposer: CandidateProposer,
        *,
        trainers: dict[str, TrainerPort] | None = None,
        evaluator: MetricsEvaluatorPort | None = None,
        verifier: Any = None,
        cv: int = 5,
        max_iterations: int = 8,
        patience: int = 3,
        random_state: int = 42,
    ) -> None:
        self._proposer = proposer
        self._registry = trainers if trainers is not None else _default_registry()
        self._evaluator = evaluator or _default_evaluator()
        self._verifier = verifier or DeterministicVerifier()
        self._cv = cv
        self._max_iterations = max_iterations
        self._patience = patience
        self._random_state = random_state

    def solve(self, dataset: Dataset) -> EngineeringRun:
        task = dataset.task
        metric = self._evaluator.default_metric(task)
        scoring = self._evaluator.scoring_name(task, metric)
        baseline = self._trivial_baseline(dataset, task, scoring)
        names = list(self._registry)

        attempts: list[AttemptRecord] = []
        best: SolutionCandidate | None = None
        best_score = float("-inf")

        for candidate in self._proposer.propose_initial(dataset, names):
            record = self._attempt(dataset, candidate, task, scoring, baseline)
            attempts.append(record)
            if record.verdict.valid and record.score > best_score:
                best, best_score = candidate, record.score

        patience = self._patience
        for _ in range(self._max_iterations):
            candidate = self._proposer.propose_next(dataset, attempts, names)
            if candidate is None:
                break
            record = self._attempt(dataset, candidate, task, scoring, baseline)
            attempts.append(record)
            if record.verdict.valid and record.score > best_score:
                best, best_score, patience = candidate, record.score, self._patience
            else:
                patience -= 1
                if patience <= 0:
                    break

        model = self._fit_final(dataset, best, task) if best is not None else None
        logger.info("Agentic AutoML done: best=%s %s=%.4f (%d attempts)", best, metric, best_score, len(attempts))
        return EngineeringRun(
            best_candidate=best,
            best_score=best_score if best is not None else float("nan"),
            model=model,
            metric=metric,
            baseline_score=baseline,
            attempts=attempts,
        )

    # -- internals --------------------------------------------------------

    def _attempt(
        self, dataset: Dataset, candidate: SolutionCandidate, task: TaskType, scoring: str, baseline: float
    ) -> AttemptRecord:
        score = self._score(dataset, candidate, task, scoring)
        verdict = self._verifier.verify(dataset, candidate, score, baseline)
        return AttemptRecord(candidate=candidate, score=score, verdict=verdict)

    def _score(self, dataset: Dataset, candidate: SolutionCandidate, task: TaskType, scoring: str) -> float:
        trainer = self._registry.get(candidate.trainer)
        if trainer is None or not trainer.supports(task):
            return float("-inf")
        from sklearn.model_selection import cross_val_score

        estimator = build_pipeline(trainer.make_estimator(task, candidate.params), dataset.X)
        try:
            scores = cross_val_score(estimator, dataset.X, dataset.y, cv=self._cv, scoring=scoring)
        except Exception as exc:  # noqa: BLE001 - a failing candidate scores -inf, never aborts the loop
            logger.warning("Candidate %s failed: %s", candidate, exc)
            return float("-inf")
        return float(scores.mean())

    def _fit_final(self, dataset: Dataset, candidate: SolutionCandidate, task: TaskType) -> Model:
        trainer = self._registry[candidate.trainer]
        estimator = build_pipeline(trainer.make_estimator(task, candidate.params), dataset.X)
        estimator.fit(dataset.X, dataset.y)
        return Model(candidate.trainer, estimator, task, list(dataset.feature_names), dict(candidate.params))

    def _trivial_baseline(self, dataset: Dataset, task: TaskType, scoring: str) -> float:
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.model_selection import cross_val_score

        dummy = DummyClassifier(strategy="prior") if task in _CLASSIFICATION else DummyRegressor(strategy="mean")
        try:
            scores = cross_val_score(dummy, dataset.X, dataset.y, cv=self._cv, scoring=scoring)
        except Exception:  # noqa: BLE001
            return float("-inf")
        return float(scores.mean())


class AgentSolutionProposer:
    """LLM-backed proposer: seeds defaults, then reflects on history via a ``FireflyAgent``."""

    def __init__(self, *, model: Any = None, agent: Any = None) -> None:
        self._model = model
        self._explicit_agent = agent
        self._agent: Any = None  # built lazily on first use (no LLM client created at startup)

    def _get_agent(self) -> Any:
        if self._agent is None:
            self._agent = self._explicit_agent if self._explicit_agent is not None else self._build_agent(self._model)
        return self._agent

    def _build_agent(self, model: Any) -> Any:
        from fireflyframework_agentic.agents import FireflyAgent
        from pydantic import BaseModel, Field

        class _Solution(BaseModel):
            trainer: str = Field(description="trainer name from the allowed list")
            params_json: str = Field(default="{}", description="hyperparameters as a JSON object")
            rationale: str = Field(default="")

        self._solution_type = _Solution
        return FireflyAgent(
            name="ml_engineer",
            model=model or "openai:gpt-4o",
            instructions=_PROPOSER_INSTRUCTIONS,
            output_type=_Solution,
            default_middleware=False,
            auto_register=False,
        )

    def propose_initial(self, dataset: Dataset, trainers: list[str]) -> list[SolutionCandidate]:
        # Classical seeding: every trainer at its defaults. The LLM adds value during reflection.
        return [SolutionCandidate(trainer=name) for name in trainers]

    def propose_next(
        self, dataset: Dataset, history: list[AttemptRecord], trainers: list[str]
    ) -> SolutionCandidate | None:
        prompt = self._describe(dataset, history, trainers)
        result = self._get_agent().run_sync(prompt)
        solution = result.output
        trainer = getattr(solution, "trainer", "")
        if trainer not in trainers:
            trainer = self._best_trainer(history, trainers)
        return SolutionCandidate(
            trainer=trainer,
            params=_safe_json(getattr(solution, "params_json", "{}")),
            rationale=getattr(solution, "rationale", ""),
        )

    def _describe(self, dataset: Dataset, history: list[AttemptRecord], trainers: list[str]) -> str:
        ranked = sorted(history, key=lambda a: a.score, reverse=True)[:8]
        lines = [
            f"  {a.candidate.trainer} params={a.candidate.params} score={a.score:.4f} valid={a.verdict.valid}"
            for a in ranked
        ]
        return (
            f"Task: {dataset.task.value}. Allowed trainers: {trainers}.\n"
            f"History (best first):\n" + "\n".join(lines) + "\nPropose the next candidate."
        )

    def _best_trainer(self, history: list[AttemptRecord], trainers: list[str]) -> str:
        valid = [a for a in history if a.verdict.valid]
        if valid:
            return max(valid, key=lambda a: a.score).candidate.trainer
        return trainers[0]


def _safe_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def _default_registry() -> dict[str, TrainerPort]:
    from fireflyframework_datascience.models.adapters import (
        HistGradientBoostingTrainer,
        LinearTrainer,
        RandomForestTrainer,
    )

    registry: dict[str, TrainerPort] = {
        "linear": LinearTrainer(),
        "random_forest": RandomForestTrainer(),
        "hist_gradient_boosting": HistGradientBoostingTrainer(),
    }
    for name, module, cls_name in (
        ("xgboost", "fireflyframework_datascience.models.adapters", "XGBoostTrainer"),
        ("lightgbm", "fireflyframework_datascience.models.adapters", "LightGBMTrainer"),
        ("catboost", "fireflyframework_datascience.models.adapters", "CatBoostTrainer"),
    ):
        try:
            import importlib

            registry[name] = getattr(importlib.import_module(module), cls_name)()
        except Exception:  # noqa: BLE001 - optional boosting libs
            continue
    return registry


def _default_evaluator() -> MetricsEvaluatorPort:
    from fireflyframework_datascience.evaluation.adapters import SklearnMetricsEvaluator

    return SklearnMetricsEvaluator()


__all__ = ["AgenticAutoML", "AgentSolutionProposer", "DeterministicVerifier"]
