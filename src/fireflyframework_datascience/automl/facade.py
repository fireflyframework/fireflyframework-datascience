# Copyright 2026 Firefly Software Foundation.
"""The AutoML engine: validate → cross-validate candidate models (optionally tuned) → fit the winner.

Usable two ways (the framework serves both app developers and data scientists):
  - imperative / notebook:  ``AutoML().fit(dataset)``
  - declarative / DI:        ``AutoML.from_context(app)`` (components resolved from the container)
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from fireflyframework_datascience.automl import AutoMLResult, LeaderboardEntry
from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.evaluation import MetricsEvaluatorPort
from fireflyframework_datascience.explainability import ExplainerPort
from fireflyframework_datascience.models import Model, TrainerPort
from fireflyframework_datascience.search import SearchPolicyPort
from fireflyframework_datascience.tracking import TrackerPort
from fireflyframework_datascience.validation import ValidatorPort

logger = logging.getLogger(__name__)


class AutoML:
    """Classical tabular AutoML over a set of trainers, with cross-validation model selection."""

    def __init__(
        self,
        trainers: Sequence[TrainerPort] | None = None,
        *,
        evaluator: MetricsEvaluatorPort | None = None,
        search_policy: SearchPolicyPort | None = None,
        validator: ValidatorPort | None = None,
        tracker: TrackerPort | None = None,
        explainer: ExplainerPort | None = None,
        cv: int = 5,
        n_trials: int = 20,
        random_state: int = 42,
    ) -> None:
        self._trainers = list(trainers) if trainers is not None else _default_trainers()
        self._evaluator = evaluator or _default_evaluator()
        self._search = search_policy or _default_search()
        self._validator = validator
        self._tracker = tracker
        self._explainer = explainer
        self._cv = cv
        self._n_trials = n_trials
        self._random_state = random_state

    @classmethod
    def from_context(cls, context: Any, **overrides: Any) -> AutoML:
        """Build an AutoML from a started :class:`ApplicationContext` (DI-wired components)."""
        container = context.container
        trainers = container.resolve_all(TrainerPort)
        return cls(
            trainers=trainers,
            evaluator=container.resolve_optional(MetricsEvaluatorPort) or _default_evaluator(),
            search_policy=container.resolve_optional(SearchPolicyPort) or _default_search(),
            validator=container.resolve_optional(ValidatorPort),
            tracker=container.resolve_optional(TrackerPort),
            explainer=container.resolve_optional(ExplainerPort),
            **overrides,
        )

    def fit(self, dataset: Dataset, *, task: TaskType | None = None, metric: str | None = None) -> AutoMLResult:
        task = task or dataset.task
        metric = metric or self._evaluator.default_metric(task)
        scoring = self._evaluator.scoring_name(task, metric)

        if self._validator is not None:
            self._validator.validate(dataset.X, dataset.y).raise_if_failed()

        candidates = [t for t in self._trainers if t.supports(task)]
        if not candidates:
            from fireflyframework_datascience.core.exceptions import FireflyDataScienceError

            raise FireflyDataScienceError(f"No registered trainer supports task {task!r}")

        run = self._tracker.start_run(f"automl:{dataset.name}") if self._tracker else None
        leaderboard: list[LeaderboardEntry] = []
        best: tuple[float, TrainerPort, dict[str, Any]] | None = None

        for trainer in candidates:
            space = trainer.param_space(task) if self._n_trials > 1 else {}
            result = self._search.optimize(
                self._objective(trainer, task, dataset, scoring),
                space,
                n_trials=self._n_trials,
                seed=self._random_state,
            )
            leaderboard.append(LeaderboardEntry(trainer.name, dict(result.best_params), result.best_score, metric))
            logger.info("AutoML candidate %s: cv %s=%.4f", trainer.name, metric, result.best_score)
            if best is None or result.best_score > best[0]:
                best = (result.best_score, trainer, dict(result.best_params))

        assert best is not None
        _, best_trainer, best_params = best
        estimator = self._pipeline(best_trainer.make_estimator(task, best_params), dataset.X)
        estimator.fit(dataset.X, dataset.y)
        model = Model(
            name=best_trainer.name,
            estimator=estimator,
            task=task,
            feature_names=list(dataset.feature_names),
            params=best_params,
        )
        leaderboard.sort(key=lambda e: e.cv_score, reverse=True)
        self._track_results(run, model, leaderboard, metric)
        return AutoMLResult(
            best_model=model,
            leaderboard=leaderboard,
            metric=metric,
            task=task,
            evaluator=self._evaluator,
            cv_scoring=scoring,
            explainer=self._explainer,
        )

    # -- internals --------------------------------------------------------

    def _objective(self, trainer: TrainerPort, task: TaskType, dataset: Dataset, scoring: str):
        from sklearn.model_selection import cross_val_score

        def objective(params: Mapping[str, Any]) -> float:
            estimator = self._pipeline(trainer.make_estimator(task, params), dataset.X)
            try:
                scores = cross_val_score(estimator, dataset.X, dataset.y, cv=self._cv, scoring=scoring)
            except Exception as exc:  # noqa: BLE001 - a failing candidate must not abort the whole search
                logger.warning("Candidate %s failed during CV: %s", trainer.name, exc)
                return float("-inf")
            return float(scores.mean())

        return objective

    def _pipeline(self, estimator: Any, X: Any) -> Any:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        numeric = X.select_dtypes(include="number").columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]
        transformers: list[tuple[str, Any, list[str]]] = []
        if numeric:
            # Impute + scale: scaling is harmless for trees and essential for linear models.
            num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
            transformers.append(("num", num_pipe, numeric))
        if categorical:
            cat_pipe = Pipeline(
                [("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]
            )
            transformers.append(("cat", cat_pipe, categorical))
        if not transformers:
            return Pipeline([("model", estimator)])
        pre = ColumnTransformer(transformers, remainder="drop")
        return Pipeline([("prep", pre), ("model", estimator)])

    def _track_results(self, run: Any, model: Model, leaderboard: list[LeaderboardEntry], metric: str) -> None:
        if self._tracker is None or run is None:
            return
        self._tracker.log_params({"winner": model.name, **{f"param.{k}": v for k, v in model.params.items()}})
        self._tracker.log_metrics({f"cv_{metric}": leaderboard[0].cv_score})
        self._tracker.log_model(model.estimator, artifact_name=model.name)
        self._tracker.end_run()


def _default_trainers() -> list[TrainerPort]:
    import importlib
    import importlib.util

    adapters = importlib.import_module("fireflyframework_datascience.models.adapters")
    trainers: list[TrainerPort] = [
        adapters.RandomForestTrainer(),
        adapters.LinearTrainer(),
        adapters.HistGradientBoostingTrainer(),
    ]
    # Match the documented "+ XGBoost / LightGBM / CatBoost when installed" behaviour (the DI and
    # agentic paths already do this) by including each boosting trainer whose library is importable.
    for lib, cls_name in (("xgboost", "XGBoostTrainer"), ("lightgbm", "LightGBMTrainer"), ("catboost", "CatBoostTrainer")):
        if importlib.util.find_spec(lib) is not None:
            trainers.append(getattr(adapters, cls_name)())
    return trainers


def _default_evaluator() -> MetricsEvaluatorPort:
    from fireflyframework_datascience.evaluation.adapters import SklearnMetricsEvaluator

    return SklearnMetricsEvaluator()


def _default_search() -> SearchPolicyPort:
    from fireflyframework_datascience.search.adapters import DefaultSearchPolicy

    return DefaultSearchPolicy()


__all__ = ["AutoML"]
