# Copyright 2026 Firefly Software Foundation.
"""AutoML module — backend port, leaderboard, and result (import-light).

The heavy :class:`~fireflyframework_datascience.automl.facade.AutoML` engine is exposed lazily via
module ``__getattr__`` so ``from fireflyframework_datascience.automl import AutoML`` works without
importing scikit-learn until it is actually used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.evaluation import EvaluationResult, MetricsEvaluatorPort
from fireflyframework_datascience.models import Model

if TYPE_CHECKING:
    from fireflyframework_datascience.automl.facade import AutoML


@dataclass
class LeaderboardEntry:
    """One model's cross-validation result in the AutoML leaderboard."""

    model_name: str
    params: dict[str, Any]
    cv_score: float
    metric: str

    def __str__(self) -> str:
        return f"{self.model_name:<24} {self.metric}={self.cv_score:.4f}"


@dataclass
class AutoMLResult:
    """The fitted winner plus the full leaderboard and the evaluator used."""

    best_model: Model
    leaderboard: list[LeaderboardEntry]
    metric: str
    task: TaskType
    evaluator: MetricsEvaluatorPort
    cv_scoring: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def best_score(self) -> float:
        return self.leaderboard[0].cv_score if self.leaderboard else float("nan")

    def predict(self, X: Any) -> Any:
        return self.best_model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self.best_model.predict_proba(X)

    def evaluate(self, dataset: Dataset) -> EvaluationResult:
        """Score the winning model on a held-out dataset."""
        y_pred = self.predict(dataset.X)
        y_proba = None
        if self.task.is_classification() and hasattr(self.best_model.estimator, "predict_proba"):
            try:
                y_proba = self.predict_proba(dataset.X)
            except (ValueError, AttributeError):
                y_proba = None
        return self.evaluator.evaluate(self.task, dataset.y, y_pred, y_proba, metric=self.metric)

    def leaderboard_table(self) -> str:
        return "\n".join(str(entry) for entry in self.leaderboard)


@runtime_checkable
class AutoMLBackendPort(Protocol):
    """Fits an AutoML pipeline and returns the best model."""

    def fit(self, dataset: Dataset, *, task: TaskType | None = None, metric: str | None = None) -> AutoMLResult: ...


def __getattr__(name: str) -> Any:  # PEP 562 lazy export of the heavy engine
    if name == "AutoML":
        from fireflyframework_datascience.automl.facade import AutoML

        return AutoML
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AutoML", "AutoMLBackendPort", "AutoMLResult", "LeaderboardEntry"]
