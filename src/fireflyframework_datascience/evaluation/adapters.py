# Copyright 2026 Firefly Software Foundation.
"""scikit-learn metrics evaluator."""

from __future__ import annotations

from typing import Any

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.evaluation import EvaluationResult

_CLASSIFICATION = {TaskType.BINARY, TaskType.MULTICLASS, TaskType.CLASSIFICATION}
_LOWER_IS_BETTER = {"rmse", "mae", "log_loss"}
# sklearn accepts 0 / 1 / np.nan for zero_division; its type stub narrows it to str, so widen here.
_ZERO_DIV: Any = 0

# metric -> sklearn cross_val_score scoring string (always "greater is better")
_SCORING = {
    "accuracy": "accuracy",
    "f1": "f1_weighted",
    "roc_auc": "roc_auc",
    "roc_auc_ovr": "roc_auc_ovr_weighted",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


class SklearnMetricsEvaluator:
    """Computes a panel of metrics and provides CV scoring names."""

    name = "sklearn"

    def default_metric(self, task: TaskType) -> str:
        if task is TaskType.BINARY:
            return "roc_auc"
        if task in _CLASSIFICATION:
            return "accuracy"
        return "rmse"

    def scoring_name(self, task: TaskType, metric: str) -> str:
        if metric == "roc_auc" and task is TaskType.MULTICLASS:
            return _SCORING["roc_auc_ovr"]
        return _SCORING.get(metric, "accuracy" if task in _CLASSIFICATION else "r2")

    def greater_is_better(self, metric: str) -> bool:
        return metric not in _LOWER_IS_BETTER

    def evaluate(
        self, task: TaskType, y_true: Any, y_pred: Any, y_proba: Any = None, *, metric: str | None = None
    ) -> EvaluationResult:
        metrics = (
            self._classification(y_true, y_pred, y_proba, task)
            if task in _CLASSIFICATION
            else self._regression(y_true, y_pred)
        )
        primary = metric or self.default_metric(task)
        return EvaluationResult(metrics=metrics, primary_metric=primary, primary_value=float(metrics.get(primary, 0.0)))

    def _classification(self, y_true: Any, y_pred: Any, y_proba: Any, task: TaskType) -> dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=_ZERO_DIV)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=_ZERO_DIV)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=_ZERO_DIV)),
        }
        if y_proba is not None:
            self._add_proba_metrics(metrics, y_true, y_proba, task)
        return metrics

    def _add_proba_metrics(self, metrics: dict[str, float], y_true: Any, y_proba: Any, task: TaskType) -> None:
        from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

        try:
            if task is TaskType.BINARY:
                positive = y_proba[:, 1] if getattr(y_proba, "ndim", 1) == 2 else y_proba
                metrics["roc_auc"] = float(roc_auc_score(y_true, positive))
                # PR-AUC (key on imbalanced data) and the Brier score (probability quality / calibration)
                metrics["average_precision"] = float(average_precision_score(y_true, positive))
                metrics["brier_score"] = float(brier_score_loss(y_true, positive))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except (ValueError, IndexError):
            pass

    def _regression(self, y_true: Any, y_pred: Any) -> dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        return {
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }


__all__ = ["SklearnMetricsEvaluator"]
