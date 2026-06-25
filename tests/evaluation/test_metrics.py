# Copyright 2026 Firefly Software Foundation.
"""Tests for the scikit-learn metrics evaluator."""

from __future__ import annotations

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.evaluation.adapters import SklearnMetricsEvaluator


def test_classification_metrics() -> None:
    evaluator = SklearnMetricsEvaluator()
    result = evaluator.evaluate(TaskType.BINARY, [0, 1, 1, 0], [0, 1, 0, 0])
    assert "accuracy" in result.metrics
    assert "f1" in result.metrics
    assert result.primary_metric == "roc_auc"


def test_regression_metrics() -> None:
    evaluator = SklearnMetricsEvaluator()
    result = evaluator.evaluate(TaskType.REGRESSION, [1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    assert result.metrics["r2"] > 0.9
    assert result.primary_metric == "rmse"
    assert result.metrics["rmse"] > 0


def test_scoring_names_and_direction() -> None:
    evaluator = SklearnMetricsEvaluator()
    assert evaluator.scoring_name(TaskType.BINARY, "roc_auc") == "roc_auc"
    assert evaluator.scoring_name(TaskType.MULTICLASS, "roc_auc") == "roc_auc_ovr_weighted"
    assert evaluator.scoring_name(TaskType.REGRESSION, "rmse") == "neg_root_mean_squared_error"
    assert evaluator.greater_is_better("accuracy") is True
    assert evaluator.greater_is_better("rmse") is False


def test_default_metric_per_task() -> None:
    evaluator = SklearnMetricsEvaluator()
    assert evaluator.default_metric(TaskType.BINARY) == "roc_auc"
    assert evaluator.default_metric(TaskType.MULTICLASS) == "accuracy"
    assert evaluator.default_metric(TaskType.REGRESSION) == "rmse"
