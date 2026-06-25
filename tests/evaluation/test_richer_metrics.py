# Copyright 2026 Firefly Software Foundation.
"""The evaluator should report PR-AUC (average precision) and the Brier score for classification.

These matter for imbalanced data (PR-AUC) and probability quality / calibration (Brier). Additive —
they do not change defaults or CV scoring. Real computed metrics, no fakes.
"""

from __future__ import annotations

import numpy as np

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.evaluation.adapters import SklearnMetricsEvaluator


def test_binary_evaluation_reports_pr_auc_and_brier() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_proba = np.array([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9], [0.6, 0.4], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1]])
    y_pred = y_proba.argmax(axis=1)

    result = SklearnMetricsEvaluator().evaluate(TaskType.BINARY, y_true, y_pred, y_proba, metric="roc_auc")

    assert "average_precision" in result.metrics
    assert "brier_score" in result.metrics
    assert 0.0 <= result.metrics["average_precision"] <= 1.0
    assert 0.0 <= result.metrics["brier_score"] <= 1.0
    # this is a strongly-separated set, so PR-AUC should be high and Brier low
    assert result.metrics["average_precision"] > 0.8
    assert result.metrics["brier_score"] < 0.2
