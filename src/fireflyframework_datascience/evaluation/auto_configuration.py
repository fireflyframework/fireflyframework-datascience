# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for the evaluation module."""

from __future__ import annotations

from fireflyframework_datascience.container.conditions import (
    auto_configuration,
    conditional_on_class,
    conditional_on_missing_bean,
)
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.evaluation import MetricsEvaluatorPort


@auto_configuration
@conditional_on_class("sklearn")
@configuration
class EvaluationAutoConfiguration:
    """Registers the scikit-learn metrics evaluator (unless the user supplied one)."""

    @bean(name="sklearn_metrics_evaluator", primary=True)
    @conditional_on_missing_bean(MetricsEvaluatorPort)
    def metrics_evaluator(self) -> MetricsEvaluatorPort:
        from fireflyframework_datascience.evaluation.adapters import SklearnMetricsEvaluator

        return SklearnMetricsEvaluator()
