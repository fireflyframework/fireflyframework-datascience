# Copyright 2026 Firefly Software Foundation.
"""Tests for trainer adapters."""

from __future__ import annotations

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.models.adapters import (
    HistGradientBoostingTrainer,
    LinearTrainer,
    RandomForestTrainer,
    XGBoostTrainer,
)


def test_classification_vs_regression_estimator() -> None:
    trainer = RandomForestTrainer()
    assert trainer.make_estimator(TaskType.BINARY).__class__.__name__ == "RandomForestClassifier"
    assert trainer.make_estimator(TaskType.REGRESSION).__class__.__name__ == "RandomForestRegressor"


def test_linear_picks_logistic_or_ridge() -> None:
    trainer = LinearTrainer()
    assert trainer.make_estimator(TaskType.BINARY).__class__.__name__ == "LogisticRegression"
    assert trainer.make_estimator(TaskType.REGRESSION).__class__.__name__ == "Ridge"


def test_params_are_merged() -> None:
    est = RandomForestTrainer().make_estimator(TaskType.BINARY, {"n_estimators": 7})
    assert est.n_estimators == 7


def test_param_spaces_nonempty() -> None:
    for trainer in (RandomForestTrainer(), XGBoostTrainer(), HistGradientBoostingTrainer()):
        space = trainer.param_space(TaskType.BINARY)
        assert space, trainer.name
        assert all(isinstance(k, str) for k in space)


def test_all_support_both_tasks() -> None:
    for trainer in (RandomForestTrainer(), LinearTrainer(), HistGradientBoostingTrainer(), XGBoostTrainer()):
        assert trainer.supports(TaskType.BINARY)
        assert trainer.supports(TaskType.REGRESSION)
