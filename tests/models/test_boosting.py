# Copyright 2026 Firefly Software Foundation.
"""Explicit tests for the gradient-boosting trainers (XGBoost, LightGBM, CatBoost)."""

from __future__ import annotations

import pytest

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader
from fireflyframework_datascience.preprocessing import build_pipeline

_BOOSTERS = [
    ("xgboost", "XGBoostTrainer"),
    ("lightgbm", "LightGBMTrainer"),
    ("catboost", "CatBoostTrainer"),
]


def _trainer(lib: str, cls_name: str):  # type: ignore[no-untyped-def]
    pytest.importorskip(lib)
    import fireflyframework_datascience.models.adapters as adapters

    return getattr(adapters, cls_name)()


@pytest.mark.parametrize(("lib", "cls_name"), _BOOSTERS)
def test_boosting_classification(lib: str, cls_name: str) -> None:
    trainer = _trainer(lib, cls_name)
    assert trainer.supports(TaskType.BINARY)
    assert trainer.param_space(TaskType.BINARY)  # declares a search space
    train, test = SklearnDatasetLoader().load("breast_cancer").train_test_split(random_state=0)
    est = build_pipeline(trainer.make_estimator(TaskType.BINARY), train.X)
    est.fit(train.X, train.y)
    acc = float((est.predict(test.X) == test.y.to_numpy()).mean())
    assert acc > 0.92, (trainer.name, acc)


@pytest.mark.parametrize(("lib", "cls_name"), _BOOSTERS)
def test_boosting_regression(lib: str, cls_name: str) -> None:
    trainer = _trainer(lib, cls_name)
    assert trainer.supports(TaskType.REGRESSION)
    train, test = SklearnDatasetLoader().load("diabetes").train_test_split(random_state=0)
    est = build_pipeline(trainer.make_estimator(TaskType.REGRESSION), train.X)
    est.fit(train.X, train.y)
    assert len(est.predict(test.X)) == test.n_rows


def test_boosting_params_are_applied() -> None:
    trainer = _trainer("xgboost", "XGBoostTrainer")
    est = trainer.make_estimator(TaskType.BINARY, {"n_estimators": 17, "max_depth": 4})
    assert est.n_estimators == 17
    assert est.max_depth == 4
