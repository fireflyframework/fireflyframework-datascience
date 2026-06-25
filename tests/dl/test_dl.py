# Copyright 2026 Firefly Software Foundation.
"""Tests for the DL module's verified sklearn-MLP reference trainer."""

from __future__ import annotations

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader
from fireflyframework_datascience.dl.adapters import MLPTrainer


def test_mlp_trainer_fits_and_predicts() -> None:
    train, test = SklearnDatasetLoader().load("iris").train_test_split(random_state=0)
    model = MLPTrainer().fit(train)
    assert model.name == "mlp"
    assert len(model.predict(test.X)) == test.n_rows


def test_mlp_supports_both_tasks() -> None:
    trainer = MLPTrainer()
    assert trainer.supports(TaskType.MULTICLASS)
    assert trainer.supports(TaskType.REGRESSION)
