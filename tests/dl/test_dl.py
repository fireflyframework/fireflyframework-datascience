# Copyright 2026 Firefly Software Foundation.
"""Tests for the DL module's verified sklearn-MLP reference trainer."""

from __future__ import annotations

import pytest

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


def test_torch_tabular_trainer_fits_and_predicts() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    from fireflyframework_datascience.dl.adapters import TorchTabularTrainer

    train, test = SklearnDatasetLoader().load("iris").train_test_split(random_state=0)
    model = TorchTabularTrainer(epochs=40).fit(train)
    assert model.name == "torch_tabular"
    preds = model.predict(test.X)
    assert len(preds) == test.n_rows
    accuracy = float((preds == test.y.to_numpy()).mean())
    assert accuracy > 0.7  # an MLP should comfortably beat chance on iris


def test_torch_tabular_regression() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    from fireflyframework_datascience.dl.adapters import TorchTabularTrainer

    train, test = SklearnDatasetLoader().load("diabetes").train_test_split(random_state=0)
    model = TorchTabularTrainer(epochs=60).fit(train)
    preds = model.predict(test.X)
    assert len(preds) == test.n_rows


@pytest.mark.integration
def test_tabpfn_predictor_fits_and_predicts() -> None:
    import os

    pytest.importorskip("tabpfn")
    if not os.getenv("TABPFN_TOKEN"):
        pytest.skip("TabPFN needs a one-time license token (TABPFN_TOKEN) to download weights")
    from fireflyframework_datascience.dl.adapters import TabPFNPredictor

    train, test = SklearnDatasetLoader().load("iris").train_test_split(random_state=0)
    model = TabPFNPredictor().fit(train)
    assert len(model.predict(test.X)) == test.n_rows
