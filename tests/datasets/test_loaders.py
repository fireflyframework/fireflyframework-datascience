# Copyright 2026 Firefly Software Foundation.
"""Tests for dataset loaders and the Dataset container."""

from __future__ import annotations

import pandas as pd

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import infer_task
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader


def test_load_breast_cancer() -> None:
    ds = SklearnDatasetLoader().load("breast_cancer")
    assert ds.task is TaskType.BINARY
    assert ds.n_features == 30
    assert ds.n_rows == 569
    assert ds.has_target


def test_load_with_prefix() -> None:
    ds = SklearnDatasetLoader().load("sklearn:wine")
    assert ds.task is TaskType.MULTICLASS
    assert ds.name == "wine"


def test_can_load() -> None:
    loader = SklearnDatasetLoader()
    assert loader.can_load("iris")
    assert loader.can_load("sklearn:diabetes")
    assert not loader.can_load("nonexistent_dataset")


def test_train_test_split_preserves_rows() -> None:
    ds = SklearnDatasetLoader().load("iris")
    train, test = ds.train_test_split(test_size=0.2, random_state=0)
    assert train.n_rows + test.n_rows == ds.n_rows
    assert train.task is ds.task
    assert test.name.endswith("[test]")


def test_infer_task() -> None:
    assert infer_task(pd.Series([0, 1, 0, 1])) is TaskType.BINARY
    assert infer_task(pd.Series([0, 1, 2, 1, 2])) is TaskType.MULTICLASS
    assert infer_task(pd.Series([float(i) for i in range(50)])) is TaskType.REGRESSION
