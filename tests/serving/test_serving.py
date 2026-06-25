# Copyright 2026 Firefly Software Foundation.
"""Tests for the in-process model server."""

from __future__ import annotations

import pytest

from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.core.exceptions import FireflyDataScienceError
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader
from fireflyframework_datascience.serving import LocalModelServer


def test_local_server_serves_predictions() -> None:
    train, test = SklearnDatasetLoader().load("iris").train_test_split(random_state=0)
    model = AutoML().fit(train).best_model

    server = LocalModelServer()
    server.load(model)
    preds = server.predict(test.X)
    assert len(preds) == test.n_rows
    assert server.model is model


def test_local_server_requires_loaded_model() -> None:
    with pytest.raises(FireflyDataScienceError, match="No model loaded"):
        LocalModelServer().predict(None)
