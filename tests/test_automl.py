# Copyright 2026 Firefly Software Foundation.
"""End-to-end AutoML tests on real scikit-learn datasets (offline)."""

from __future__ import annotations

from fireflyframework_datascience import FireflyDataScienceApplication
from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader
from fireflyframework_datascience.models.adapters import RandomForestTrainer
from fireflyframework_datascience.search.adapters import OptunaSearchPolicy


def test_classification_end_to_end() -> None:
    ds = SklearnDatasetLoader().load("breast_cancer")
    train, test = ds.train_test_split(random_state=0)
    result = AutoML().fit(train)

    assert result.task is TaskType.BINARY
    assert len(result.leaderboard) == 3  # RF, Linear, HistGB
    assert result.leaderboard[0].cv_score >= result.leaderboard[-1].cv_score  # sorted desc

    evaluation = result.evaluate(test)
    assert evaluation.metrics["accuracy"] > 0.9
    assert len(result.predict(test.X)) == test.n_rows


def test_regression_end_to_end() -> None:
    ds = SklearnDatasetLoader().load("diabetes")
    train, test = ds.train_test_split(random_state=0)
    result = AutoML().fit(train)

    assert result.task is TaskType.REGRESSION
    evaluation = result.evaluate(test)
    assert evaluation.metrics["r2"] > 0.1  # better than the mean predictor
    assert evaluation.metrics["rmse"] > 0


def test_optuna_tuning_produces_params() -> None:
    ds = SklearnDatasetLoader().load("breast_cancer")
    train, _ = ds.train_test_split(random_state=0)
    result = AutoML(trainers=[RandomForestTrainer()], search_policy=OptunaSearchPolicy(), n_trials=4, cv=3).fit(train)

    assert len(result.leaderboard) == 1
    assert result.best_model.params  # tuned params were applied


def test_from_context_wires_all_trainers() -> None:
    app = FireflyDataScienceApplication.run(print_output=False)
    automl = AutoML.from_context(app, n_trials=1, cv=3)

    ds = SklearnDatasetLoader().load("iris")
    train, test = ds.train_test_split(random_state=0)
    result = automl.fit(train)

    assert result.task is TaskType.MULTICLASS
    assert len(result.leaderboard) >= 3  # DI wired RF/Linear/HistGB (+ boosting if installed)
    assert result.evaluate(test).metrics["accuracy"] > 0.8


def test_model_save_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ds = SklearnDatasetLoader().load("iris")
    train, test = ds.train_test_split(random_state=0)
    result = AutoML().fit(train)
    path = tmp_path / "model.joblib"
    result.best_model.save(path)

    from fireflyframework_datascience.models import Model

    loaded = Model.load(path)
    assert len(loaded.predict(test.X)) == test.n_rows
