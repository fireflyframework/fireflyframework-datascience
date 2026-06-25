# Copyright 2026 Firefly Software Foundation.
"""AutoML(calibrate=True) wraps the winning classifier so its probabilities are trustworthy.

Tree/boosting probabilities are often poorly calibrated; risk-/cost-sensitive decisions need
trustworthy probabilities. Real data (breast_cancer), no fakes.
"""

from __future__ import annotations


def test_automl_calibrate_wraps_winner_with_valid_probabilities() -> None:
    from sklearn.calibration import CalibratedClassifierCV

    from fireflyframework_datascience.automl import AutoML
    from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

    ds = SklearnDatasetLoader().load("breast_cancer")
    train, test = ds.train_test_split(test_size=0.25, random_state=0)

    result = AutoML(cv=3, n_trials=1, random_state=0, calibrate=True).fit(train)

    # the winner is wrapped in a calibrator and still exposes valid probabilities
    assert isinstance(result.best_model.estimator, CalibratedClassifierCV)
    proba = result.predict_proba(test.X)
    assert proba.shape[0] == test.n_rows
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
    # calibrated probabilities are trustworthy: a finite, low Brier score on holdout
    brier = result.evaluate(test).metrics["brier_score"]
    assert 0.0 <= brier <= 0.25


def test_automl_without_calibrate_is_unchanged() -> None:
    from sklearn.calibration import CalibratedClassifierCV

    from fireflyframework_datascience.automl import AutoML
    from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

    ds = SklearnDatasetLoader().load("breast_cancer")
    train, _ = ds.train_test_split(test_size=0.25, random_state=0)
    result = AutoML(cv=3, n_trials=1, random_state=0).fit(train)
    assert not isinstance(result.best_model.estimator, CalibratedClassifierCV)
