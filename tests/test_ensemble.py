# Copyright 2026 Firefly Software Foundation.
"""AutoML(ensemble=True) stacks the top-k leaderboard models into one stronger model.

Single-best selection leaves accuracy on the table; stacking the strongest candidates is the standard
last-mile lift in production AutoML. Real data (breast_cancer), no fakes.
"""

from __future__ import annotations


def test_automl_ensemble_builds_a_competitive_stack() -> None:
    from sklearn.ensemble import StackingClassifier

    from fireflyframework_datascience.automl import AutoML
    from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

    ds = SklearnDatasetLoader().load("breast_cancer")
    train, test = ds.train_test_split(test_size=0.25, random_state=0)

    result = AutoML(cv=3, n_trials=1, random_state=0, ensemble=True).fit(train)

    assert result.best_model.name == "stacking_ensemble"
    assert isinstance(result.best_model.estimator, StackingClassifier)
    # the stack combines >= 2 base learners and is competitive on holdout
    assert len(result.best_model.params["members"]) >= 2
    assert result.evaluate(test).metrics["roc_auc"] > 0.95


def test_automl_without_ensemble_is_single_model() -> None:
    from sklearn.ensemble import StackingClassifier

    from fireflyframework_datascience.automl import AutoML
    from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

    ds = SklearnDatasetLoader().load("breast_cancer")
    train, _ = ds.train_test_split(test_size=0.25, random_state=0)
    result = AutoML(cv=3, n_trials=1, random_state=0).fit(train)
    assert not isinstance(result.best_model.estimator, StackingClassifier)
    assert result.best_model.name != "stacking_ensemble"
