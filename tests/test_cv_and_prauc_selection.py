# Copyright 2026 Firefly Software Foundation.
"""Robust model selection: select on PR-AUC, and cross-validate with a custom splitter.

Two gaps these tests close (real data — breast_cancer — no fakes):
  1. ``average_precision`` (PR-AUC) is reported on holdout but was NOT a selectable CV metric:
     ``fit(metric="average_precision")`` silently fell back to selecting on accuracy. On imbalanced
     binary problems that picks the wrong winner.
  2. ``AutoML(cv=...)`` must accept a scikit-learn splitter (TimeSeriesSplit for temporal data,
     StratifiedKFold for explicit control), not only an int — so users can avoid silent leakage.
"""

from __future__ import annotations


def test_pr_auc_is_a_selectable_cv_metric() -> None:
    from fireflyframework_datascience.core.types import TaskType
    from fireflyframework_datascience.evaluation.adapters import SklearnMetricsEvaluator

    ev = SklearnMetricsEvaluator()
    # PR-AUC must map to sklearn's average_precision scorer (greater-is-better), not fall back to accuracy.
    assert ev.scoring_name(TaskType.BINARY, "average_precision") == "average_precision"
    assert ev.greater_is_better("average_precision") is True


def test_automl_selects_on_pr_auc() -> None:
    from fireflyframework_datascience.automl import AutoML
    from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

    ds = SklearnDatasetLoader().load("breast_cancer")
    train, _ = ds.train_test_split(test_size=0.25, random_state=0)

    result = AutoML(cv=3, n_trials=1, random_state=0).fit(train, metric="average_precision")

    assert result.metric == "average_precision"
    # The winner was selected using the PR-AUC scorer, not the accuracy fallback.
    assert result.cv_scoring == "average_precision"
    assert result.leaderboard[0].metric == "average_precision"
    # PR-AUC is a probability in [0, 1] (a high-scoring task); a fallback would surface a different scale.
    assert 0.5 <= result.leaderboard[0].cv_score <= 1.0


def test_automl_accepts_a_cv_splitter() -> None:
    from sklearn.model_selection import StratifiedKFold

    from fireflyframework_datascience.automl import AutoML
    from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

    ds = SklearnDatasetLoader().load("breast_cancer")
    train, test = ds.train_test_split(test_size=0.25, random_state=0)

    # A caller-supplied splitter must flow through to cross-validation and yield a fitted, usable model.
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    result = AutoML(cv=splitter, n_trials=1, random_state=0).fit(train)

    assert result.best_model is not None
    assert result.evaluate(test).metrics["roc_auc"] > 0.95


def test_automl_accepts_a_time_series_splitter() -> None:
    from sklearn.model_selection import TimeSeriesSplit

    from fireflyframework_datascience.automl import AutoML
    from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

    ds = SklearnDatasetLoader().load("breast_cancer")
    train, _ = ds.train_test_split(test_size=0.25, random_state=0)

    # Forward-chaining CV (no future leakage) must be accepted and produce a valid leaderboard score.
    result = AutoML(cv=TimeSeriesSplit(n_splits=3), n_trials=1, random_state=0).fit(train)

    assert result.best_model is not None
    assert result.leaderboard[0].cv_score > 0.0
