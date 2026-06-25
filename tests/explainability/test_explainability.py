# Copyright 2026 Firefly Software Foundation.
"""Explainability tests — real data, no fakes, no mocks.

The contract we assert is the one users actually rely on: a global explanation must rank a genuinely
informative feature above an injected pure-noise column. We use scikit-learn's real ``breast_cancer``
dataset plus a deterministic noise column whose importance must be ~0.
"""

from __future__ import annotations

import numpy as np
import pytest

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.models import Model


def _breast_cancer_with_noise() -> tuple[Dataset, Model]:
    """A real fitted RandomForest on breast_cancer + one pure-noise column."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    X["__noise__"] = np.random.default_rng(0).normal(size=len(X))  # pure noise, no signal
    y = raw.target
    estimator = Pipeline([("model", RandomForestClassifier(n_estimators=80, random_state=0))]).fit(X, y)
    cols = list(X.columns)
    model = Model(name="random_forest", estimator=estimator, task=TaskType.BINARY, feature_names=cols)
    dataset = Dataset(name="breast_cancer", X=X, y=y, task=TaskType.BINARY, target_name="target", feature_names=cols)
    return dataset, model


def test_permutation_importance_ranks_signal_above_noise() -> None:
    from fireflyframework_datascience.explainability import GlobalExplanation
    from fireflyframework_datascience.explainability.adapters import PermutationImportanceExplainer

    dataset, model = _breast_cancer_with_noise()
    explainer = PermutationImportanceExplainer(n_repeats=5, random_state=0)

    explanation = explainer.explain_global(model, dataset)

    assert isinstance(explanation, GlobalExplanation)
    # one importance per input feature, keyed by the real column names
    assert set(explanation.feature_importances) == set(dataset.feature_names)
    # the pure-noise column carries essentially no importance...
    assert explanation.feature_importances["__noise__"] <= 0.005
    # ...and ranks strictly below the most informative real feature
    assert explanation.feature_importances["__noise__"] < max(explanation.feature_importances.values())
    # a genuinely informative breast-cancer feature surfaces in the top of the ranking
    top_names = [name for name, _ in explanation.top(8)]
    assert any(f in top_names for f in ("worst perimeter", "worst concave points", "worst radius", "worst area"))
    assert "__noise__" not in top_names


def test_automl_result_explains_the_winner_on_real_data() -> None:
    from fireflyframework_datascience.automl import AutoML
    from fireflyframework_datascience.explainability import GlobalExplanation

    dataset, _ = _breast_cancer_with_noise()
    train, test = dataset.train_test_split(test_size=0.3, random_state=0)

    result = AutoML(cv=3, n_trials=1, random_state=0).fit(train)
    explanation = result.explain(test)

    assert isinstance(explanation, GlobalExplanation)
    assert set(explanation.feature_importances) == set(dataset.feature_names)
    # the injected noise column is the least informative through the full AutoML pipeline
    assert explanation.feature_importances["__noise__"] < max(explanation.feature_importances.values())


def test_explainer_is_auto_configured_in_the_container() -> None:
    from fireflyframework_datascience import FireflyDataScienceApplication
    from fireflyframework_datascience.explainability import ExplainerPort

    app = FireflyDataScienceApplication.run(print_output=False)
    explainer = app.container.resolve_optional(ExplainerPort)

    assert explainer is not None
    assert isinstance(explainer, ExplainerPort)
    # the dependency-free default, or SHAP when the optional `explain` extra is installed
    assert explainer.name in {"permutation_importance", "shap"}


def test_shap_explainer_global_and_local_on_real_data() -> None:
    pytest.importorskip("shap")  # only runs when the optional `explain` extra is installed
    from fireflyframework_datascience.explainability import GlobalExplanation, LocalExplanation
    from fireflyframework_datascience.explainability.adapters import ShapExplainer

    dataset, model = _breast_cancer_with_noise()
    explainer = ShapExplainer(max_samples=40)

    global_exp = explainer.explain_global(model, dataset)
    assert isinstance(global_exp, GlobalExplanation)
    assert set(global_exp.feature_importances) == set(dataset.feature_names)
    assert global_exp.feature_importances["__noise__"] < max(global_exp.feature_importances.values())

    local = explainer.explain_local(model, dataset.X.iloc[:3])
    assert len(local) == 3
    assert all(isinstance(item, LocalExplanation) for item in local)
    assert set(local[0].contributions) == set(dataset.feature_names)
