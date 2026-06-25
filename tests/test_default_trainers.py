# Copyright 2026 Firefly Software Foundation.
"""The plain ``AutoML()`` constructor must include installed boosting libraries by default.

The docs/benchmarks advertise "+ XGBoost / LightGBM / CatBoost when installed"; this asserts the
imperative path matches that claim (previously only the DI / agentic path did).
"""
from __future__ import annotations

import importlib.util


def test_default_trainers_include_installed_boosting_libraries() -> None:
    from fireflyframework_datascience.automl.facade import _default_trainers

    names = {t.name for t in _default_trainers()}
    assert {"random_forest", "linear", "hist_gradient_boosting"} <= names

    for lib, trainer_name in [("xgboost", "xgboost"), ("lightgbm", "lightgbm"), ("catboost", "catboost")]:
        if importlib.util.find_spec(lib) is not None:
            assert trainer_name in names, f"{trainer_name!r} must be a default trainer when {lib!r} is installed"
