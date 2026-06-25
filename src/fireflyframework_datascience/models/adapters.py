# Copyright 2026 Firefly Software Foundation.
"""Trainer adapters: scikit-learn estimators and gradient-boosting libraries.

Each trainer builds an *unfitted* estimator and declares a hyperparameter search space. The AutoML
facade wraps the estimator in a preprocessing pipeline, cross-validates it, and fits the winner.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.tuning import CategoricalParam, FloatParam, IntParam, ParamSpace

_CLASSIFICATION = {TaskType.BINARY, TaskType.MULTICLASS, TaskType.CLASSIFICATION}


def _merge(defaults: dict[str, Any], params: Mapping[str, Any] | None) -> dict[str, Any]:
    return {**defaults, **(dict(params) if params else {})}


class RandomForestTrainer:
    name = "random_forest"

    def supports(self, task: TaskType) -> bool:
        return task in _CLASSIFICATION or task is TaskType.REGRESSION

    def make_estimator(self, task: TaskType, params: Mapping[str, Any] | None = None) -> Any:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cfg = _merge({"n_estimators": 200, "n_jobs": -1, "random_state": 42}, params)
        cls = RandomForestClassifier if task in _CLASSIFICATION else RandomForestRegressor
        return cls(**cfg)

    def param_space(self, task: TaskType) -> ParamSpace:
        return {
            "n_estimators": IntParam(100, 500, step=50),
            "max_depth": IntParam(3, 24),
            "min_samples_split": IntParam(2, 16),
            "max_features": CategoricalParam(("sqrt", "log2", None)),
        }


class LinearTrainer:
    name = "linear"

    def supports(self, task: TaskType) -> bool:
        return task in _CLASSIFICATION or task is TaskType.REGRESSION

    def make_estimator(self, task: TaskType, params: Mapping[str, Any] | None = None) -> Any:
        if task in _CLASSIFICATION:
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(**_merge({"max_iter": 1000, "C": 1.0}, params))
        from sklearn.linear_model import Ridge

        return Ridge(**_merge({"alpha": 1.0}, params))

    def param_space(self, task: TaskType) -> ParamSpace:
        if task in _CLASSIFICATION:
            return {"C": FloatParam(1e-3, 1e2, log=True)}
        return {"alpha": FloatParam(1e-3, 1e2, log=True)}


class HistGradientBoostingTrainer:
    name = "hist_gradient_boosting"

    def supports(self, task: TaskType) -> bool:
        return task in _CLASSIFICATION or task is TaskType.REGRESSION

    def make_estimator(self, task: TaskType, params: Mapping[str, Any] | None = None) -> Any:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        cfg = _merge({"learning_rate": 0.1, "max_iter": 200, "random_state": 42}, params)
        cls = HistGradientBoostingClassifier if task in _CLASSIFICATION else HistGradientBoostingRegressor
        return cls(**cfg)

    def param_space(self, task: TaskType) -> ParamSpace:
        return {
            "learning_rate": FloatParam(0.01, 0.3, log=True),
            "max_iter": IntParam(100, 500, step=50),
            "max_leaf_nodes": IntParam(15, 63),
            "l2_regularization": FloatParam(1e-6, 1.0, log=True),
        }


class XGBoostTrainer:
    name = "xgboost"

    def supports(self, task: TaskType) -> bool:
        return task in _CLASSIFICATION or task is TaskType.REGRESSION

    def make_estimator(self, task: TaskType, params: Mapping[str, Any] | None = None) -> Any:
        import xgboost as xgb

        cfg = _merge(
            {
                "n_estimators": 300,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "tree_method": "hist",
                "n_jobs": -1,
                "verbosity": 0,
                "random_state": 42,
            },
            params,
        )
        cls = xgb.XGBClassifier if task in _CLASSIFICATION else xgb.XGBRegressor
        return cls(**cfg)

    def param_space(self, task: TaskType) -> ParamSpace:
        return {
            "n_estimators": IntParam(100, 600, step=50),
            "max_depth": IntParam(3, 10),
            "learning_rate": FloatParam(0.01, 0.3, log=True),
            "subsample": FloatParam(0.6, 1.0),
            "colsample_bytree": FloatParam(0.6, 1.0),
        }


class LightGBMTrainer:
    name = "lightgbm"

    def supports(self, task: TaskType) -> bool:
        return task in _CLASSIFICATION or task is TaskType.REGRESSION

    def make_estimator(self, task: TaskType, params: Mapping[str, Any] | None = None) -> Any:
        import lightgbm as lgb

        cfg = _merge(
            {
                "n_estimators": 300,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "n_jobs": -1,
                "verbose": -1,
                "random_state": 42,
            },
            params,
        )
        cls = lgb.LGBMClassifier if task in _CLASSIFICATION else lgb.LGBMRegressor
        return cls(**cfg)

    def param_space(self, task: TaskType) -> ParamSpace:
        return {
            "n_estimators": IntParam(100, 600, step=50),
            "num_leaves": IntParam(15, 127),
            "learning_rate": FloatParam(0.01, 0.3, log=True),
            "subsample": FloatParam(0.6, 1.0),
        }


class CatBoostTrainer:
    name = "catboost"

    def supports(self, task: TaskType) -> bool:
        return task in _CLASSIFICATION or task is TaskType.REGRESSION

    def make_estimator(self, task: TaskType, params: Mapping[str, Any] | None = None) -> Any:
        from catboost import CatBoostClassifier, CatBoostRegressor

        cfg = _merge(
            {
                "iterations": 300,
                "learning_rate": 0.1,
                "depth": 6,
                "verbose": 0,
                "allow_writing_files": False,
                "random_seed": 42,
            },
            params,
        )
        cls = CatBoostClassifier if task in _CLASSIFICATION else CatBoostRegressor
        return cls(**cfg)

    def param_space(self, task: TaskType) -> ParamSpace:
        return {
            "iterations": IntParam(100, 600, step=50),
            "depth": IntParam(3, 10),
            "learning_rate": FloatParam(0.01, 0.3, log=True),
        }


__all__ = [
    "CatBoostTrainer",
    "HistGradientBoostingTrainer",
    "LightGBMTrainer",
    "LinearTrainer",
    "RandomForestTrainer",
    "XGBoostTrainer",
]
