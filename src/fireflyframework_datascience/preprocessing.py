# Copyright 2026 Firefly Software Foundation.
"""Shared preprocessing: impute + scale numeric, impute + one-hot categorical, around any estimator."""

from __future__ import annotations

from typing import Any


def build_pipeline(estimator: Any, X: Any) -> Any:
    """Wrap ``estimator`` in a sklearn ``Pipeline`` with dtype-aware preprocessing of ``X``."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        transformers.append(("num", num_pipe, numeric))
    if categorical:
        cat_pipe = Pipeline(
            [("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]
        )
        transformers.append(("cat", cat_pipe, categorical))
    if not transformers:
        return Pipeline([("model", estimator)])
    return Pipeline([("prep", ColumnTransformer(transformers, remainder="drop")), ("model", estimator)])


__all__ = ["build_pipeline"]
