# Copyright 2026 Firefly Software Foundation.
"""Shared preprocessing: impute + scale numeric, impute + one-hot categorical, around any estimator."""

from __future__ import annotations

from typing import Any


def build_preprocessor(X: Any) -> Any:
    """Return a dtype-aware ``ColumnTransformer`` (impute+scale numeric, impute+one-hot categorical).

    One-hot output is dense so the result feeds neural nets as well as classical estimators. Returns
    ``None`` when ``X`` has no columns to transform.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric)
        )
    if categorical:
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical))
    if not transformers:
        return None
    return ColumnTransformer(transformers, remainder="drop")


def build_pipeline(estimator: Any, X: Any) -> Any:
    """Wrap ``estimator`` in a sklearn ``Pipeline`` with dtype-aware preprocessing of ``X``."""
    from sklearn.pipeline import Pipeline

    preprocessor = build_preprocessor(X)
    if preprocessor is None:
        return Pipeline([("model", estimator)])
    return Pipeline([("prep", preprocessor), ("model", estimator)])


__all__ = ["build_pipeline", "build_preprocessor"]
