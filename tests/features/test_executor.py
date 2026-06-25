# Copyright 2026 Firefly Software Foundation.
"""Tests for the secure feature-code executor."""

from __future__ import annotations

import pandas as pd
import pytest

from fireflyframework_datascience.features.executor import FeatureCodeExecutor, FeatureExecutionError


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})


def test_safe_code_adds_column(frame: pd.DataFrame) -> None:
    out = FeatureCodeExecutor().execute("df['ratio'] = df['a'] / (df['b'] + 1)", frame)
    assert "ratio" in out.columns
    assert "a" in out.columns  # original columns preserved


def test_does_not_mutate_input(frame: pd.DataFrame) -> None:
    FeatureCodeExecutor().execute("df['c'] = df['a'] + df['b']", frame)
    assert "c" not in frame.columns  # executor works on a copy


def test_import_is_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(FeatureExecutionError, match="[Uu]nsafe"):
        FeatureCodeExecutor().execute("import os\ndf['x'] = 1", frame)


def test_dunder_access_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(FeatureExecutionError):
        FeatureCodeExecutor().execute("df['x'] = ().__class__.__bases__", frame)


def test_runtime_error_is_typed(frame: pd.DataFrame) -> None:
    with pytest.raises(FeatureExecutionError, match="runtime"):
        FeatureCodeExecutor().execute("df['x'] = df['nonexistent'] * 2", frame)


def test_no_new_column_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(FeatureExecutionError, match="no new column"):
        FeatureCodeExecutor().execute("x = 1 + 1", frame)


def test_non_numeric_feature_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(FeatureExecutionError, match="not numeric"):
        FeatureCodeExecutor().execute("df['label'] = 'cat'", frame)
