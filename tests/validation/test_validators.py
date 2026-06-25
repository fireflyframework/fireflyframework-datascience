# Copyright 2026 Firefly Software Foundation.
"""Tests for the basic data validator."""

from __future__ import annotations

import pandas as pd

from fireflyframework_datascience.validation.adapters import BasicValidator


def test_clean_data_passes() -> None:
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    report = BasicValidator().validate(X, pd.Series([0, 1, 0]))
    assert report.ok
    assert report.stats["n_rows"] == 3
    assert report.stats["n_cols"] == 2


def test_null_target_fails() -> None:
    X = pd.DataFrame({"a": [1, 2, 3]})
    report = BasicValidator().validate(X, pd.Series([0, None, 1]))
    assert not report.ok
    assert any("Target" in e for e in report.errors)


def test_constant_column_warns() -> None:
    X = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    report = BasicValidator().validate(X)
    assert report.ok
    assert any("onstant" in w for w in report.warnings)


def test_empty_data_fails() -> None:
    X = pd.DataFrame({"a": []})
    report = BasicValidator().validate(X)
    assert not report.ok


def test_raise_if_failed() -> None:
    from fireflyframework_datascience.core.exceptions import FireflyDataScienceError

    X = pd.DataFrame({"a": [1, 2]})
    report = BasicValidator().validate(X, pd.Series([0, None]))
    try:
        report.raise_if_failed()
        raise AssertionError("expected failure")
    except FireflyDataScienceError:
        pass
