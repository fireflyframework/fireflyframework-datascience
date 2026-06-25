# Copyright 2026 Firefly Software Foundation.
"""Smoke test for the offline AutoML benchmark harness."""

from __future__ import annotations

import math


def _harness():  # type: ignore[no-untyped-def]
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks"))
    import automl_benchmark

    return automl_benchmark


def test_run_suite_offline() -> None:
    harness = _harness()
    results = harness.run_suite(["breast_cancer", "diabetes"], cv=3)
    assert len(results) == 2
    by_name = {r.dataset: r for r in results}
    assert by_name["breast_cancer"].task == "binary"
    assert by_name["diabetes"].task == "regression"
    for r in results:
        assert math.isfinite(r.cv_score)
        assert math.isfinite(r.holdout_score)
        assert r.winner
        assert r.fit_seconds >= 0


def test_format_table() -> None:
    harness = _harness()
    results = harness.run_suite(["iris"], cv=3)
    table = harness.format_table(results)
    assert "dataset" in table
    assert "iris" in table
