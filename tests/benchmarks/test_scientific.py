# Copyright 2026 Firefly Software Foundation.
"""Integration test for the nested-CV scientific evaluation harness (network)."""

from __future__ import annotations

import pytest


def _mod():  # type: ignore[no-untyped-def]
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks"))
    import scientific_eval

    return scientific_eval


@pytest.mark.integration
def test_nested_cv_evaluation() -> None:
    mod = _mod()
    res = mod.evaluate(mod._load(1464, None))  # blood-transfusion (small, binary)
    assert {"LogReg", "RandomForest", "XGBoost", "Firefly AutoML"} <= set(res["means"])
    assert 0.0 < res["means"]["Firefly AutoML"] <= 1.0
    assert sum(res["picks"].values()) == mod.OUTER_FOLDS  # Firefly picked a model on every fold
