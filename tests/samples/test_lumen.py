# Copyright 2026 Firefly Software Foundation.
"""Smoke test for the Lumen credit-risk sample (the end-to-end showcase)."""

from __future__ import annotations


def _load_sample():  # type: ignore[no-untyped-def]
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "samples"))
    import lumen_credit_risk

    return lumen_credit_risk


def test_lumen_end_to_end() -> None:
    sample = _load_sample()
    report = sample.run()

    # GenAI feature engineering discovered the domain driver and rejected the noise feature.
    assert "debt_to_income" in report["accepted_features"]
    assert "noise" in report["rejected_features"]
    assert report["fe_lift"] > 0

    # Classical AutoML produced a credible model and served a prediction.
    assert report["winner"]
    assert report["holdout"]["roc_auc"] > 0.65
    assert report["sample_prediction"] in (0, 1)


def test_lending_dataset_shape() -> None:
    sample = _load_sample()
    ds = sample.make_lending_dataset(n=200)
    assert ds.n_rows == 200
    assert "income" in ds.X.columns
    assert ds.task.is_classification()
