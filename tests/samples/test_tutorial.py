# Copyright 2026 Firefly Software Foundation.
"""Guarantees the end-to-end tutorial actually runs (the user asked us to ensure it works)."""

from __future__ import annotations


def _tutorial():  # type: ignore[no-untyped-def]
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "samples"))
    import tutorial

    return tutorial


def test_tutorial_runs_end_to_end() -> None:
    report = _tutorial().run()
    assert report["boot"]["beans"] > 0
    assert report["validation_ok"] is True
    assert report["automl_winner"]
    assert report["automl_roc_auc"] > 0.7
    assert "debt_to_income" in report["fe_accepted"]  # the hidden driver is discovered
    assert "noise" in report["fe_rejected"]  # the useless feature is gated out
    assert report["fe_lift"] > 0
    assert report["agentic_best"]
    assert report["agentic_verified"] >= 1
    assert report["sample_prediction"] in (0, 1)
