# Copyright 2026 Firefly Software Foundation.
"""Smoke tests for the advanced-AutoML sample (calibration, ensembling, PR-AUC, explainability, audit).

The offline test forces the deterministic proposer (so it runs in CI with no key); the integration
test runs the *same* sample against a live LLM when ``ANTHROPIC_API_KEY`` is present.
"""

from __future__ import annotations

import os

import pytest


def _load_sample():  # type: ignore[no-untyped-def]
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "samples"))
    import advanced_automl

    return advanced_automl


def test_advanced_automl_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the offline (deterministic) path regardless of the ambient environment.
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    report = _load_sample().run()

    # Offline, deterministic feature engineering with a full audit trail (one record per proposal).
    assert report["proposer"] == "static"
    assert "noise" in report["rejected_features"]  # the no-information feature is always rejected
    assert len(report["audit_trail"]) == 3
    assert all(r["decision"] in ("accepted", "rejected") for r in report["audit_trail"])
    assert {r["feature"] for r in report["audit_trail"]} == {"area_to_perimeter", "concavity_interaction", "noise"}

    # Winner is a stacking ensemble, selected on PR-AUC.
    assert report["winner"] == "stacking_ensemble"
    assert report["selection_metric"] == "average_precision"
    assert report["cv_scoring"] == "average_precision"

    # Richer metrics are reported, including calibration quality (Brier) and PR-AUC, on a strong model.
    holdout = report["holdout"]
    assert {"roc_auc", "average_precision", "brier_score"} <= set(holdout)
    assert holdout["roc_auc"] > 0.95
    assert 0.0 <= holdout["brier_score"] <= 0.25  # well-calibrated probabilities

    # Explainability produced ranked global importances.
    assert report["explanation_method"] in ("permutation_importance", "shap")
    assert len(report["top_features"]) > 0


@pytest.mark.integration
def test_advanced_automl_with_real_llm() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("needs ANTHROPIC_API_KEY")

    report = _load_sample().run()

    # The real LLM proposed features and every gate decision was persisted to the audit trail.
    assert report["proposer"] == "llm"
    assert len(report["audit_trail"]) >= 1
    assert all(r["decision"] in ("accepted", "rejected") for r in report["audit_trail"])

    # The robust selection pipeline still produced a calibrated stacking ensemble selected on PR-AUC.
    assert report["winner"] == "stacking_ensemble"
    assert report["selection_metric"] == "average_precision"
    assert report["holdout"]["roc_auc"] > 0.9
    assert len(report["top_features"]) > 0
