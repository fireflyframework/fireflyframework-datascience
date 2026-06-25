# Copyright 2026 Firefly Software Foundation.
"""Real-LLM integration test — runs the GenAI showcase against a live model.

Marked `integration` (excluded from the default gate) and skipped unless an LLM key is present, so it
costs nothing in normal CI but verifies the real path when credentials are available.
"""

from __future__ import annotations

import os

import pytest


def _showcase():  # type: ignore[no-untyped-def]
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "samples"))
    import genai_llm_showcase

    return genai_llm_showcase


@pytest.mark.integration
def test_real_llm_feature_engineering() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("needs ANTHROPIC_API_KEY")
    result = _showcase().genai_feature_engineering()
    # The LLM proposed features; some were evaluated and a verdict was reached for each.
    assert result["accepted"] or result["rejected"]
    for _name, code in result["rejected"]:
        assert "df[" in code  # the model returned executable feature code
    assert "roc_auc" in result["summary"]


@pytest.mark.integration
def test_real_llm_agentic_loop() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("needs ANTHROPIC_API_KEY")
    result = _showcase().agentic_loop()
    assert len(result["attempts"]) >= 3  # seed population + at least one LLM-reflected attempt
    assert result["best"][0]  # a winning trainer was selected
    assert result["holdout_predictions"] > 0  # the fitted model predicts
