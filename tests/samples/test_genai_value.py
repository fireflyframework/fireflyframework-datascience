# Copyright 2026 Firefly Software Foundation.
"""Integration test for the GenAI-value ablation (real LLM)."""

from __future__ import annotations

import os

import pytest


def _mod():  # type: ignore[no-untyped-def]
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks"))
    import genai_value

    return genai_value


@pytest.mark.integration
def test_genai_value_runs() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("needs ANTHROPIC_API_KEY")
    mod = _mod()
    mod.SEEDS = [0]  # one split → one LLM call, for speed
    res = mod.run()
    assert {"linear (raw)", "linear + GenAI", "Firefly (raw)", "Firefly + GenAI"} <= set(res["scores"])
    assert len(res["scores"]["linear + GenAI"]) == 1
    assert res["accepted_features"]  # the LLM proposed code the gate accepted
