# Copyright 2026 Firefly Software Foundation.
"""Integration test for the OpenML AMLB-style harness (needs network)."""

from __future__ import annotations

import math

import pytest


def _harness():  # type: ignore[no-untyped-def]
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks"))
    import amlb_benchmark

    return amlb_benchmark


@pytest.mark.integration
def test_amlb_subset_runs() -> None:
    harness = _harness()
    results = harness.run_amlb([1464], cv=3)  # blood-transfusion: small binary CC18 task
    assert len(results) == 1
    result = results[0]
    assert result.task == "binary"
    assert math.isfinite(result.holdout_score)
    assert result.winner
