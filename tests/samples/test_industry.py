# Copyright 2026 Firefly Software Foundation.
"""Integration test for the industry showcase (real OpenML data — needs network)."""

from __future__ import annotations

import pytest


def _showcase():  # type: ignore[no-untyped-def]
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "samples"))
    import industry_showcase

    return industry_showcase


@pytest.mark.integration
def test_finance_case_runs() -> None:
    report = _showcase().run_case(*_showcase().FINANCE)
    assert report["validation_ok"] is True
    assert report["winner"]
    assert report["holdout"] > 0.6
    assert report["rows"] == 1000
