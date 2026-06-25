# Copyright 2026 Firefly Software Foundation.
"""Integration test: Firefly AutoML matches-or-beats the baseline on a CV metric (needs network)."""

from __future__ import annotations

import pytest


def _beat():  # type: ignore[no-untyped-def]
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks"))
    import beat_baseline

    return beat_baseline


@pytest.mark.integration
def test_firefly_beats_baseline_on_phoneme() -> None:
    mod = _beat()
    ds = mod._load(1489, None)  # phoneme — clearly non-linear
    base = mod._baseline_cv_auc(ds)
    fire, winner = mod._firefly_cv_auc(ds)
    assert fire >= base  # AutoML never does worse (it includes the baseline's family)
    assert fire - base > 0.05  # and beats it decisively here
    assert winner  # a model was selected
