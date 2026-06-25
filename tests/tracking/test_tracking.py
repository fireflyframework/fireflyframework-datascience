# Copyright 2026 Firefly Software Foundation.
"""Tests for the no-op tracker (default)."""

from __future__ import annotations

from fireflyframework_datascience.tracking.adapters import NoOpTracker


def test_noop_tracker_lifecycle() -> None:
    tracker = NoOpTracker()
    handle = tracker.start_run("my-run")
    assert handle.name == "my-run"
    # all of these are no-ops and must not raise
    tracker.log_params({"a": 1})
    tracker.log_metrics({"score": 0.5})
    tracker.log_model(object(), "model")
    tracker.end_run()
