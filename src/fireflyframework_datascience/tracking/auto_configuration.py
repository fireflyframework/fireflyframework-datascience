# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for the tracking module."""

from __future__ import annotations

from fireflyframework_datascience.container.conditions import (
    auto_configuration,
    conditional_on_missing_bean,
    conditional_on_property,
)
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.tracking import TrackerPort


@auto_configuration
@configuration
class TrackingAutoConfiguration:
    """Registers the no-op tracker by default; MLflow when explicitly enabled via config."""

    @bean(name="noop_tracker")
    @conditional_on_missing_bean(TrackerPort)
    def noop_tracker(self) -> TrackerPort:
        from fireflyframework_datascience.tracking.adapters import NoOpTracker

        return NoOpTracker()

    @bean(name="mlflow_tracker", primary=True)
    @conditional_on_property("tracking_enabled", having_value=True)
    def mlflow_tracker(self) -> TrackerPort:
        from fireflyframework_datascience.tracking.adapters import MLflowTracker

        return MLflowTracker()
