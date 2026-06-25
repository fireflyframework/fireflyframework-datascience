# Copyright 2026 Firefly Software Foundation.
"""Tracking adapters: a no-op tracker (default) and an MLflow tracker (opt-in)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fireflyframework_datascience.tracking import RunHandle

logger = logging.getLogger(__name__)


class NoOpTracker:
    """Default tracker — records nothing. Keeps the core dependency-free."""

    name = "noop"

    def start_run(self, run_name: str | None = None) -> RunHandle:
        return RunHandle(run_id="noop", name=run_name or "run")

    def log_params(self, params: Mapping[str, Any]) -> None:
        logger.debug("params: %s", dict(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        logger.debug("metrics: %s", dict(metrics))

    def log_model(self, model: Any, artifact_name: str = "model") -> None:
        logger.debug("model logged: %s", artifact_name)

    def end_run(self) -> None:
        logger.debug("run ended")


class MLflowTracker:
    """Tracks runs to MLflow. Requires the ``tracking`` extra."""

    name = "mlflow"

    def __init__(self, tracking_uri: str | None = None, experiment: str = "firefly-datascience") -> None:
        try:
            import mlflow  # type: ignore[import-not-found, import-untyped]
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            from fireflyframework_datascience.core.exceptions import AdapterUnavailableError

            raise AdapterUnavailableError("MLflowTracker", "tracking") from exc
        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    def start_run(self, run_name: str | None = None) -> RunHandle:
        run = self._mlflow.start_run(run_name=run_name)
        return RunHandle(run_id=run.info.run_id, name=run_name or run.info.run_id)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_model(self, model: Any, artifact_name: str = "model") -> None:
        self._mlflow.sklearn.log_model(model, name=artifact_name)

    def end_run(self) -> None:
        self._mlflow.end_run()


__all__ = ["MLflowTracker", "NoOpTracker"]
