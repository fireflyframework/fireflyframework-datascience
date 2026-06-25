# Copyright 2026 Firefly Software Foundation.
"""Tracking module — experiment tracking & model registry ports (import-light)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class RunHandle:
    """An opaque handle to an in-progress tracking run."""

    run_id: str
    name: str


@runtime_checkable
class TrackerPort(Protocol):
    """Records params, metrics, and artifacts for an experiment run."""

    name: str

    def start_run(self, run_name: str | None = None) -> RunHandle: ...

    def log_params(self, params: Mapping[str, Any]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None: ...

    def log_model(self, model: Any, artifact_name: str = "model") -> None: ...

    def end_run(self) -> None: ...


@runtime_checkable
class RegistryPort(Protocol):
    """Persists and retrieves models by name/version."""

    name: str

    def register(self, model: Any, name: str) -> str: ...

    def load(self, name: str, version: str | None = None) -> Any: ...


__all__ = ["RegistryPort", "RunHandle", "TrackerPort"]
