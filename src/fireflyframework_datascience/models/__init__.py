# Copyright 2026 Firefly Software Foundation.
"""Models module — the fitted ``Model`` wrapper and the ``TrainerPort``.

Import-light: heavy estimators live in :mod:`fireflyframework_datascience.models.adapters`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.tuning import ParamSpace


@dataclass
class Model:
    """A fitted estimator (or pipeline) with its metadata."""

    name: str
    estimator: Any
    task: TaskType
    feature_names: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: Any) -> Any:
        return self.estimator.predict(X)

    def predict_proba(self, X: Any) -> Any:
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"Estimator {self.name!r} does not support predict_proba")
        return self.estimator.predict_proba(X)

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> Model:
        # SECURITY: joblib uses pickle, which executes arbitrary code on load. Only load models from
        # trusted, first-party locations (your own registry / artifact store) — never untrusted input.
        # The registry adapter (SP1+) enforces an allowlist; this low-level loader assumes a trusted path.
        import joblib

        return joblib.load(Path(path))


@runtime_checkable
class TrainerPort(Protocol):
    """Builds an unfitted estimator for a task and declares its search space."""

    name: str

    def supports(self, task: TaskType) -> bool: ...

    def make_estimator(self, task: TaskType, params: Mapping[str, Any] | None = None) -> Any: ...

    def param_space(self, task: TaskType) -> ParamSpace: ...


__all__ = ["Model", "TrainerPort"]
