# Copyright 2026 Firefly Software Foundation.
"""Evaluation module — metrics port and result type (import-light)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fireflyframework_datascience.core.types import TaskType


@dataclass
class EvaluationResult:
    """A panel of metric values plus the designated primary metric."""

    metrics: dict[str, float] = field(default_factory=dict)
    primary_metric: str = ""
    primary_value: float = 0.0

    def __str__(self) -> str:
        body = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"EvaluationResult(primary={self.primary_metric}={self.primary_value:.4f}; {body})"


@runtime_checkable
class MetricsEvaluatorPort(Protocol):
    """Computes metrics and maps tasks/metrics to cross-validation scoring strings."""

    def default_metric(self, task: TaskType) -> str: ...

    def scoring_name(self, task: TaskType, metric: str) -> str: ...

    def greater_is_better(self, metric: str) -> bool: ...

    def evaluate(
        self, task: TaskType, y_true: Any, y_pred: Any, y_proba: Any = None, *, metric: str | None = None
    ) -> EvaluationResult: ...


__all__ = ["EvaluationResult", "MetricsEvaluatorPort"]
