# Copyright 2026 Firefly Software Foundation.
"""Core domain types shared across the framework.

Kept deliberately light (no third-party ML imports) so the core stays importable without any optional
extra installed.
"""

from __future__ import annotations

from enum import StrEnum


class TaskType(StrEnum):
    """The kind of supervised/unsupervised learning task."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    FORECASTING = "forecasting"

    def is_classification(self) -> bool:
        """True for any classification flavour (binary/multiclass/generic)."""
        return self in {TaskType.BINARY, TaskType.MULTICLASS, TaskType.CLASSIFICATION}


class Modality(StrEnum):
    """The data modality a pipeline operates on."""

    TABULAR = "tabular"
    TEXT = "text"
    VISION = "vision"
    TIMESERIES = "timeseries"
    MULTIMODAL = "multimodal"


class Scope(StrEnum):
    """Bean lifecycle scopes understood by the DI container."""

    SINGLETON = "singleton"
    TRANSIENT = "transient"
