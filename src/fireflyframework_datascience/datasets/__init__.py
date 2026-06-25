# Copyright 2026 Firefly Software Foundation.
"""Datasets module — the ``Dataset`` container and the ``DatasetLoaderPort``.

This module is import-light (pandas / scikit-learn are imported lazily inside methods), so the port and
the ``Dataset`` type are usable without the ``tabular`` extra installed. Concrete loaders live in
:mod:`fireflyframework_datascience.datasets.adapters`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fireflyframework_datascience.core.types import TaskType

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class Dataset:
    """A named tabular dataset: features ``X``, optional target ``y``, and metadata."""

    name: str
    X: Any
    y: Any = None
    task: TaskType = TaskType.CLASSIFICATION
    target_name: str | None = None
    feature_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(len(self.X))

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def has_target(self) -> bool:
        return self.y is not None

    def train_test_split(
        self, *, test_size: float = 0.25, random_state: int = 42
    ) -> tuple[Dataset, Dataset]:
        """Split into (train, test) datasets, stratifying on the target for classification."""
        from sklearn.model_selection import train_test_split as _split

        stratify = self.y if self.task.is_classification() and self.y is not None else None
        x_train, x_test, y_train, y_test = _split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        return (
            self._clone(f"{self.name}[train]", x_train, y_train),
            self._clone(f"{self.name}[test]", x_test, y_test),
        )

    def with_features(self, X: pd.DataFrame) -> Dataset:
        """Return a copy with replaced feature matrix ``X`` (used by feature engineering)."""
        return self._clone(self.name, X, self.y, feature_names=list(X.columns))

    def _clone(self, name: str, X: Any, y: Any, *, feature_names: Sequence[str] | None = None) -> Dataset:
        return Dataset(
            name=name,
            X=X,
            y=y,
            task=self.task,
            target_name=self.target_name,
            feature_names=list(feature_names) if feature_names is not None else list(self.feature_names),
            metadata=dict(self.metadata),
        )


@runtime_checkable
class DatasetLoaderPort(Protocol):
    """Loads a :class:`Dataset` from a string ``source`` (a name, an id, or a URI)."""

    name: str

    def can_load(self, source: str) -> bool: ...

    def load(self, source: str, *, target: str | None = None, **kwargs: Any) -> Dataset: ...


def infer_task(y: Any) -> TaskType:
    """Infer a :class:`TaskType` from a target series/array."""
    import pandas as pd

    series = pd.Series(y)
    n_unique = series.nunique(dropna=True)
    if series.dtype.kind in {"f"} and n_unique > 20:
        return TaskType.REGRESSION
    if series.dtype.kind in {"i", "u"} and n_unique > 20:
        return TaskType.REGRESSION
    if n_unique == 2:
        return TaskType.BINARY
    return TaskType.MULTICLASS


__all__ = ["Dataset", "DatasetLoaderPort", "infer_task"]
