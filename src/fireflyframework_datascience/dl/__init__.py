# Copyright 2026 Firefly Software Foundation.
"""Deep learning & tabular-foundation-model module — ports (import-light).

``DLTrainerPort`` covers neural trainers (a verified sklearn-MLP reference ships here; PyTorch
Lightning / HuggingFace adapters plug in behind the ``dl`` / ``nlp`` extras). ``TabFMPort`` covers
tabular foundation models (TabPFN) behind the ``tabfm`` extra. Adapters:
:mod:`fireflyframework_datascience.dl.adapters`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fireflyframework_datascience.core.types import TaskType

if TYPE_CHECKING:
    from fireflyframework_datascience.datasets import Dataset
    from fireflyframework_datascience.models import Model


@runtime_checkable
class DLTrainerPort(Protocol):
    """Trains a neural model on a dataset and returns a fitted ``Model``."""

    name: str

    def supports(self, task: TaskType) -> bool: ...

    def fit(self, dataset: Dataset) -> Model: ...


@runtime_checkable
class TabFMPort(Protocol):
    """A tabular foundation model (e.g. TabPFN) — in-context fit/predict."""

    name: str

    def supports(self, task: TaskType) -> bool: ...

    def fit(self, dataset: Dataset) -> Model: ...


__all__ = ["DLTrainerPort", "TabFMPort"]
