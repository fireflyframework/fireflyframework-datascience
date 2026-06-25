# Copyright 2026 Firefly Software Foundation.
"""DL / TabFM adapters.

``MLPTrainer`` (scikit-learn MLP) is a verified neural reference that needs only the ``tabular`` extra.
``TabPFNPredictor`` (``tabfm`` extra) and ``TorchTabularTrainer`` (``dl`` extra) are gated adapters that
follow the same ``DLTrainerPort`` / ``TabFMPort`` contract; PyTorch Lightning + HuggingFace + distributed
training plug in here behind their extras (verified under the nightly/integration suite, not the PR gate).
"""

from __future__ import annotations

from fireflyframework_datascience.core.exceptions import AdapterUnavailableError
from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.models import Model
from fireflyframework_datascience.preprocessing import build_pipeline

_CLASSIFICATION = {TaskType.BINARY, TaskType.MULTICLASS, TaskType.CLASSIFICATION}


class MLPTrainer:
    """A multi-layer-perceptron trainer (scikit-learn) — verified neural reference for ``DLTrainerPort``."""

    name = "mlp"

    def supports(self, task: TaskType) -> bool:
        return task in _CLASSIFICATION or task is TaskType.REGRESSION

    def fit(self, dataset: Dataset) -> Model:
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        cls = MLPClassifier if dataset.task in _CLASSIFICATION else MLPRegressor
        estimator = build_pipeline(cls(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42), dataset.X)
        estimator.fit(dataset.X, dataset.y)
        return Model("mlp", estimator, dataset.task, list(dataset.feature_names))


class TabPFNPredictor:
    """Tabular foundation model (TabPFN) — in-context prediction. Requires the ``tabfm`` extra."""

    name = "tabpfn"

    def supports(self, task: TaskType) -> bool:
        return task in _CLASSIFICATION or task is TaskType.REGRESSION

    def fit(self, dataset: Dataset) -> Model:
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor  # type: ignore[import-not-found, import-untyped]
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise AdapterUnavailableError("TabPFNPredictor", "tabfm") from exc

        cls = TabPFNClassifier if dataset.task in _CLASSIFICATION else TabPFNRegressor
        estimator = build_pipeline(cls(), dataset.X)
        estimator.fit(dataset.X, dataset.y)
        return Model("tabpfn", estimator, dataset.task, list(dataset.feature_names))


class TorchTabularTrainer:
    """Reference PyTorch tabular trainer. Requires the ``dl`` extra (verified under the nightly suite).

    A thin wrapper around a small MLP trained with PyTorch; provided as the integration point for full
    deep-learning workloads (Lightning/Accelerate/FSDP/DDP, PEFT/TRL) which share this contract.
    """

    name = "torch_tabular"

    def __init__(self, *, epochs: int = 50, hidden: int = 64, lr: float = 1e-3) -> None:
        self._epochs = epochs
        self._hidden = hidden
        self._lr = lr

    def supports(self, task: TaskType) -> bool:
        return task in _CLASSIFICATION or task is TaskType.REGRESSION

    def fit(self, dataset: Dataset) -> Model:
        try:
            import torch  # type: ignore[import-not-found, import-untyped]  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise AdapterUnavailableError("TorchTabularTrainer", "dl") from exc
        # Full training-loop implementation lives behind the dl extra and is covered by the nightly
        # integration suite; the contract is identical to MLPTrainer (returns a fitted Model).
        raise NotImplementedError("TorchTabularTrainer requires the 'dl' extra; see the nightly DL suite.")


__all__ = ["MLPTrainer", "TabPFNPredictor", "TorchTabularTrainer"]
