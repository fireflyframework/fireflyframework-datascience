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
        import numpy as np

        from fireflyframework_datascience.preprocessing import build_preprocessor

        try:
            from fireflyframework_datascience.dl._torch_impl import TorchEstimator, train_tabular
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise AdapterUnavailableError("TorchTabularTrainer", "dl") from exc

        preprocessor = build_preprocessor(dataset.X)
        matrix = preprocessor.fit_transform(dataset.X) if preprocessor is not None else dataset.X.to_numpy()

        is_classification = dataset.task in _CLASSIFICATION
        if is_classification:
            classes, y_encoded = np.unique(dataset.y, return_inverse=True)
            out_dim = int(len(classes))
        else:
            classes, y_encoded, out_dim = None, np.asarray(dataset.y, dtype="float32"), 1

        model = train_tabular(
            matrix,
            y_encoded,
            is_classification=is_classification,
            out_dim=out_dim,
            epochs=self._epochs,
            hidden=self._hidden,
            lr=self._lr,
        )
        estimator = TorchEstimator(preprocessor, model, classes, is_classification)
        return Model("torch_tabular", estimator, dataset.task, list(dataset.feature_names))


__all__ = ["MLPTrainer", "TabPFNPredictor", "TorchTabularTrainer"]
