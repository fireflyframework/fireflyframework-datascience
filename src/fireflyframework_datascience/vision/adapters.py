# Copyright 2026 Firefly Software Foundation.
"""PyTorch image-classification adapter (requires the ``dl`` extra)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fireflyframework_datascience.core.exceptions import AdapterUnavailableError
from fireflyframework_datascience.vision import ImageModel


class TorchCNNClassifier:
    """Trains a small CNN on ``(N, C, H, W)`` image arrays — the vision modality, fit/predict contract."""

    name = "torch_cnn"

    def __init__(self, *, epochs: int = 15, lr: float = 1e-3, batch_size: int = 16) -> None:
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size

    def fit(self, images: Any, labels: Sequence[Any]) -> ImageModel:
        try:
            from fireflyframework_datascience.vision._cnn_impl import CNNPredictor, train_cnn
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise AdapterUnavailableError("TorchCNNClassifier", "dl") from exc

        classes = sorted(set(labels), key=str)
        label_to_id = {label: i for i, label in enumerate(classes)}
        y = [label_to_id[label] for label in labels]
        net = train_cnn(
            images, y, num_classes=len(classes), epochs=self._epochs, lr=self._lr, batch_size=self._batch_size
        )
        return ImageModel("torch_cnn", CNNPredictor(net, classes), classes)


__all__ = ["TorchCNNClassifier"]
