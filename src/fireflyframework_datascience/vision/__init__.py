# Copyright 2026 Firefly Software Foundation.
"""Vision module — image-classification port and model (import-light).

The PyTorch CNN adapter (``dl`` extra) lives in :mod:`fireflyframework_datascience.vision.adapters`.
Images are ``(N, C, H, W)`` float arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ImageModel:
    """A fitted image classifier (wraps a trained CNN behind ``predict``)."""

    name: str
    predictor: Any
    classes: list[Any] = field(default_factory=list)

    def predict(self, images: Any) -> list[Any]:
        return self.predictor.predict(images)


@runtime_checkable
class ImageClassifierPort(Protocol):
    """Trains an image classifier on ``(images, labels)`` and returns an :class:`ImageModel`."""

    name: str

    def fit(self, images: Any, labels: Sequence[Any]) -> ImageModel: ...


__all__ = ["ImageClassifierPort", "ImageModel"]
