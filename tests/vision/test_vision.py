# Copyright 2026 Firefly Software Foundation.
"""Tests for the PyTorch CNN image classifier (verified on synthetic images, no download)."""

from __future__ import annotations

import numpy as np
import pytest


def _make_images(n: int = 120, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.RandomState(seed)
    images, labels = [], []
    for i in range(n):
        img = (rng.rand(1, 8, 8) * 0.1).astype("float32")
        if i % 2 == 0:
            img[:, :4, :] += 1.0  # bright top half
            labels.append("top")
        else:
            img[:, 4:, :] += 1.0  # bright bottom half
            labels.append("bottom")
        images.append(img)
    return np.stack(images), labels


def test_torch_cnn_classifies_synthetic_images() -> None:
    pytest.importorskip("torch")
    from fireflyframework_datascience.vision.adapters import TorchCNNClassifier

    X, y = _make_images(120)
    model = TorchCNNClassifier(epochs=12).fit(X[:90], y[:90])
    preds = model.predict(X[90:])
    accuracy = float(np.mean([p == t for p, t in zip(preds, y[90:], strict=True)]))
    assert accuracy > 0.7  # top-vs-bottom brightness is trivially learnable
    assert set(model.classes) == {"top", "bottom"}
