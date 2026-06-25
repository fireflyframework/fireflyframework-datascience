# Copyright 2026 Firefly Software Foundation.
"""A small PyTorch CNN for image classification (isolated so torch loads only when used)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch  # type: ignore[import-not-found, import-untyped]
from torch import nn  # type: ignore[import-not-found, import-untyped]
from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import-not-found, import-untyped]


class SmallCNN(nn.Module):
    """Conv → Conv → adaptive-pool → MLP head. Input-size agnostic via ``AdaptiveAvgPool2d``."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 4 * 4, 32), nn.ReLU(), nn.Linear(32, num_classes))

    def forward(self, x: Any) -> Any:
        return self.head(self.features(x))


def train_cnn(
    images: Any, y: Any, *, num_classes: int, epochs: int, lr: float, batch_size: int, seed: int = 42
) -> SmallCNN:
    """Train :class:`SmallCNN` on an ``(N, C, H, W)`` float array; return it in eval mode (CPU)."""
    torch.manual_seed(seed)
    x_tensor = torch.tensor(np.asarray(images, dtype="float32"))
    y_tensor = torch.tensor(np.asarray(y), dtype=torch.long)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)
    net = SmallCNN(x_tensor.shape[1], num_classes)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(net(xb), yb).backward()
            optimizer.step()
    net.eval()
    return net


class CNNPredictor:
    """Wraps a trained CNN; maps argmax logits back to the original labels."""

    def __init__(self, net: SmallCNN, classes: list[Any]) -> None:
        self._net = net
        self._classes = classes

    def predict(self, images: Any) -> list[Any]:
        x_tensor = torch.tensor(np.asarray(images, dtype="float32"))
        with torch.no_grad():
            logits = self._net(x_tensor)
        return [self._classes[i] for i in logits.argmax(dim=1).tolist()]


__all__ = ["CNNPredictor", "SmallCNN", "train_cnn"]
