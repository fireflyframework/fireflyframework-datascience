# Copyright 2026 Firefly Software Foundation.
"""PyTorch Lightning tabular MLP — the concrete deep-learning training implementation.

Isolated so the heavy ``torch`` / ``lightning`` imports load only when the ``dl`` extra is present and
``TorchTabularTrainer.fit`` is actually called.
"""

from __future__ import annotations

from typing import Any

import lightning as L  # type: ignore[import-not-found, import-untyped]
import numpy as np
import torch  # type: ignore[import-not-found, import-untyped]
import torch.nn.functional as F  # type: ignore[import-not-found, import-untyped]
from torch import nn  # type: ignore[import-not-found, import-untyped]
from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import-not-found, import-untyped]


class TabularMLP(L.LightningModule):
    """A small configurable MLP for tabular classification or regression."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, lr: float, is_classification: bool) -> None:
        super().__init__()
        self.is_classification = is_classification
        self.lr = lr
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max(hidden // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden // 2, 8), out_dim),
        )

    def forward(self, x: Any) -> Any:
        return self.net(x)

    def training_step(self, batch: Any, _: int) -> Any:
        x, y = batch
        out = self(x)
        if self.is_classification:
            return F.cross_entropy(out, y.long())
        return F.mse_loss(out.squeeze(-1), y.float())

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train_tabular(
    Xt: Any, y: Any, *, is_classification: bool, out_dim: int, epochs: int, hidden: int, lr: float, seed: int = 42
) -> TabularMLP:
    """Train a :class:`TabularMLP` on a dense numeric matrix and return it in eval mode (CPU)."""
    L.seed_everything(seed, workers=True)
    x_tensor = torch.tensor(np.asarray(Xt, dtype="float32"))
    y_tensor = torch.tensor(np.asarray(y))
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=True)
    model = TabularMLP(x_tensor.shape[1], out_dim, hidden, lr, is_classification)
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader)
    model.eval()
    return model


class TorchEstimator:
    """sklearn-like wrapper: applies the fitted preprocessor, then the trained net. ``Model`` wraps this."""

    def __init__(self, preprocessor: Any, model: TabularMLP, classes: Any, is_classification: bool) -> None:
        self._pre = preprocessor
        self._model = model
        self._classes = classes
        self._is_classification = is_classification

    def _matrix(self, X: Any) -> Any:
        transformed = self._pre.transform(X) if self._pre is not None else np.asarray(X)
        return torch.tensor(np.asarray(transformed, dtype="float32"))

    def predict(self, X: Any) -> Any:
        with torch.no_grad():
            out = self._model(self._matrix(X))
        if self._is_classification:
            indices = out.argmax(dim=1).cpu().numpy()
            return self._classes[indices]
        return out.squeeze(-1).cpu().numpy()

    def predict_proba(self, X: Any) -> Any:
        with torch.no_grad():
            out = self._model(self._matrix(X))
        return F.softmax(out, dim=1).cpu().numpy()


__all__ = ["TabularMLP", "TorchEstimator", "train_tabular"]
