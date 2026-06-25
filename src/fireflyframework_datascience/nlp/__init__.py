# Copyright 2026 Firefly Software Foundation.
"""NLP module — text-classification port and model (import-light).

The HuggingFace adapter (``nlp`` extra) lives in :mod:`fireflyframework_datascience.nlp.adapters`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class TextModel:
    """A fitted text classifier (wraps a tokenizer + model behind ``predict``)."""

    name: str
    predictor: Any
    classes: list[Any] = field(default_factory=list)

    def predict(self, texts: Sequence[str]) -> list[Any]:
        return self.predictor.predict(list(texts))


@runtime_checkable
class TextClassifierPort(Protocol):
    """Fine-tunes a text classifier on ``(texts, labels)`` and returns a :class:`TextModel`."""

    name: str

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> TextModel: ...


__all__ = ["TextClassifierPort", "TextModel"]
