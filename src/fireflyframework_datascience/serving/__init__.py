# Copyright 2026 Firefly Software Foundation.
"""Serving module — the model-server port and an in-process server (import-light).

``LocalModelServer`` serves a fitted ``Model`` in-process (no external dependency) and is the default;
heavier servers (BentoML/KServe, vLLM/TGI for LLMs) live in
:mod:`fireflyframework_datascience.serving.adapters` behind the ``serving`` extra.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from fireflyframework_datascience.core.exceptions import FireflyDataScienceError
from fireflyframework_datascience.models import Model


@runtime_checkable
class ModelServerPort(Protocol):
    """Loads a fitted model and answers prediction requests."""

    name: str

    def load(self, model: Model) -> None: ...

    def predict(self, X: Any) -> Any: ...


class LocalModelServer:
    """Serves a fitted ``Model`` in the host process — the default, dependency-free server."""

    name = "local"

    def __init__(self) -> None:
        self._model: Model | None = None

    def load(self, model: Model) -> None:
        self._model = model

    def _require(self) -> Model:
        if self._model is None:
            raise FireflyDataScienceError("No model loaded — call load(model) first")
        return self._model

    def predict(self, X: Any) -> Any:
        return self._require().predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self._require().predict_proba(X)

    @property
    def model(self) -> Model | None:
        return self._model


__all__ = ["LocalModelServer", "ModelServerPort"]
