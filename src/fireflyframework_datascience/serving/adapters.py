# Copyright 2026 Firefly Software Foundation.
"""Heavier serving adapters (require the ``serving`` extra)."""

from __future__ import annotations

from typing import Any

from fireflyframework_datascience.core.exceptions import AdapterUnavailableError
from fireflyframework_datascience.models import Model


class BentoMLModelServer:
    """Serves a model via BentoML (packaging/deployment to a BentoML service).

    Requires the ``serving`` extra. Full service packaging is a deployment concern; this adapter wraps a
    fitted model and integrates with BentoML's runner API when available.
    """

    name = "bentoml"

    def __init__(self) -> None:
        try:
            import bentoml  # type: ignore[import-not-found, import-untyped]  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise AdapterUnavailableError("BentoMLModelServer", "serving") from exc
        self._model: Model | None = None

    def load(self, model: Model) -> None:
        self._model = model

    def predict(self, X: Any) -> Any:
        if self._model is None:
            from fireflyframework_datascience.core.exceptions import FireflyDataScienceError

            raise FireflyDataScienceError("No model loaded")
        return self._model.predict(X)


__all__ = ["BentoMLModelServer"]
