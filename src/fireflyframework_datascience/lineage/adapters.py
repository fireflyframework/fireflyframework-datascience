# Copyright 2026 Firefly Software Foundation.
"""OpenLineage emitter (requires the ``lineage`` extra)."""

from __future__ import annotations

from fireflyframework_datascience.core.exceptions import AdapterUnavailableError
from fireflyframework_datascience.lineage import LineageEvent


class OpenLineageEmitter:
    """Emits events to an OpenLineage backend (e.g. Marquez). Requires the ``lineage`` extra."""

    name = "openlineage"

    def __init__(self, url: str = "http://localhost:5000", namespace: str = "firefly-datascience") -> None:
        try:
            from openlineage.client import OpenLineageClient  # type: ignore[import-not-found, import-untyped]
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise AdapterUnavailableError("OpenLineageEmitter", "lineage") from exc
        self._client = OpenLineageClient(url=url)
        self._namespace = namespace

    def emit(self, event: LineageEvent) -> None:
        # Translate to the OpenLineage RunEvent shape; kept minimal for the reference adapter.
        self._client.emit(event)  # type: ignore[arg-type]


__all__ = ["OpenLineageEmitter"]
