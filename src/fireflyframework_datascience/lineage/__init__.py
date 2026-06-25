# Copyright 2026 Firefly Software Foundation.
"""Lineage module — data/model lineage port (import-light) with a no-op default."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class LineageEvent:
    """A lineage event: a named run with input/output dataset references and metadata."""

    name: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LineagePort(Protocol):
    """Emits lineage events to a backend (OpenLineage/Marquez, …)."""

    name: str

    def emit(self, event: LineageEvent) -> None: ...


class NoOpLineage:
    """Default lineage backend — logs at debug level, emits nowhere."""

    name = "noop"

    def emit(self, event: LineageEvent) -> None:
        logger.debug("lineage event: %s inputs=%s outputs=%s", event.name, event.inputs, event.outputs)


__all__ = ["LineageEvent", "LineagePort", "NoOpLineage"]
