# Copyright 2026 Firefly Software Foundation.
"""A persistent, auditable trail of GenAI gate decisions.

The framework's thesis is that *every* GenAI decision is logged and auditable. The in-memory
:class:`EngineeringResult` carries the trail for one run; an :class:`AuditLogPort` persists each
decision durably so a risk / compliance / audit function can reconstruct *why* a feature was kept or
dropped, long after the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AuditLogPort(Protocol):
    """Durably records one GenAI gate decision per call."""

    def record(self, event: dict[str, Any]) -> None: ...


class JsonlAuditLog:
    """Appends each decision as a JSON line — simple, greppable, append-only."""

    name = "jsonl"

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def record(self, event: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")


__all__ = ["AuditLogPort", "JsonlAuditLog"]
