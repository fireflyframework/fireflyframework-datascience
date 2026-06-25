# Copyright 2026 Firefly Software Foundation.
"""Validation module — data-validation port and report (import-light)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ValidationReport:
    """The outcome of validating a feature matrix (and optional target)."""

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        if not self.ok:
            from fireflyframework_datascience.core.exceptions import FireflyDataScienceError

            raise FireflyDataScienceError("Data validation failed: " + "; ".join(self.errors))


@runtime_checkable
class ValidatorPort(Protocol):
    """Validates a feature matrix ``X`` and optional target ``y``."""

    name: str

    def validate(self, X: Any, y: Any = None) -> ValidationReport: ...


__all__ = ["ValidationReport", "ValidatorPort"]
