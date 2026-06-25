# Copyright 2026 Firefly Software Foundation.
"""Core layer: configuration, types, exceptions, banner, ordering, plugin discovery."""

from __future__ import annotations

from fireflyframework_datascience.core.exceptions import (
    AdapterUnavailableError,
    AutoConfigurationError,
    ConfigurationError,
    FireflyDataScienceError,
    WiringError,
)
from fireflyframework_datascience.core.ordering import (
    DEFAULT_ORDER,
    HIGHEST_PRECEDENCE,
    LOWEST_PRECEDENCE,
    get_order,
    order,
)
from fireflyframework_datascience.core.types import Modality, TaskType

__all__ = [
    "DEFAULT_ORDER",
    "HIGHEST_PRECEDENCE",
    "LOWEST_PRECEDENCE",
    "AdapterUnavailableError",
    "AutoConfigurationError",
    "ConfigurationError",
    "FireflyDataScienceError",
    "Modality",
    "TaskType",
    "WiringError",
    "get_order",
    "order",
]
