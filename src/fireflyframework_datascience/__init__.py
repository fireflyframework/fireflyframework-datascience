# Copyright 2026 Firefly Software Foundation.
"""Firefly DataScience — AutoML that fuses GenAI with classical ML & Deep Learning.

A state-of-the-art Python metaframework for AutoML, built on the Firefly Framework. It combines
GenAI (via ``fireflyframework-agentic`` / Pydantic AI) with traditional ML and Deep Learning behind
hexagonal, swappable ports, with auto-configuration and dependency injection that feel native to the
Firefly ecosystem.
"""

from __future__ import annotations

from fireflyframework_datascience._version import __version__
from fireflyframework_datascience.application import (
    ApplicationContext,
    FireflyDataScienceApplication,
)
from fireflyframework_datascience.core.config import FireflyDataScienceConfig
from fireflyframework_datascience.core.exceptions import (
    AdapterUnavailableError,
    AutoConfigurationError,
    ConfigurationError,
    FireflyDataScienceError,
    WiringError,
)
from fireflyframework_datascience.core.types import Modality, TaskType

__all__ = [
    "AdapterUnavailableError",
    "ApplicationContext",
    "AutoConfigurationError",
    "ConfigurationError",
    "FireflyDataScienceApplication",
    "FireflyDataScienceConfig",
    "FireflyDataScienceError",
    "Modality",
    "TaskType",
    "WiringError",
    "__version__",
]
