# Copyright 2026 Firefly Software Foundation.
"""The core auto-configuration — always applies, provides framework runtime metadata.

Serves as the reference example of the auto-configuration pattern every adapter module follows.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

from fireflyframework_datascience._version import __version__
from fireflyframework_datascience.container.conditions import auto_configuration
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.core.config import FireflyDataScienceConfig


@dataclass(frozen=True)
class RuntimeInfo:
    """Snapshot of the framework runtime, registered as a bean at startup."""

    framework_version: str
    python_version: str
    platform: str
    default_ml_framework: str
    genai_enabled: bool


@auto_configuration
@configuration
class CoreAutoConfiguration:
    """Registers always-on core beans."""

    @bean
    def runtime_info(self, config: FireflyDataScienceConfig) -> RuntimeInfo:
        return RuntimeInfo(
            framework_version=__version__,
            python_version=platform.python_version(),
            platform=platform.platform(),
            default_ml_framework=config.default_ml_framework,
            genai_enabled=config.genai.enabled,
        )
