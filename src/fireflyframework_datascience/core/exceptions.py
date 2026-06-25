# Copyright 2026 Firefly Software Foundation.
"""Exception hierarchy for Firefly DataScience.

All framework errors derive from :class:`FireflyDataScienceError` so callers can catch the whole
family with a single ``except``.
"""

from __future__ import annotations


class FireflyDataScienceError(Exception):
    """Base class for every Firefly DataScience error."""


class ConfigurationError(FireflyDataScienceError):
    """Raised when configuration is invalid, missing, or cannot be loaded."""


class WiringError(FireflyDataScienceError):
    """Raised when the dependency-injection container cannot satisfy a dependency."""


class AutoConfigurationError(FireflyDataScienceError):
    """Raised when an auto-configuration class fails to load or apply."""


class AdapterUnavailableError(FireflyDataScienceError):
    """Raised when an adapter's optional dependency (``extra``) is not installed.

    The message tells the user exactly which extra to install, e.g.
    ``pip install 'fireflyframework-datascience[tabular]'``.
    """

    def __init__(self, adapter: str, extra: str) -> None:
        super().__init__(
            f"Adapter {adapter!r} requires the optional dependency group {extra!r}. "
            f"Install it with: pip install 'fireflyframework-datascience[{extra}]'"
        )
        self.adapter = adapter
        self.extra = extra
