# Copyright 2026 Firefly Software Foundation.
"""Startup banner (mirrors pyfly's ``BannerPrinter`` / ``BannerMode``).

Three modes — ``TEXT`` (full ASCII art), ``MINIMAL`` (single line), ``OFF`` (nothing) — selectable via
config key ``banner.mode`` or the ``FIREFLY_DATASCIENCE_BANNER__MODE`` environment variable.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING

from fireflyframework_datascience._version import __version__

if TYPE_CHECKING:
    from fireflyframework_datascience.core.config import FireflyDataScienceConfig

# Raw string: backslashes are literal art, not escape sequences.
_ASCII = r"""
   __ _           __ _          ___  ___
  / _(_)_ _ ___  / _| |_ _  _  |   \/ __|
 |  _| | '_/ -_)|  _| | || | | | |) \__ \
 |_| |_|_| \___||_| |_|\_, |_| |___/|___/
                       |__/
"""


class BannerMode(StrEnum):
    """How (or whether) to print the startup banner."""

    TEXT = "TEXT"
    MINIMAL = "MINIMAL"
    OFF = "OFF"


class BannerPrinter:
    """Renders the Firefly DataScience startup banner."""

    def __init__(
        self,
        mode: BannerMode = BannerMode.TEXT,
        *,
        framework_version: str,
        app_name: str | None = None,
        app_version: str | None = None,
        active_profiles: Sequence[str] = (),
        genai_enabled: bool = False,
    ) -> None:
        self.mode = mode
        self.framework_version = framework_version
        self.app_name = app_name
        self.app_version = app_version
        self.active_profiles = list(active_profiles)
        self.genai_enabled = genai_enabled

    @classmethod
    def from_config(
        cls,
        config: FireflyDataScienceConfig,
        *,
        app_name: str | None = None,
        app_version: str | None = None,
    ) -> BannerPrinter:
        """Build a printer from a loaded :class:`FireflyDataScienceConfig`."""
        return cls(
            mode=config.banner.mode,
            framework_version=__version__,
            app_name=app_name,
            app_version=app_version,
            active_profiles=config.profiles,
            genai_enabled=config.genai.enabled,
        )

    def _status_line(self) -> str:
        parts = [f":: Firefly DataScience :: (v{self.framework_version})"]
        if self.app_name:
            app = self.app_name + (f" v{self.app_version}" if self.app_version else "")
            parts.append(f"app={app}")
        if self.active_profiles:
            parts.append(f"profiles={self.active_profiles}")
        parts.append(f"genai={'on' if self.genai_enabled else 'off'}")
        return "  ".join(parts)

    def render(self) -> str:
        """Return the banner as plain text (empty string when ``OFF``)."""
        if self.mode is BannerMode.OFF:
            return ""
        if self.mode is BannerMode.MINIMAL:
            return f":: Firefly DataScience :: (v{self.framework_version})"
        tagline = "   AutoML · GenAI × classical ML × Deep Learning · built on Firefly Agentic"
        return f"{_ASCII.rstrip()}\n{tagline}\n\n{self._status_line()}"
