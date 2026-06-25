# Copyright 2026 Firefly Software Foundation.
"""Tests for the startup banner."""

from __future__ import annotations

from fireflyframework_datascience.core.banner import BannerMode, BannerPrinter
from fireflyframework_datascience.core.config import FireflyDataScienceConfig


def test_off_mode_renders_empty() -> None:
    printer = BannerPrinter(BannerMode.OFF, framework_version="26.6.0")
    assert printer.render() == ""


def test_minimal_mode_contains_version() -> None:
    printer = BannerPrinter(BannerMode.MINIMAL, framework_version="26.6.0")
    rendered = printer.render()
    assert "Firefly DataScience" in rendered
    assert "26.6.0" in rendered


def test_text_mode_contains_ascii_and_status() -> None:
    printer = BannerPrinter(
        BannerMode.TEXT,
        framework_version="26.6.0",
        active_profiles=["dev"],
        genai_enabled=True,
    )
    rendered = printer.render()
    assert "AutoML" in rendered
    assert "profiles=['dev']" in rendered
    assert "genai=on" in rendered


def test_from_config_reads_mode_and_genai() -> None:
    config = FireflyDataScienceConfig(
        banner={"mode": "MINIMAL"},  # type: ignore[arg-type]
        genai={"enabled": True},  # type: ignore[arg-type]
    )
    printer = BannerPrinter.from_config(config)
    assert printer.mode is BannerMode.MINIMAL
    assert printer.genai_enabled is True
