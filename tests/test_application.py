# Copyright 2026 Firefly Software Foundation.
"""Tests for the application bootstrap lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

from fireflyframework_datascience import FireflyDataScienceApplication
from fireflyframework_datascience.container.conditions import (
    auto_configuration,
    conditional_on_property,
)
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.core.auto_configuration import (
    CoreAutoConfiguration,
    RuntimeInfo,
)
from fireflyframework_datascience.core.banner import BannerMode
from fireflyframework_datascience.core.config import FireflyDataScienceConfig


@dataclass(frozen=True)
class GenAIMarker:
    note: str


@auto_configuration
@conditional_on_property("genai.enabled", having_value=True)
@configuration
class GenAIGatedAutoConfig:
    @bean
    def marker(self, config: FireflyDataScienceConfig) -> GenAIMarker:
        return GenAIMarker(note="genai-on")


def test_app_starts_with_core_autoconfig() -> None:
    ctx = FireflyDataScienceApplication.run(
        auto_configurations=[CoreAutoConfiguration],
        banner_mode=BannerMode.OFF,
        print_output=False,
    )
    assert ctx.bean_count >= 2  # config + runtime_info
    info = ctx.get(RuntimeInfo)
    assert isinstance(info, RuntimeInfo)
    assert CoreAutoConfiguration in ctx.applied_auto_configurations


def test_conditional_autoconfig_skipped_when_off() -> None:
    ctx = FireflyDataScienceApplication.run(
        auto_configurations=[CoreAutoConfiguration, GenAIGatedAutoConfig],
        print_output=False,
    )
    assert GenAIGatedAutoConfig not in ctx.applied_auto_configurations
    assert ctx.get_optional(GenAIMarker) is None


def test_conditional_autoconfig_applied_when_enabled() -> None:
    config = FireflyDataScienceConfig(genai={"enabled": True})  # type: ignore[arg-type]
    ctx = FireflyDataScienceApplication.run(
        config=config,
        auto_configurations=[CoreAutoConfiguration, GenAIGatedAutoConfig],
        print_output=False,
    )
    assert GenAIGatedAutoConfig in ctx.applied_auto_configurations
    marker = ctx.get(GenAIMarker)
    assert marker.note == "genai-on"


def test_discovery_finds_core_autoconfig_via_entry_points() -> None:
    # No explicit auto_configurations → uses entry-point discovery.
    ctx = FireflyDataScienceApplication.run(print_output=False)
    assert any(ac.__name__ == "CoreAutoConfiguration" for ac in ctx.applied_auto_configurations)
