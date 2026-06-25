# Copyright 2026 Firefly Software Foundation.
"""Tests for conditional auto-configuration predicates."""

from __future__ import annotations

from fireflyframework_datascience.container.conditions import (
    ConditionContext,
    auto_configuration,
    conditional_on_bean,
    conditional_on_class,
    conditional_on_missing_bean,
    conditional_on_property,
    get_conditions,
    is_auto_configuration,
)
from fireflyframework_datascience.container.container import Container
from fireflyframework_datascience.core.config import FireflyDataScienceConfig


class _Bean:
    pass


def _ctx(**overrides: object) -> ConditionContext:
    config = FireflyDataScienceConfig(**overrides)  # type: ignore[arg-type]
    return ConditionContext(config=config, container=Container())


def test_auto_configuration_marker() -> None:
    @auto_configuration
    class AC:
        pass

    assert is_auto_configuration(AC)


def test_on_class_present_and_absent() -> None:
    present = conditional_on_class("json")(lambda: None)
    absent = conditional_on_class("totally_missing_module_xyz")(lambda: None)
    ctx = _ctx()
    assert get_conditions(present)[0].matches(ctx) is True
    assert get_conditions(absent)[0].matches(ctx) is False


def test_on_property_truthy() -> None:
    target = conditional_on_property("genai.enabled")(lambda: None)
    assert get_conditions(target)[0].matches(_ctx(genai={"enabled": True})) is True
    assert get_conditions(target)[0].matches(_ctx(genai={"enabled": False})) is False


def test_on_property_having_value() -> None:
    target = conditional_on_property("default_ml_framework", having_value="xgboost")(lambda: None)
    assert get_conditions(target)[0].matches(_ctx(default_ml_framework="xgboost")) is True
    assert get_conditions(target)[0].matches(_ctx(default_ml_framework="sklearn")) is False


def test_on_missing_bean_and_on_bean() -> None:
    config = FireflyDataScienceConfig()
    container = Container()
    ctx = ConditionContext(config=config, container=container)

    missing = conditional_on_missing_bean(_Bean)(lambda: None)
    present = conditional_on_bean(_Bean)(lambda: None)
    assert get_conditions(missing)[0].matches(ctx) is True
    assert get_conditions(present)[0].matches(ctx) is False

    container.register_instance(_Bean, _Bean())
    assert get_conditions(missing)[0].matches(ctx) is False
    assert get_conditions(present)[0].matches(ctx) is True
