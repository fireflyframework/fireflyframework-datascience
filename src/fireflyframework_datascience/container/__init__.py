# Copyright 2026 Firefly Software Foundation.
"""Dependency-injection container, conditions, and stereotypes (mirrors pyfly's IoC)."""

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
from fireflyframework_datascience.container.stereotypes import (
    bean,
    component,
    configuration,
    get_bean_methods,
    is_configuration,
)

__all__ = [
    "ConditionContext",
    "Container",
    "auto_configuration",
    "bean",
    "component",
    "conditional_on_bean",
    "conditional_on_class",
    "conditional_on_missing_bean",
    "conditional_on_property",
    "configuration",
    "get_bean_methods",
    "get_conditions",
    "is_auto_configuration",
    "is_configuration",
]
