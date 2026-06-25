# Copyright 2026 Firefly Software Foundation.
"""Conditional auto-configuration (mirrors pyfly / Spring Boot ``@ConditionalOn*``).

Decorators attach :class:`Condition` objects to a class; the application context evaluates them against
a :class:`ConditionContext` (the loaded config + the partially-wired container) to decide whether an
auto-configuration (or one of its beans) should apply.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from fireflyframework_datascience.container.container import Container
    from fireflyframework_datascience.core.config import FireflyDataScienceConfig

T = TypeVar("T")

_CONDITIONS_ATTR = "__firefly_ds_conditions__"
_AUTO_CONFIG_ATTR = "__firefly_ds_auto_config__"
_MISSING = object()


@dataclass(frozen=True)
class ConditionContext:
    """What a condition is evaluated against."""

    config: FireflyDataScienceConfig
    container: Container


@runtime_checkable
class Condition(Protocol):
    """A predicate that gates an auto-configuration or bean."""

    def matches(self, ctx: ConditionContext) -> bool: ...

    def describe(self) -> str: ...


def _attach(target: T, condition: Condition) -> T:
    conditions: list[Condition] = list(getattr(target, _CONDITIONS_ATTR, ()))
    conditions.append(condition)
    setattr(target, _CONDITIONS_ATTR, conditions)
    return target


def get_conditions(obj: object) -> list[Condition]:
    """Return all conditions attached to ``obj`` (empty if none)."""
    return list(getattr(obj, _CONDITIONS_ATTR, ()))


def auto_configuration(cls: type[T]) -> type[T]:
    """Mark a class as an auto-configuration (discoverable via the entry-point group)."""
    setattr(cls, _AUTO_CONFIG_ATTR, True)
    return cls


def is_auto_configuration(obj: object) -> bool:
    """True if ``obj`` is marked as an auto-configuration."""
    return bool(getattr(obj, _AUTO_CONFIG_ATTR, False))


# --- concrete conditions ------------------------------------------------


@dataclass(frozen=True)
class _OnClass:
    module: str

    def matches(self, ctx: ConditionContext) -> bool:
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            return False

    def describe(self) -> str:
        return f"on_class({self.module})"


@dataclass(frozen=True)
class _OnProperty:
    key: str
    having_value: Any
    match_if_missing: bool

    def matches(self, ctx: ConditionContext) -> bool:
        value = _resolve_dotted(ctx.config, self.key)
        if value is _MISSING:
            return self.match_if_missing
        if self.having_value is None:
            return _truthy(value)
        return _coerce_equal(value, self.having_value)

    def describe(self) -> str:
        return f"on_property({self.key}={self.having_value!r})"


@dataclass(frozen=True)
class _OnMissingBean:
    bean_type: type

    def matches(self, ctx: ConditionContext) -> bool:
        return not ctx.container.has(self.bean_type)

    def describe(self) -> str:
        return f"on_missing_bean({self.bean_type.__name__})"


@dataclass(frozen=True)
class _OnBean:
    bean_type: type

    def matches(self, ctx: ConditionContext) -> bool:
        return ctx.container.has(self.bean_type)

    def describe(self) -> str:
        return f"on_bean({self.bean_type.__name__})"


def conditional_on_class(module: str) -> Callable[[T], T]:
    """Apply only if ``module`` is importable (an optional extra is installed)."""
    return lambda target: _attach(target, _OnClass(module))


def conditional_on_property(key: str, *, having_value: Any = None, match_if_missing: bool = False) -> Callable[[T], T]:
    """Apply only if config ``key`` equals ``having_value`` (or is truthy when no value given)."""
    return lambda target: _attach(target, _OnProperty(key, having_value, match_if_missing))


def conditional_on_missing_bean(bean_type: type) -> Callable[[T], T]:
    """Apply only if no bean of ``bean_type`` is already registered (user override wins)."""
    return lambda target: _attach(target, _OnMissingBean(bean_type))


def conditional_on_bean(bean_type: type) -> Callable[[T], T]:
    """Apply only if a bean of ``bean_type`` is already registered."""
    return lambda target: _attach(target, _OnBean(bean_type))


# --- helpers ------------------------------------------------------------


def _resolve_dotted(config: object, dotted: str) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if current is None:
            return _MISSING
        current = getattr(current, part, _MISSING)
        if current is _MISSING:
            return _MISSING
    return current


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_equal(value: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or isinstance(value, bool):
        return _truthy(value) == _truthy(expected)
    return str(value).strip().lower() == str(expected).strip().lower()
