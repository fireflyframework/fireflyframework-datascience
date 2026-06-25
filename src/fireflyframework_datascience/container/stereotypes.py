# Copyright 2026 Firefly Software Foundation.
"""Stereotype decorators: ``@configuration`` + ``@bean`` (factory methods) and ``@component``.

An auto-configuration is typically a ``@configuration`` class whose ``@bean`` methods produce adapters;
the application context instantiates the class and calls each ``@bean`` method with injected arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from fireflyframework_datascience.core.types import Scope

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

_CONFIGURATION_ATTR = "__firefly_ds_configuration__"
_COMPONENT_ATTR = "__firefly_ds_component__"
_BEAN_ATTR = "__firefly_ds_bean__"


@dataclass(frozen=True)
class BeanMeta:
    """Metadata recorded on a ``@bean`` factory method."""

    name: str | None
    scope: Scope
    primary: bool


@dataclass(frozen=True)
class ComponentMeta:
    """Metadata recorded on a ``@component`` class."""

    name: str | None
    scope: Scope
    primary: bool


def configuration(cls: type[T]) -> type[T]:
    """Mark a class as holding ``@bean`` factory methods."""
    setattr(cls, _CONFIGURATION_ATTR, True)
    return cls


def is_configuration(obj: object) -> bool:
    """True if ``obj`` is a ``@configuration`` class."""
    return bool(getattr(obj, _CONFIGURATION_ATTR, False))


@overload
def bean(func: F) -> F: ...
@overload
def bean(*, name: str | None = ..., scope: Scope = ..., primary: bool = ...) -> Callable[[F], F]: ...


def bean(
    func: F | None = None,
    *,
    name: str | None = None,
    scope: Scope = Scope.SINGLETON,
    primary: bool = False,
) -> F | Callable[[F], F]:
    """Mark a method as a bean factory. The return annotation is the provided type."""

    def _decorate(fn: F) -> F:
        setattr(fn, _BEAN_ATTR, BeanMeta(name=name, scope=scope, primary=primary))
        return fn

    return _decorate(func) if func is not None else _decorate


def get_bean_methods(instance: object) -> list[tuple[Callable[..., Any], BeanMeta]]:
    """Return ``(method, meta)`` pairs for every ``@bean`` method on a configuration instance."""
    methods: list[tuple[Callable[..., Any], BeanMeta]] = []
    for attr_name in dir(instance):
        if attr_name.startswith("__"):
            continue
        attr = getattr(instance, attr_name)
        meta = getattr(attr, _BEAN_ATTR, None)
        if isinstance(meta, BeanMeta):
            methods.append((attr, meta))
    return methods


def component(
    cls: type[T] | None = None,
    *,
    name: str | None = None,
    scope: Scope = Scope.SINGLETON,
    primary: bool = False,
) -> type[T] | Callable[[type[T]], type[T]]:
    """Mark a class as an injectable component (registered by its own type)."""

    def _decorate(target: type[T]) -> type[T]:
        setattr(target, _COMPONENT_ATTR, ComponentMeta(name=name, scope=scope, primary=primary))
        return target

    return _decorate(cls) if cls is not None else _decorate


def get_component_meta(obj: object) -> ComponentMeta | None:
    """Return the :class:`ComponentMeta` for a ``@component`` class, or ``None``."""
    meta = getattr(obj, _COMPONENT_ATTR, None)
    return meta if isinstance(meta, ComponentMeta) else None
