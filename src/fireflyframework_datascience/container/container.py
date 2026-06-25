# Copyright 2026 Firefly Software Foundation.
"""A lean, type-hint-driven dependency-injection container.

Mirrors the ergonomics of pyfly's ``Container`` (constructor injection, singleton/transient scopes,
``resolve`` / ``resolve_all`` / interface binding, ``@order`` sorting, circular-dependency detection)
without dragging in the full pyfly framework. Resolution is by type annotation.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from fireflyframework_datascience.core.exceptions import WiringError
from fireflyframework_datascience.core.ordering import get_order
from fireflyframework_datascience.core.types import Scope

T = TypeVar("T")

_EMPTY = inspect.Parameter.empty


class _Registration:
    """A single bean registration."""

    __slots__ = ("provided_type", "factory", "scope", "name", "primary", "order", "source")

    def __init__(
        self,
        provided_type: type,
        factory: Callable[[], Any],
        *,
        scope: Scope,
        name: str,
        primary: bool,
        order: int,
        source: object,
    ) -> None:
        self.provided_type = provided_type
        self.factory = factory
        self.scope = scope
        self.name = name
        self.primary = primary
        self.order = order
        self.source = source


class Container:
    """Type-hint-based IoC container."""

    def __init__(self) -> None:
        self._by_type: dict[type, list[_Registration]] = {}
        self._by_name: dict[str, _Registration] = {}
        self._singletons: dict[int, Any] = {}
        self._building: set[type] = set()
        self._lock = threading.RLock()

    # -- registration -----------------------------------------------------

    def register_instance(self, provided_type: type[T], instance: T, *, name: str = "", primary: bool = False) -> None:
        """Register an already-constructed instance as a singleton."""
        reg = _Registration(
            provided_type,
            lambda: instance,
            scope=Scope.SINGLETON,
            name=name,
            primary=primary,
            order=get_order(type(instance)),
            source=instance,
        )
        self._singletons[id(reg)] = instance
        self._add(reg)

    def register_type(
        self,
        impl: type[T],
        *,
        provided_type: type | None = None,
        scope: Scope = Scope.SINGLETON,
        name: str = "",
        primary: bool = False,
    ) -> None:
        """Register a class; its constructor dependencies are injected on demand."""
        ptype = provided_type or impl
        reg = _Registration(
            ptype,
            lambda: self._construct(impl),
            scope=scope,
            name=name or impl.__name__,
            primary=primary,
            order=get_order(impl),
            source=impl,
        )
        self._add(reg)

    def register_factory(
        self,
        provided_type: type[T],
        factory: Callable[..., T],
        *,
        scope: Scope = Scope.SINGLETON,
        name: str = "",
        primary: bool = False,
        order: int = 0,
    ) -> None:
        """Register a factory callable; its parameters are injected by type hint."""
        reg = _Registration(
            provided_type,
            lambda: self._call_injected(factory),
            scope=scope,
            name=name or getattr(factory, "__name__", provided_type.__name__),
            primary=primary,
            order=order,
            source=factory,
        )
        self._add(reg)

    def _add(self, reg: _Registration) -> None:
        with self._lock:
            self._by_type.setdefault(reg.provided_type, []).append(reg)
            if reg.name:
                self._by_name[reg.name] = reg

    # -- resolution -------------------------------------------------------

    def has(self, provided_type: type) -> bool:
        """True if at least one bean is registered for ``provided_type``."""
        return bool(self._by_type.get(provided_type))

    def resolve(self, provided_type: type[T]) -> T:
        """Resolve a single bean for ``provided_type`` (honours ``@primary`` when ambiguous)."""
        regs = self._by_type.get(provided_type)
        if not regs:
            raise WiringError(f"No bean registered for type {provided_type!r}")
        reg = self._select(regs, provided_type)
        return self._instantiate(reg)

    def resolve_optional(self, provided_type: type[T]) -> T | None:
        """Resolve a bean, or ``None`` if none is registered."""
        return self.resolve(provided_type) if self.has(provided_type) else None

    def resolve_by_name(self, name: str) -> Any:
        """Resolve a bean by its registered name."""
        reg = self._by_name.get(name)
        if reg is None:
            raise WiringError(f"No bean registered under name {name!r}")
        return self._instantiate(reg)

    def resolve_all(self, provided_type: type[T]) -> list[T]:
        """Resolve every bean registered for ``provided_type``, sorted by ``@order``."""
        regs = sorted(self._by_type.get(provided_type, []), key=lambda r: r.order)
        return [self._instantiate(r) for r in regs]

    def bean_names(self) -> list[str]:
        """All registered bean names, sorted."""
        return sorted(self._by_name)

    def eager_init(self) -> None:
        """Eagerly instantiate every singleton, validating the wiring graph fail-fast."""
        for regs in list(self._by_type.values()):
            for reg in regs:
                if reg.scope is Scope.SINGLETON:
                    self._instantiate(reg)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_type.values())

    # -- internals --------------------------------------------------------

    def _select(self, regs: list[_Registration], provided_type: type) -> _Registration:
        if len(regs) == 1:
            return regs[0]
        primaries = [r for r in regs if r.primary]
        if len(primaries) == 1:
            return primaries[0]
        raise WiringError(
            f"Ambiguous dependency for {provided_type!r}: {len(regs)} candidates and "
            f"{len(primaries)} marked primary. Mark exactly one as primary or resolve by name."
        )

    def _instantiate(self, reg: _Registration) -> Any:
        if reg.scope is Scope.SINGLETON and id(reg) in self._singletons:
            return self._singletons[id(reg)]
        instance = reg.factory()
        if reg.scope is Scope.SINGLETON:
            self._singletons[id(reg)] = instance
        return instance

    def _construct(self, impl: type) -> Any:
        if impl in self._building:
            raise WiringError(f"Circular dependency detected while constructing {impl!r}")
        self._building.add(impl)
        try:
            kwargs = self._injected_kwargs(impl.__init__, impl, owner=impl)
            return impl(**kwargs)
        finally:
            self._building.discard(impl)

    def _call_injected(self, factory: Callable[..., Any]) -> Any:
        kwargs = self._injected_kwargs(factory, factory)
        return factory(**kwargs)

    def _injected_kwargs(
        self, func: Callable[..., Any], hint_source: Any, *, owner: type | None = None
    ) -> dict[str, Any]:
        try:
            hints = get_type_hints(func)
        except Exception:  # noqa: BLE001 - degrade gracefully on un-evaluatable hints
            hints = {}
        sig = inspect.signature(func)
        kwargs: dict[str, Any] = {}
        for pname, param in sig.parameters.items():
            if pname == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            annotation = hints.get(pname)
            if annotation is None:
                if param.default is _EMPTY:
                    raise WiringError(
                        f"Cannot inject parameter {pname!r} of {getattr(hint_source, '__name__', hint_source)!r}: "
                        f"no type annotation and no default"
                    )
                continue
            actual, optional = _unwrap_optional(annotation)
            if isinstance(actual, type) and self.has(actual):
                kwargs[pname] = self.resolve(actual)
            elif param.default is not _EMPTY:
                continue
            elif optional:
                kwargs[pname] = None
            else:
                raise WiringError(
                    f"Cannot resolve dependency {pname!r}: {actual!r} required by "
                    f"{getattr(hint_source, '__name__', hint_source)!r}"
                )
        return kwargs


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return (inner_type, is_optional) for ``Optional[X]`` / ``X | None``; else (annotation, False)."""
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
    return annotation, False
