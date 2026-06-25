# Copyright 2026 Firefly Software Foundation.
"""Tests for the dependency-injection container."""

from __future__ import annotations

import pytest

from fireflyframework_datascience.container.container import Container
from fireflyframework_datascience.core.exceptions import WiringError
from fireflyframework_datascience.core.ordering import order
from fireflyframework_datascience.core.types import Scope


class Repo:
    pass


class Service:
    def __init__(self, repo: Repo) -> None:
        self.repo = repo


class OptionalConsumer:
    def __init__(self, repo: Repo | None = None) -> None:
        self.repo = repo


class Port:
    pass


@order(10)
class AdapterA(Port):
    pass


@order(5)
class AdapterB(Port):
    pass


class Cyclic1:
    def __init__(self, other: Cyclic2) -> None:
        self.other = other


class Cyclic2:
    def __init__(self, other: Cyclic1) -> None:
        self.other = other


def test_register_instance_and_resolve() -> None:
    c = Container()
    repo = Repo()
    c.register_instance(Repo, repo)
    assert c.resolve(Repo) is repo
    assert c.has(Repo)


def test_constructor_injection() -> None:
    c = Container()
    c.register_type(Repo)
    c.register_type(Service)
    svc = c.resolve(Service)
    assert isinstance(svc, Service)
    assert isinstance(svc.repo, Repo)


def test_singleton_is_cached() -> None:
    c = Container()
    c.register_type(Repo)
    assert c.resolve(Repo) is c.resolve(Repo)


def test_transient_is_fresh() -> None:
    c = Container()
    c.register_type(Repo, scope=Scope.TRANSIENT)
    assert c.resolve(Repo) is not c.resolve(Repo)


def test_optional_dependency_resolves_none() -> None:
    c = Container()
    c.register_type(OptionalConsumer)
    consumer = c.resolve(OptionalConsumer)
    assert consumer.repo is None


def test_missing_bean_raises() -> None:
    c = Container()
    with pytest.raises(WiringError):
        c.resolve(Repo)


def test_resolve_all_sorted_by_order() -> None:
    c = Container()
    c.register_type(AdapterA, provided_type=Port, name="a")
    c.register_type(AdapterB, provided_type=Port, name="b")
    adapters = c.resolve_all(Port)
    assert [type(a) for a in adapters] == [AdapterB, AdapterA]  # order 5 before 10


def test_ambiguous_without_primary_raises() -> None:
    c = Container()
    c.register_type(AdapterA, provided_type=Port, name="a")
    c.register_type(AdapterB, provided_type=Port, name="b")
    with pytest.raises(WiringError):
        c.resolve(Port)


def test_primary_disambiguates() -> None:
    c = Container()
    c.register_type(AdapterA, provided_type=Port, name="a")
    c.register_type(AdapterB, provided_type=Port, name="b", primary=True)
    assert isinstance(c.resolve(Port), AdapterB)


def test_factory_injection() -> None:
    c = Container()
    c.register_instance(Repo, Repo())

    def make_service(repo: Repo) -> Service:
        return Service(repo)

    c.register_factory(Service, make_service)
    assert isinstance(c.resolve(Service), Service)


def test_circular_dependency_detected() -> None:
    c = Container()
    c.register_type(Cyclic1)
    c.register_type(Cyclic2)
    with pytest.raises(WiringError, match="[Cc]ircular"):
        c.resolve(Cyclic1)


def test_resolve_by_name() -> None:
    c = Container()
    c.register_type(AdapterA, provided_type=Port, name="a")
    assert isinstance(c.resolve_by_name("a"), AdapterA)
