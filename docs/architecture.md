# Architecture

**Firefly DataScience is a hexagonal, auto-configured data-science framework: a light DI container wires ports to adapters, and a Spring-Boot-style application context boots it all from packaging entry points.**

This page explains how the pieces fit together: the five layers, the ports-and-adapters (hexagonal) core, entry-point auto-configuration, the dependency-injection container, and the `FireflyDataScienceApplication` startup lifecycle.

![Five-layer architecture](img/architecture.svg)

## The five layers

The framework is organised top-to-bottom so that the domain never depends on a vendor SDK:

1. **Application** — `FireflyDataScienceApplication` / `ApplicationContext`: the bootstrap and the started, wired context you resolve beans from.
2. **Auto-configuration** — `@auto_configuration` `@configuration` classes discovered via entry points; each contributes adapters conditionally.
3. **Container** — the `Container`: a type-hint-driven IoC container with singleton/transient scopes and constructor injection.
4. **Domain / Ports** — protocol interfaces (e.g. `DatasetLoaderPort`) plus the light, dependency-free core types in `core.types` (`TaskType`, `Modality`, `Scope`).
5. **Adapters** — concrete implementations of the ports backed by optional extras (scikit-learn, OpenML, deep-learning, GenAI, ...), each gated by a condition.

The core stays importable with **no** optional ML extra installed — vendor imports live inside adapters and `@bean` methods, never at module top level.

## Hexagonal: ports and adapters

![Ports and adapters](img/hexagonal.svg)

A **port** is a `Protocol` the domain depends on. An **adapter** is a concrete class that implements it. The container binds them by type annotation, so swapping an adapter never touches calling code.

```python
from typing import Protocol


class DatasetLoaderPort(Protocol):
    def load(self, name: str) -> object: ...
```

```python
class SklearnDatasetLoader:
    def load(self, name: str) -> object:
        from sklearn import datasets
        return getattr(datasets, f"load_{name}")()
```

The adapter is contributed by an auto-configuration, gated on the optional dependency being importable:

```python
from fireflyframework_datascience.container.conditions import (
    auto_configuration,
    conditional_on_class,
)
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.datasets import DatasetLoaderPort


@auto_configuration
@conditional_on_class("sklearn")
@configuration
class DatasetsAutoConfiguration:
    @bean(name="sklearn_dataset_loader")
    def sklearn_loader(self) -> DatasetLoaderPort:
        from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader
        return SklearnDatasetLoader()
```

The `@bean` method's **return annotation is the provided type** — `DatasetLoaderPort` here — so the container registers `SklearnDatasetLoader` under the port. Resolving `DatasetLoaderPort` yields whichever adapter won.

## Entry-point auto-configuration

Adapter packages register their auto-configuration class under the `firefly_datascience.auto_configuration` entry-point group in `pyproject.toml`:

```toml
[project.entry-points."firefly_datascience.auto_configuration"]
core     = "fireflyframework_datascience.core.auto_configuration:CoreAutoConfiguration"
datasets = "fireflyframework_datascience.datasets.auto_configuration:DatasetsAutoConfiguration"
models   = "fireflyframework_datascience.models.auto_configuration:ModelsAutoConfiguration"
```

At startup `discover_auto_configurations()` loads every class in the group, tolerating any whose optional extra is missing (it is simply skipped), then sorts them by `@order`:

```python
from fireflyframework_datascience.core.plugin import discover_auto_configurations

for cls in discover_auto_configurations():
    print(cls.__name__)
```

### Conditions

Conditions gate both whole auto-configurations and individual `@bean` methods. Each is evaluated against a `ConditionContext` (the loaded config plus the partially-wired container):

```python
from fireflyframework_datascience.container.conditions import (
    conditional_on_class,        # an optional extra is importable
    conditional_on_property,     # a config key is set / equals a value
    conditional_on_missing_bean, # user override wins
    conditional_on_bean,         # another bean is already present
)
```

`conditional_on_missing_bean(DatasetLoaderPort)` is the **secure-by-default override hook**: a framework default applies only when you have not already registered your own.

## The DI container

`Container` is a lean IoC container; resolution is by type annotation, with constructor injection and circular-dependency detection.

```python
from fireflyframework_datascience.container.container import Container
from fireflyframework_datascience.core.types import Scope

container = Container()

# Three ways to register a bean:
container.register_instance(DatasetLoaderPort, SklearnDatasetLoader())  # pre-built
container.register_type(SklearnDatasetLoader, scope=Scope.SINGLETON)    # ctor-injected
container.register_factory(DatasetLoaderPort, lambda: SklearnDatasetLoader())  # factory

loader = container.resolve(DatasetLoaderPort)        # single bean (honours @primary)
maybe = container.resolve_optional(DatasetLoaderPort)  # None if absent
allof = container.resolve_all(DatasetLoaderPort)       # every bean, sorted by @order
```

Key behaviours:

- **Scopes** — `Scope.SINGLETON` (cached, the default) and `Scope.TRANSIENT` (new each resolve).
- **Ambiguity** — multiple beans for one type require exactly one marked `primary=True`, else `resolve` raises `WiringError`. Resolve by name with `resolve_by_name(...)` to disambiguate.
- **Injection** — constructor / factory parameters are filled by type hint; `Optional[X]` / `X | None` params resolve to `None` when no bean exists.
- **Fail-fast** — `eager_init()` instantiates every singleton at boot, validating the whole wiring graph before your code runs.

## The application lifecycle

`FireflyDataScienceApplication.start()` runs a fixed sequence, mirroring pyfly's lifecycle:

1. Load config (`FireflyDataScienceConfig.load`) — unless one is passed in.
2. Print the banner.
3. Create the `Container` and register the config as a bean.
4. Discover auto-configurations (entry points + any extras), de-duplicate, sort by `@order`.
5. Evaluate each auto-configuration's conditions; for those that pass, instantiate the class and register every `@bean` method whose own conditions also pass.
6. `eager_init()` all singletons (fail-fast).
7. Print the wiring summary and return a ready `ApplicationContext`.

```python
from fireflyframework_datascience.application import FireflyDataScienceApplication

# One-call bootstrap.
ctx = FireflyDataScienceApplication.run()

print(ctx.bean_count, "beans")
print([ac.__name__ for ac in ctx.applied_auto_configurations])

loader = ctx.get(DatasetLoaderPort)          # resolve a bean by type
tracker = ctx.get_optional(SomeOptionalPort)  # None if not wired
```

You can steer the boot without forking the framework:

```python
ctx = FireflyDataScienceApplication.run(
    config_dir="config",
    profiles=["prod"],
    extra_auto_configurations=[MyCustomAutoConfiguration],  # add your own
    print_output=False,                                     # quiet boot for tests
)
```

Passing `auto_configurations=[...]` **replaces** discovery entirely (handy for hermetic tests); `extra_auto_configurations=[...]` **appends** to whatever was discovered. The returned `ApplicationContext` exposes `.config`, `.container`, `.applied_auto_configurations`, `.bean_count`, and the `get` / `get_optional` resolvers.

## See also

- [Getting started](quickstart.md)
- [Configuration](./configuration.md)
- [Ports and adapters reference](index.md)
- [Writing an auto-configuration](index.md)
