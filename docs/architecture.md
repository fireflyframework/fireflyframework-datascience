# Architecture

**Firefly DataScience is a hexagonal, auto-configured data-science framework: a lean DI container wires ports to adapters, and a Spring-Boot-style application context boots it all from packaging entry points.**

This page explains how the pieces fit together: the five layers, the ports-and-adapters (hexagonal) core, entry-point auto-configuration, the dependency-injection container, and the `FireflyDataScienceApplication` startup lifecycle. The design goal throughout is that the domain never depends on a vendor SDK, and that an adapter can be added, swapped, or overridden without touching calling code.

![Five-layer architecture](img/architecture.svg)

## The five layers

The framework is organised top-to-bottom so that the domain never depends on a vendor SDK:

1. **Application** — `FireflyDataScienceApplication` / `ApplicationContext`: the bootstrap and the started, wired context you resolve beans from.
2. **Auto-configuration** — `@auto_configuration` `@configuration` classes discovered via entry points; each contributes adapters conditionally.
3. **Container** — the `Container`: a type-hint-driven IoC container with singleton/transient scopes and constructor injection.
4. **Domain / Ports** — protocol interfaces (e.g. `DatasetLoaderPort`) plus the light, dependency-free core types in `core.types` (`TaskType`, `Modality`, `Scope`).
5. **Adapters** — concrete implementations of the ports backed by optional extras (scikit-learn, OpenML, deep-learning, GenAI, ...), each gated by a condition.

The core stays importable with **no** optional ML extra installed — vendor imports live inside adapters and `@bean` methods, never at module top level. `core.types` enforces this with hand-written `StrEnum`s (`TaskType`, `Modality`, `Scope`) and no third-party ML imports.

!!! firefly "The reproducible pattern — the LLM proposes; the classical engine decides"

    The same separation that keeps vendor SDKs out of the domain keeps GenAI out of the decision
    path. GenAI lives in **adapters** behind ports; the deterministic classical engine trains, scores
    and selects. The architecture is what makes the rule enforceable: a GenAI adapter can only ever
    *propose* — the container resolves a port, and the classical engine decides whether a proposal
    survives a measured improvement over a seeded baseline.

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

Each data-science port is declared as a `Protocol` in **its own domain module** (not in a central package): `DatasetLoaderPort` in `datasets`, `TrainerPort` in `models`, `AutoMLBackendPort` in `automl`, `FeatureEngineerPort` in `features`, `SearchPolicyPort` in `search`, `MetricsEvaluatorPort` in `evaluation`, `ValidatorPort` in `validation`, and `TrackerPort` / `RegistryPort` in `tracking`. Each is a contract the domain calls; the concrete class that fulfils it is decided at boot.

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
    def sklearn_loader(self) -> DatasetLoaderPort:  # (1)!
        from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader
        return SklearnDatasetLoader()
```

1. The `@bean` method's **return annotation is the provided type** — `DatasetLoaderPort` here — so the container registers `SklearnDatasetLoader` under the port. Resolving `DatasetLoaderPort` yields whichever adapter won. (At boot, `_apply_one` reads `get_type_hints(method)["return"]`; a `@bean` method with no return annotation is skipped.)

### Key types

A small, stable vocabulary spans the wiring layer. These are the names you actually import:

| Type / decorator | Module | Role |
| --- | --- | --- |
| `Container` | `container.container` | The IoC container; resolution by type annotation. |
| `Scope` | `core.types` | `SINGLETON` (cached, default) or `TRANSIENT` (new each resolve). |
| `@configuration` / `@bean` | `container.stereotypes` | Mark a class as holding factory methods; mark a method as a bean factory. |
| `@component` | `container.stereotypes` | Mark a class as injectable by its own type. |
| `@auto_configuration` | `container.conditions` | Mark a class discoverable via the entry-point group. |
| `@order` | `core.ordering` | Set ordering (lower runs/resolves first). |
| `ConditionContext` | `container.conditions` | What a condition is evaluated against (`config` + `container`). |
| `ApplicationContext` | `application` | A started app: the loaded config plus the wired container. |
| `WiringError` | `core.exceptions` | Raised on ambiguous, missing, or circular dependencies. |

The `@bean` decorator defaults to `scope=Scope.SINGLETON` and `primary=False`; pass `name=`, `scope=`, or `primary=` to override. `@component` and the container's `register_*` methods share the same defaults.

## Entry-point auto-configuration

Adapter packages register their auto-configuration class under the `firefly_datascience.auto_configuration` entry-point group in `pyproject.toml`:

```toml
[project.entry-points."firefly_datascience.auto_configuration"]
core     = "fireflyframework_datascience.core.auto_configuration:CoreAutoConfiguration"
datasets = "fireflyframework_datascience.datasets.auto_configuration:DatasetsAutoConfiguration"
models   = "fireflyframework_datascience.models.auto_configuration:ModelsAutoConfiguration"
```

At startup `discover_auto_configurations()` loads every class in the group, tolerating any whose optional extra is missing (it is simply skipped — its `@conditional_on_class` would have excluded it anyway), then sorts them by `@order`:

```python
from fireflyframework_datascience.core.plugin import discover_auto_configurations

for cls in discover_auto_configurations():
    print(cls.__name__)
```

`CoreAutoConfiguration` is the always-on reference example: it has no `@conditional_on_class`, so it always applies, and it registers a single `RuntimeInfo` bean snapshotting the framework version, Python version, platform, default ML framework, and whether GenAI is enabled:

```python
@auto_configuration
@configuration
class CoreAutoConfiguration:
    @bean
    def runtime_info(self, config: FireflyDataScienceConfig) -> RuntimeInfo:  # (1)!
        return RuntimeInfo(
            framework_version=__version__,
            python_version=platform.python_version(),
            platform=platform.platform(),
            default_ml_framework=config.default_ml_framework,
            genai_enabled=config.genai.enabled,
        )
```

1. The method's only parameter, `config`, is filled by type hint: `FireflyDataScienceConfig` is already registered as a bean (the application context registers it first), so the container injects it when it calls the factory.

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

- `conditional_on_class("sklearn")` matches when `importlib.util.find_spec("sklearn")` resolves — i.e. the optional extra is installed.
- `conditional_on_property("genai.enabled")` reads a dotted path off the config; with no `having_value` it matches when the value is truthy (`"1"`, `"true"`, `"yes"`, `"on"`, or any truthy object), and `match_if_missing=True` controls behaviour when the key is absent.
- `conditional_on_bean(SomePort)` / `conditional_on_missing_bean(SomePort)` query the partially-wired container — so ordering (`@order`) decides what is already present when a condition runs.

`conditional_on_missing_bean(DatasetLoaderPort)` is the **secure-by-default override hook**: a framework default applies only when you have not already registered your own. Because conditions see the live container, registering your adapter first (lower `@order`, or via `extra_auto_configurations`) is enough to win.

## The DI container

`Container` is a lean IoC container; resolution is by type annotation, with constructor injection and circular-dependency detection. There are three ways to register a bean:

=== "Register an instance"

    ```python
    from fireflyframework_datascience.container.container import Container

    container = Container()
    container.register_instance(DatasetLoaderPort, SklearnDatasetLoader())  # (1)!
    ```

    1. Register an already-constructed object as a singleton. Use this when you built the
       instance yourself (e.g. the application context registers the loaded config this way).

=== "Register a type"

    ```python
    from fireflyframework_datascience.core.types import Scope

    container.register_type(SklearnDatasetLoader, scope=Scope.SINGLETON)  # (1)!
    ```

    1. Register a class; its constructor parameters are resolved by type hint on demand.
       Pass `provided_type=` to register it under a port rather than its own class.

=== "Register a factory"

    ```python
    container.register_factory(DatasetLoaderPort, lambda: SklearnDatasetLoader())  # (1)!
    ```

    1. Register a callable whose own parameters are injected by type hint. `@bean` methods are
       registered this way under the hood.

Resolution mirrors the three shapes you need in practice:

```python
loader = container.resolve(DatasetLoaderPort)          # single bean (honours @primary)
maybe = container.resolve_optional(DatasetLoaderPort)  # None if absent
allof = container.resolve_all(DatasetLoaderPort)       # every bean, sorted by @order
```

Key behaviours:

- **Scopes** — `Scope.SINGLETON` (cached, the default) and `Scope.TRANSIENT` (new each resolve).
- **Ambiguity** — multiple beans for one type require exactly one marked `primary=True`, else `resolve` raises `WiringError`. Resolve by name with `resolve_by_name(...)` to disambiguate.
- **Injection** — constructor / factory parameters are filled by type hint; `Optional[X]` / `X | None` params resolve to `None` when no bean exists, and a parameter with a default is left to its default when no matching bean is found.
- **Circular dependencies** — detected during construction; a cycle raises `WiringError` rather than recursing.
- **Fail-fast** — `eager_init()` instantiates every singleton at boot, validating the whole wiring graph before your code runs.

!!! note "Resolution is by annotation, not by name"

    `resolve(...)` looks up registrations by the *provided type*. Names are a side channel:
    `register_*` accept a `name=`, and `resolve_by_name(...)` / `bean_names()` work off it. A bean
    with no usable return annotation is never registered (the application context skips it).

## The application lifecycle

`FireflyDataScienceApplication.start()` runs a fixed sequence, mirroring pyfly's lifecycle:

1. Load config (`FireflyDataScienceConfig.load`) — unless one is passed in.
2. Print the banner.
3. Create the `Container` and register the config as a bean.
4. Discover auto-configurations (entry points + any extras), de-duplicate while preserving order, sort by `@order`.
5. Evaluate each auto-configuration's conditions; for those that pass, instantiate the class and register every `@bean` method whose own conditions also pass.
6. `eager_init()` all singletons (fail-fast).
7. Print the wiring summary and return a ready `ApplicationContext`.

```python
from fireflyframework_datascience.application import FireflyDataScienceApplication

# One-call bootstrap.
ctx = FireflyDataScienceApplication.run()

print(ctx.bean_count, "beans")
print([ac.__name__ for ac in ctx.applied_auto_configurations])

loader = ctx.get(DatasetLoaderPort)           # resolve a bean by type
tracker = ctx.get_optional(SomeOptionalPort)  # None if not wired
```

When the banner is on, boot ends by printing the wiring summary — a quick check that the expected adapters were applied:

!!! success "Expected"

    ```text
    Firefly DataScience is ready.
      profiles      : default
      beans         : 7
      auto-config   : 3 applied (CoreAutoConfiguration, DatasetsAutoConfiguration, ModelsAutoConfiguration)
      ml framework  : sklearn
      genai         : disabled
      sandbox       : ...
    ```

    The exact bean count and applied list depend on which optional extras are installed; the line
    *shape* (profiles, beans, auto-config, ml framework, genai, sandbox) is fixed.

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

!!! tip "Quiet boots and hermetic tests"

    Pass `print_output=False` to silence the banner and wiring summary, and
    `auto_configurations=[...]` to pin an exact set of auto-configurations — together they make the
    context fully deterministic for tests, with no dependence on which extras happen to be installed.

## Auto-configuration flow

Adapters self-register via the `firefly_datascience.auto_configuration` entry-point group; the application context discovers them, evaluates their conditions, and registers the surviving beans.

<p align="center">
  <img src="img/auto-configuration.svg" alt="Entry-point auto-configuration" width="62%">
</p>

## See also

- [Quickstart](quickstart.md) — boot the application context in one call.
- [Configuration](configuration.md) — the `FireflyDataScienceConfig` that conditions read.
- [AutoML](automl.md) — what the wired ports drive end to end.
- [GenAI features](genai-features.md) — the gated adapters behind the ports.
- [Security](security.md) — the override and sandbox guarantees this wiring underpins.
