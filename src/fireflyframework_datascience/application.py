# Copyright 2026 Firefly Software Foundation.
"""Application bootstrap (mirrors pyfly's ``PyFlyApplication`` lifecycle).

Startup sequence: load config → print banner → create the DI container (register the config) →
discover auto-configurations → evaluate their conditions and register beans → eagerly initialize
singletons → print the wiring summary → return a ready :class:`ApplicationContext`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, get_type_hints

from rich.console import Console

from fireflyframework_datascience.container.conditions import (
    ConditionContext,
    get_conditions,
    is_auto_configuration,
)
from fireflyframework_datascience.container.container import Container
from fireflyframework_datascience.container.stereotypes import (
    get_bean_methods,
    get_component_meta,
    is_configuration,
)
from fireflyframework_datascience.core.banner import BannerMode, BannerPrinter
from fireflyframework_datascience.core.config import FireflyDataScienceConfig
from fireflyframework_datascience.core.ordering import get_order
from fireflyframework_datascience.core.plugin import discover_auto_configurations

logger = logging.getLogger(__name__)


class ApplicationContext:
    """A started application: the loaded config plus the wired container."""

    def __init__(
        self,
        config: FireflyDataScienceConfig,
        container: Container,
        applied_auto_configurations: Sequence[type],
    ) -> None:
        self._config = config
        self._container = container
        self._applied = list(applied_auto_configurations)

    @property
    def config(self) -> FireflyDataScienceConfig:
        return self._config

    @property
    def container(self) -> Container:
        return self._container

    @property
    def applied_auto_configurations(self) -> list[type]:
        return list(self._applied)

    @property
    def bean_count(self) -> int:
        return len(self._container)

    def get(self, provided_type: type) -> Any:
        """Resolve a bean by type."""
        return self._container.resolve(provided_type)

    def get_optional(self, provided_type: type) -> Any:
        """Resolve a bean by type, or ``None`` if absent."""
        return self._container.resolve_optional(provided_type)


class FireflyDataScienceApplication:
    """Bootstraps a Firefly DataScience application."""

    def __init__(
        self,
        *,
        config: FireflyDataScienceConfig | None = None,
        config_dir: str | None = None,
        profiles: list[str] | None = None,
        app_name: str | None = None,
        app_version: str | None = None,
        banner_mode: BannerMode | None = None,
        print_output: bool = True,
        auto_configurations: list[type] | None = None,
        extra_auto_configurations: list[type] | None = None,
    ) -> None:
        self._config = config
        self._config_dir = config_dir
        self._profiles = profiles
        self._app_name = app_name
        self._app_version = app_version
        self._banner_mode = banner_mode
        self._print_output = print_output
        self._auto_configurations = auto_configurations
        self._extra_auto_configurations = extra_auto_configurations or []
        self._console = Console()

    @classmethod
    def run(cls, **kwargs: Any) -> ApplicationContext:
        """Construct and start an application in one call."""
        return cls(**kwargs).start()

    def start(self) -> ApplicationContext:
        config = self._config or FireflyDataScienceConfig.load(config_dir=self._config_dir, profiles=self._profiles)
        if self._banner_mode is not None:
            config.banner.mode = self._banner_mode

        self._emit_banner(config)

        container = Container()
        container.register_instance(FireflyDataScienceConfig, config)

        auto_configs = self._collect_auto_configurations()
        applied = self._apply_auto_configurations(auto_configs, config, container)

        container.eager_init()

        self._emit_wiring_summary(config, container, applied)
        logger.info(
            "Firefly DataScience application ready: %d beans, %d auto-configurations", len(container), len(applied)
        )
        return ApplicationContext(config, container, applied)

    # -- steps ------------------------------------------------------------

    def _collect_auto_configurations(self) -> list[type]:
        base = self._auto_configurations if self._auto_configurations is not None else discover_auto_configurations()
        combined = [*base, *self._extra_auto_configurations]
        # de-duplicate while preserving order, then sort by @order
        seen: set[type] = set()
        unique = [ac for ac in combined if not (ac in seen or seen.add(ac))]
        unique.sort(key=get_order)
        return unique

    def _apply_auto_configurations(
        self, auto_configs: list[type], config: FireflyDataScienceConfig, container: Container
    ) -> list[type]:
        applied: list[type] = []
        for ac in auto_configs:
            ctx = ConditionContext(config=config, container=container)
            if not _all_match(get_conditions(ac), ctx):
                logger.debug("Auto-configuration %s skipped (conditions not met)", ac.__name__)
                continue
            self._apply_one(ac, config, container)
            applied.append(ac)
        return applied

    def _apply_one(self, ac: type, config: FireflyDataScienceConfig, container: Container) -> None:
        if not (is_auto_configuration(ac) or is_configuration(ac)):
            logger.debug("%s is neither @auto_configuration nor @configuration; skipping", ac.__name__)
            return
        instance = ac()
        # Register any @bean factory methods whose own conditions pass.
        for method, meta in get_bean_methods(instance):
            ctx = ConditionContext(config=config, container=container)
            if not _all_match(get_conditions(method), ctx):
                continue
            provided = get_type_hints(method).get("return")
            if provided is None:
                logger.debug("@bean %r on %s has no return annotation; skipping", method.__name__, ac.__name__)
                continue
            container.register_factory(
                provided,
                method,
                scope=meta.scope,
                name=meta.name or method.__name__,
                primary=meta.primary,
                order=get_order(method),
            )
        # A configuration class may itself be a @component.
        comp = get_component_meta(ac)
        if comp is not None:
            container.register_type(ac, scope=comp.scope, name=comp.name or ac.__name__, primary=comp.primary)

    # -- output -----------------------------------------------------------

    def _emit_banner(self, config: FireflyDataScienceConfig) -> None:
        if not self._print_output:
            return
        printer = BannerPrinter.from_config(config, app_name=self._app_name, app_version=self._app_version)
        text = printer.render()
        if text:
            self._console.print(text, style="bold cyan", highlight=False)

    def _emit_wiring_summary(self, config: FireflyDataScienceConfig, container: Container, applied: list[type]) -> None:
        if not self._print_output or config.banner.mode is BannerMode.OFF:
            return
        profiles = config.profiles or ["default"]
        lines = [
            "[bold cyan]Firefly DataScience[/bold cyan] is ready.",
            f"  profiles      : {', '.join(profiles)}",
            f"  beans         : {len(container)}",
            f"  auto-config   : {len(applied)} applied ({', '.join(ac.__name__ for ac in applied) or 'none'})",
            f"  ml framework  : {config.default_ml_framework}",
            f"  genai         : {'enabled' if config.genai.enabled else 'disabled'}",
            f"  sandbox       : {config.execution.sandbox}",
        ]
        self._console.print("\n".join(lines), highlight=False)


def _all_match(conditions: Sequence[Any], ctx: ConditionContext) -> bool:
    return all(condition.matches(ctx) for condition in conditions)
