# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for the search module."""

from __future__ import annotations

from fireflyframework_datascience.container.conditions import auto_configuration, conditional_on_class
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.search import SearchPolicyPort


@auto_configuration
@configuration
class SearchAutoConfiguration:
    """Registers the default search policy always; Optuna when installed (and made primary).

    Both are registered under ``SearchPolicyPort`` (so ``resolve_all`` sees both); Optuna is marked
    primary so a plain ``resolve`` returns it when available.
    """

    @bean(name="default_search_policy")
    def default_policy(self) -> SearchPolicyPort:
        from fireflyframework_datascience.search.adapters import DefaultSearchPolicy

        return DefaultSearchPolicy()

    @bean(name="optuna_search_policy", primary=True)
    @conditional_on_class("optuna")
    def optuna_policy(self) -> SearchPolicyPort:
        from fireflyframework_datascience.search.adapters import OptunaSearchPolicy

        return OptunaSearchPolicy()
