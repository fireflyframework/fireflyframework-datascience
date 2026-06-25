# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for the lineage module."""

from __future__ import annotations

from fireflyframework_datascience.container.conditions import auto_configuration, conditional_on_missing_bean
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.lineage import LineagePort


@auto_configuration
@configuration
class LineageAutoConfiguration:
    """Registers the no-op lineage backend by default."""

    @bean(name="noop_lineage", primary=True)
    @conditional_on_missing_bean(LineagePort)
    def noop_lineage(self) -> LineagePort:
        from fireflyframework_datascience.lineage import NoOpLineage

        return NoOpLineage()
