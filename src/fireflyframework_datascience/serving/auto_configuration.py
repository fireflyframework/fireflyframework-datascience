# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for the serving module."""

from __future__ import annotations

from fireflyframework_datascience.container.conditions import auto_configuration, conditional_on_missing_bean
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.serving import ModelServerPort


@auto_configuration
@configuration
class ServingAutoConfiguration:
    """Registers the in-process model server by default."""

    @bean(name="local_model_server", primary=True)
    @conditional_on_missing_bean(ModelServerPort)
    def local_server(self) -> ModelServerPort:
        from fireflyframework_datascience.serving import LocalModelServer

        return LocalModelServer()
