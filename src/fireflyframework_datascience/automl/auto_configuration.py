# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for the AutoML module."""

from __future__ import annotations

from fireflyframework_datascience.automl import AutoMLBackendPort
from fireflyframework_datascience.container.conditions import auto_configuration, conditional_on_class
from fireflyframework_datascience.container.stereotypes import bean, configuration


@auto_configuration
@conditional_on_class("sklearn")
@configuration
class AutoMLAutoConfiguration:
    """Registers an AutoML backend wired from the container's trainers/evaluator/search/validator."""

    @bean(name="automl_backend", primary=True)
    def automl_backend(self) -> AutoMLBackendPort:
        # A factory-style placeholder bean: the real, DI-wired engine is built via
        # AutoML.from_context(app). This bean provides a sensible default instance.
        from fireflyframework_datascience.automl.facade import AutoML

        return AutoML()
