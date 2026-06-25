# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for the agentic ML-engineering loop (opt-in via ``genai.enabled``)."""

from __future__ import annotations

from fireflyframework_datascience.container.conditions import (
    auto_configuration,
    conditional_on_class,
    conditional_on_property,
)
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.core.config import FireflyDataScienceConfig
from fireflyframework_datascience.engineering import AgenticLoopPort


@auto_configuration
@conditional_on_class("sklearn")
@conditional_on_property("genai.enabled", having_value=True)
@configuration
class EngineeringAutoConfiguration:
    """Registers the agentic AutoML loop when GenAI is enabled."""

    @bean(name="agentic_automl", primary=True)
    def agentic_automl(self, config: FireflyDataScienceConfig) -> AgenticLoopPort:
        from fireflyframework_datascience.engineering.loop import AgenticAutoML, AgentSolutionProposer

        proposer = AgentSolutionProposer(model=config.genai.default_model)
        return AgenticAutoML(proposer)
