# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for GenAI feature engineering (opt-in via ``genai.enabled``)."""

from __future__ import annotations

from fireflyframework_datascience.container.conditions import (
    auto_configuration,
    conditional_on_class,
    conditional_on_property,
)
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.core.config import FireflyDataScienceConfig
from fireflyframework_datascience.features import FeatureEngineerPort


@auto_configuration
@conditional_on_class("sklearn")
@conditional_on_property("genai.enabled", having_value=True)
@configuration
class FeaturesAutoConfiguration:
    """Registers the GenAI feature engineer when GenAI is enabled (classical-first by default)."""

    @bean(name="genai_feature_engineer", primary=True)
    def feature_engineer(self, config: FireflyDataScienceConfig) -> FeatureEngineerPort:
        from fireflyframework_datascience.features import CostBenefitGate
        from fireflyframework_datascience.features.executor import FeatureCodeExecutor
        from fireflyframework_datascience.features.genai import AgentFeatureProposer, GenAIFeatureEngineer

        proposer = AgentFeatureProposer(model=config.genai.default_model)
        gate = CostBenefitGate(min_gain=0.0)
        # Honor the FULL declared execution config — do not silently drop a control. Timeout +
        # require_approval (fail-closed HITL) + sandbox tier (in-process 'monty'/'local' run the
        # restricted executor; 'docker'/'e2b' fail until those adapters land). With the default
        # require_approval=True and no approver wired, automated GenAI fail-closes: set
        # execution.require_approval=False (or wire an approver) to run it unattended.
        executor = FeatureCodeExecutor(
            timeout_seconds=config.execution.timeout_seconds,
            require_approval=config.execution.require_approval,
            sandbox=config.execution.sandbox,
        )
        return GenAIFeatureEngineer(proposer, gate=gate, executor=executor)
