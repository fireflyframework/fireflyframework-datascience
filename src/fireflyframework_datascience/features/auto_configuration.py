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
        from fireflyframework_datascience.features.genai import AgentFeatureProposer, GenAIFeatureEngineer

        proposer = AgentFeatureProposer(model=config.genai.default_model)
        gate = CostBenefitGate(min_gain=0.0)
        return GenAIFeatureEngineer(proposer, gate=gate)
