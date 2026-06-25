# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for the validation module."""

from __future__ import annotations

from fireflyframework_datascience.container.conditions import (
    auto_configuration,
    conditional_on_class,
    conditional_on_missing_bean,
)
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.validation import ValidatorPort


@auto_configuration
@conditional_on_class("pandas")
@configuration
class ValidationAutoConfiguration:
    """Registers the basic validator (default); Pandera is available as an opt-in adapter."""

    @bean(name="basic_validator", primary=True)
    @conditional_on_missing_bean(ValidatorPort)
    def basic_validator(self) -> ValidatorPort:
        from fireflyframework_datascience.validation.adapters import BasicValidator

        return BasicValidator()
