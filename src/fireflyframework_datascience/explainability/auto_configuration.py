# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for the explainability module.

Registers a default :class:`ExplainerPort`. When the optional ``explain`` extra (``shap``) is
installed, the SHAP explainer is registered as primary; otherwise the dependency-free
permutation-importance explainer is used.
"""

from __future__ import annotations

import importlib.util

from fireflyframework_datascience.container.conditions import (
    auto_configuration,
    conditional_on_class,
    conditional_on_missing_bean,
)
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.explainability import ExplainerPort


@auto_configuration
@conditional_on_class("sklearn")
@configuration
class ExplainabilityAutoConfiguration:
    """Registers a single default explainer: SHAP when the ``explain`` extra is installed, else the
    dependency-free permutation-importance explainer. A user-supplied ``ExplainerPort`` wins."""

    @bean(name="default_explainer", primary=True)
    @conditional_on_missing_bean(ExplainerPort)
    def explainer(self) -> ExplainerPort:
        if importlib.util.find_spec("shap") is not None:
            from fireflyframework_datascience.explainability.adapters import ShapExplainer

            try:
                return ShapExplainer()
            except Exception:  # noqa: BLE001 - fall back to the always-available default
                pass
        from fireflyframework_datascience.explainability.adapters import PermutationImportanceExplainer

        return PermutationImportanceExplainer()
