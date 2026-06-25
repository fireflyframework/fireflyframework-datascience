# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for the NLP module."""

from __future__ import annotations

from fireflyframework_datascience.container.conditions import auto_configuration, conditional_on_class
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.nlp import TextClassifierPort


@auto_configuration
@conditional_on_class("transformers")
@configuration
class NLPAutoConfiguration:
    """Registers the HuggingFace text classifier when the ``nlp`` extra is installed."""

    @bean(name="hf_text_classifier")
    def hf_text_classifier(self) -> TextClassifierPort:
        from fireflyframework_datascience.nlp.adapters import HFTextClassifier

        return HFTextClassifier()
