# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for the vision module."""

from __future__ import annotations

from fireflyframework_datascience.container.conditions import auto_configuration, conditional_on_class
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.vision import ImageClassifierPort


@auto_configuration
@conditional_on_class("torch")
@configuration
class VisionAutoConfiguration:
    """Registers the PyTorch CNN image classifier when torch is installed."""

    @bean(name="torch_cnn_classifier")
    def torch_cnn_classifier(self) -> ImageClassifierPort:
        from fireflyframework_datascience.vision.adapters import TorchCNNClassifier

        return TorchCNNClassifier()
