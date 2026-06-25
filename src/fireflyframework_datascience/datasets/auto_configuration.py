# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for the datasets module."""

from __future__ import annotations

from fireflyframework_datascience.container.conditions import auto_configuration, conditional_on_class
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.datasets import DatasetLoaderPort


@auto_configuration
@conditional_on_class("sklearn")
@configuration
class DatasetsAutoConfiguration:
    """Registers built-in dataset loaders when scikit-learn is available."""

    @bean(name="sklearn_dataset_loader")
    def sklearn_loader(self) -> DatasetLoaderPort:
        from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

        return SklearnDatasetLoader()

    @bean(name="openml_dataset_loader")
    @conditional_on_class("openml")
    def openml_loader(self) -> DatasetLoaderPort:
        from fireflyframework_datascience.datasets.adapters import OpenMLDatasetLoader

        return OpenMLDatasetLoader()
