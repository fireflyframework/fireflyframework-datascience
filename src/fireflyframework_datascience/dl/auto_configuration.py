# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for the DL / TabFM module."""

from __future__ import annotations

from fireflyframework_datascience.container.conditions import auto_configuration, conditional_on_class
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.dl import DLTrainerPort, TabFMPort


@auto_configuration
@conditional_on_class("sklearn")
@configuration
class DLAutoConfiguration:
    """Registers the sklearn-MLP DL trainer always; TabPFN when the ``tabfm`` extra is installed."""

    @bean(name="mlp_trainer")
    def mlp_trainer(self) -> DLTrainerPort:
        from fireflyframework_datascience.dl.adapters import MLPTrainer

        return MLPTrainer()

    @bean(name="tabpfn_predictor")
    @conditional_on_class("tabpfn")
    def tabpfn_predictor(self) -> TabFMPort:
        from fireflyframework_datascience.dl.adapters import TabPFNPredictor

        return TabPFNPredictor()
