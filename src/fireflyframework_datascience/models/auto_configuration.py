# Copyright 2026 Firefly Software Foundation.
"""Auto-configuration for the models module — registers trainers for installed libraries."""

from __future__ import annotations

from fireflyframework_datascience.container.conditions import auto_configuration, conditional_on_class
from fireflyframework_datascience.container.stereotypes import bean, configuration
from fireflyframework_datascience.models import TrainerPort


@auto_configuration
@conditional_on_class("sklearn")
@configuration
class ModelsAutoConfiguration:
    """Registers scikit-learn trainers always, boosting trainers when their library is present."""

    @bean(name="random_forest_trainer")
    def random_forest(self) -> TrainerPort:
        from fireflyframework_datascience.models.adapters import RandomForestTrainer

        return RandomForestTrainer()

    @bean(name="linear_trainer")
    def linear(self) -> TrainerPort:
        from fireflyframework_datascience.models.adapters import LinearTrainer

        return LinearTrainer()

    @bean(name="hist_gradient_boosting_trainer")
    def hist_gradient_boosting(self) -> TrainerPort:
        from fireflyframework_datascience.models.adapters import HistGradientBoostingTrainer

        return HistGradientBoostingTrainer()

    @bean(name="xgboost_trainer")
    @conditional_on_class("xgboost")
    def xgboost(self) -> TrainerPort:
        from fireflyframework_datascience.models.adapters import XGBoostTrainer

        return XGBoostTrainer()

    @bean(name="lightgbm_trainer")
    @conditional_on_class("lightgbm")
    def lightgbm(self) -> TrainerPort:
        from fireflyframework_datascience.models.adapters import LightGBMTrainer

        return LightGBMTrainer()

    @bean(name="catboost_trainer")
    @conditional_on_class("catboost")
    def catboost(self) -> TrainerPort:
        from fireflyframework_datascience.models.adapters import CatBoostTrainer

        return CatBoostTrainer()
