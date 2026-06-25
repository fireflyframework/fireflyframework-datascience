# Copyright 2026 Firefly Software Foundation.
"""Ensembling — combine the top-k AutoML candidates into one stronger model.

Single-best selection leaves accuracy on the table; stacking the strongest candidates with a
cross-fit meta-learner is the standard last-mile lift in production AutoML. An :class:`EnsemblePort`
builds the combined estimator from the leaderboard's base learners.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fireflyframework_datascience.core.types import TaskType


@runtime_checkable
class EnsemblePort(Protocol):
    """Builds an (unfitted) ensemble estimator from named base learners."""

    name: str

    def supports(self, task: TaskType) -> bool: ...

    def build(self, base_estimators: list[tuple[str, Any]], task: TaskType) -> Any: ...


class StackingEnsemble:
    """Stacks base learners via scikit-learn ``Stacking{Classifier,Regressor}`` (cross-fit meta-learner)."""

    name = "stacking_ensemble"

    def __init__(self, *, cv: int = 3) -> None:
        self._cv = cv

    def supports(self, task: TaskType) -> bool:
        return True

    def build(self, base_estimators: list[tuple[str, Any]], task: TaskType) -> Any:
        if task.is_classification():
            from sklearn.ensemble import StackingClassifier
            from sklearn.linear_model import LogisticRegression

            return StackingClassifier(
                estimators=base_estimators, final_estimator=LogisticRegression(max_iter=1000), cv=self._cv
            )
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import RidgeCV

        return StackingRegressor(estimators=base_estimators, final_estimator=RidgeCV(), cv=self._cv)


__all__ = ["EnsemblePort", "StackingEnsemble"]
