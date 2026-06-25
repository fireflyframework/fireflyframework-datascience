# Copyright 2026 Firefly Software Foundation.
"""Probability calibration — wrap a fitted classifier so its predicted probabilities are trustworthy.

Tree and boosting models often produce over-confident probabilities; risk- and cost-sensitive
decisions (lending, medicine) need them well-calibrated. A :class:`CalibratorPort` post-processes the
selected model; the default adapter uses scikit-learn's cross-validated calibration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fireflyframework_datascience.core.types import TaskType


@runtime_checkable
class CalibratorPort(Protocol):
    """Calibrates a classifier's probabilities, returning a fitted, calibrated estimator."""

    name: str

    def supports(self, task: TaskType) -> bool: ...

    def calibrate(self, estimator: Any, X: Any, y: Any, task: TaskType) -> Any: ...


class SklearnCalibrator:
    """Cross-validated calibration via scikit-learn ``CalibratedClassifierCV`` (isotonic by default)."""

    name = "sklearn_calibration"

    def __init__(self, *, method: str = "isotonic", cv: int = 3) -> None:
        self._method = method
        self._cv = cv

    def supports(self, task: TaskType) -> bool:
        return task.is_classification()

    def calibrate(self, estimator: Any, X: Any, y: Any, task: TaskType) -> Any:
        from sklearn.calibration import CalibratedClassifierCV

        calibrated = CalibratedClassifierCV(estimator, method=self._method, cv=self._cv)
        calibrated.fit(X, y)
        return calibrated


__all__ = ["CalibratorPort", "SklearnCalibrator"]
