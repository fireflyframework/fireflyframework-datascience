# Copyright 2026 Firefly Software Foundation.
"""Explainability adapters.

- :class:`PermutationImportanceExplainer` — the dependency-free default (scikit-learn, already in the
  ``tabular`` extra). Model-agnostic: permutes each input feature and measures the score drop.
- :class:`ShapExplainer` — optional, behind the ``explain`` extra; adds local (per-prediction)
  attributions. Raises :class:`AdapterUnavailableError` if ``shap`` is not installed.
"""

from __future__ import annotations

from typing import Any

from fireflyframework_datascience.core.exceptions import AdapterUnavailableError
from fireflyframework_datascience.explainability import GlobalExplanation, LocalExplanation
from fireflyframework_datascience.models import Model


class PermutationImportanceExplainer:
    """Global feature importance via scikit-learn permutation importance (model-agnostic)."""

    name = "permutation_importance"

    def __init__(self, *, n_repeats: int = 10, random_state: int = 42, scoring: str | None = None) -> None:
        self._n_repeats = n_repeats
        self._random_state = random_state
        self._scoring = scoring

    def supports(self, model: Model) -> bool:
        return hasattr(model.estimator, "predict")

    def explain_global(self, model: Model, dataset: Any) -> GlobalExplanation:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model.estimator,
            dataset.X,
            dataset.y,
            n_repeats=self._n_repeats,
            random_state=self._random_state,
            scoring=self._scoring,
        )
        names = list(dataset.feature_names) or list(dataset.X.columns)
        importances = {n: float(m) for n, m in zip(names, result.importances_mean, strict=False)}
        std = {n: float(s) for n, s in zip(names, result.importances_std, strict=False)}
        try:
            baseline = float(model.estimator.score(dataset.X, dataset.y))
        except Exception:  # noqa: BLE001 - baseline is informational only
            baseline = float("nan")
        return GlobalExplanation(
            method="permutation_importance", feature_importances=importances, std=std, baseline_score=baseline
        )


class ShapExplainer:
    """SHAP-based global + local attributions (optional ``explain`` extra)."""

    name = "shap"

    def __init__(self, *, max_samples: int = 200) -> None:
        try:
            import shap  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise AdapterUnavailableError(
                "ShapExplainer requires the 'explain' extra: pip install 'fireflyframework-datascience[explain]'"
            ) from exc
        self._max_samples = max_samples

    def supports(self, model: Model) -> bool:
        return hasattr(model.estimator, "predict")

    def _underlying(self, model: Model) -> Any:
        """Reach the final estimator inside a sklearn Pipeline, if present."""
        est = model.estimator
        steps = getattr(est, "named_steps", None)
        return steps["model"] if steps and "model" in steps else est

    def explain_global(self, model: Model, dataset: Any) -> GlobalExplanation:
        import numpy as np

        names = list(dataset.feature_names) or list(dataset.X.columns)
        values = self._shap_values(model, dataset.X)
        mean_abs = np.abs(values).mean(axis=0)
        importances = {n: float(v) for n, v in zip(names, mean_abs, strict=False)}
        return GlobalExplanation(method="shap", feature_importances=importances)

    def explain_local(self, model: Model, X: Any) -> list[LocalExplanation]:
        names = list(getattr(X, "columns", []))
        values = self._shap_values(model, X)
        preds = model.predict(X)
        out: list[LocalExplanation] = []
        for i in range(len(values)):
            contributions = {n: float(v) for n, v in zip(names, values[i], strict=False)}
            out.append(LocalExplanation(method="shap", prediction=preds[i], contributions=contributions))
        return out

    def _shap_values(self, model: Model, X: Any) -> Any:
        import numpy as np
        import shap

        sample = X.iloc[: self._max_samples] if hasattr(X, "iloc") else X[: self._max_samples]
        # Transform through the pipeline's preprocessing so SHAP sees the estimator's real inputs is
        # complex with one-hot columns; for the model-agnostic path we explain the whole pipeline.
        explainer = shap.Explainer(model.estimator.predict, sample)
        values = explainer(sample).values
        # binary/regression -> (n, f); some explainers return (n, f, classes): collapse to class-1/abs.
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        return values


__all__ = ["PermutationImportanceExplainer", "ShapExplainer"]
