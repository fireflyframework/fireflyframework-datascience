# Copyright 2026 Firefly Software Foundation.
"""Explainability module — the ``ExplainerPort`` and typed explanation results (import-light).

Explanations follow the framework's classical-first thesis: the importances are produced by
deterministic, well-understood methods (permutation importance, model-native importances, SHAP) — not
by an LLM. Heavy adapters live in :mod:`fireflyframework_datascience.explainability.adapters`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fireflyframework_datascience.datasets import Dataset
    from fireflyframework_datascience.models import Model


@dataclass
class GlobalExplanation:
    """Dataset-level feature importances for a fitted model."""

    method: str
    feature_importances: dict[str, float]
    std: dict[str, float] = field(default_factory=dict)
    baseline_score: float = float("nan")

    def top(self, k: int = 20) -> list[tuple[str, float]]:
        """The ``k`` most important features, highest first."""
        return sorted(self.feature_importances.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def to_frame(self) -> Any:
        """A tidy ``feature/importance/std`` DataFrame, sorted by importance (descending)."""
        import pandas as pd

        rows = [
            {"feature": n, "importance": v, "std": self.std.get(n, float("nan"))}
            for n, v in self.top(len(self.feature_importances))
        ]
        return pd.DataFrame(rows)


@dataclass
class LocalExplanation:
    """Per-prediction feature attributions for a single row."""

    method: str
    prediction: Any
    contributions: dict[str, float]
    base_value: float = float("nan")

    def top(self, k: int = 20) -> list[tuple[str, float]]:
        """The ``k`` features with the largest absolute contribution to this prediction."""
        return sorted(self.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]


@runtime_checkable
class ExplainerPort(Protocol):
    """Produces explanations for a fitted :class:`Model`."""

    name: str

    def supports(self, model: Model) -> bool: ...

    def explain_global(self, model: Model, dataset: Dataset) -> GlobalExplanation: ...


__all__ = ["ExplainerPort", "GlobalExplanation", "LocalExplanation"]
