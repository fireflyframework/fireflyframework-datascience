# Copyright 2026 Firefly Software Foundation.
"""Search module — the search-policy port and result (import-light)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fireflyframework_datascience.tuning import ParamSpace

Objective = Callable[[Mapping[str, Any]], float]


@dataclass
class SearchResult:
    """The outcome of a hyperparameter search (scores are always 'greater is better')."""

    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float = float("-inf")
    n_trials: int = 0


@runtime_checkable
class SearchPolicyPort(Protocol):
    """Optimizes an objective over a declarative :data:`ParamSpace`."""

    name: str

    def optimize(
        self, objective: Objective, space: ParamSpace, *, n_trials: int = 25, seed: int = 42
    ) -> SearchResult: ...


__all__ = ["Objective", "SearchPolicyPort", "SearchResult"]
