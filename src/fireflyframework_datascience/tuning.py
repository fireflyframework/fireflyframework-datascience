# Copyright 2026 Firefly Software Foundation.
"""Declarative hyperparameter search-space primitives.

Trainers describe their search space with these specs; a :class:`SearchPolicyPort` interprets them
(e.g. Optuna ``trial.suggest_*``). This decouples trainers from any particular optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ParamSpace = dict[str, "ParamSpec"]


@dataclass(frozen=True)
class IntParam:
    """Integer hyperparameter in ``[low, high]`` (optionally log-scaled)."""

    low: int
    high: int
    step: int = 1
    log: bool = False


@dataclass(frozen=True)
class FloatParam:
    """Float hyperparameter in ``[low, high]`` (optionally log-scaled)."""

    low: float
    high: float
    log: bool = False


@dataclass(frozen=True)
class CategoricalParam:
    """Categorical hyperparameter over a fixed set of choices."""

    choices: tuple[Any, ...]


ParamSpec = IntParam | FloatParam | CategoricalParam
