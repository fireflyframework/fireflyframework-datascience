# Copyright 2026 Firefly Software Foundation.
"""Tests for search policies."""

from __future__ import annotations

from fireflyframework_datascience.search.adapters import DefaultSearchPolicy, OptunaSearchPolicy
from fireflyframework_datascience.tuning import IntParam


def test_default_policy_evaluates_defaults() -> None:
    result = DefaultSearchPolicy().optimize(lambda params: 1.0, {}, n_trials=5)
    assert result.n_trials == 1
    assert result.best_params == {}
    assert result.best_score == 1.0


def test_optuna_maximizes() -> None:
    space = {"x": IntParam(0, 10)}
    result = OptunaSearchPolicy().optimize(lambda params: -((params["x"] - 7) ** 2), space, n_trials=25, seed=1)
    assert abs(result.best_params["x"] - 7) <= 2
    assert result.n_trials == 25


def test_optuna_empty_space_runs_once() -> None:
    result = OptunaSearchPolicy().optimize(lambda params: 0.5, {}, n_trials=10)
    assert result.n_trials == 1
    assert result.best_score == 0.5
