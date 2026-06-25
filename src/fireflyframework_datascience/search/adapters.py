# Copyright 2026 Firefly Software Foundation.
"""Search-policy adapters: default (evaluate defaults) and Optuna (seeded TPE)."""

from __future__ import annotations

from typing import Any

from fireflyframework_datascience.search import Objective, SearchResult
from fireflyframework_datascience.tuning import CategoricalParam, FloatParam, IntParam, ParamSpace


class DefaultSearchPolicy:
    """Evaluates the estimator's default hyperparameters once (no tuning). Fast and deterministic."""

    name = "default"

    def optimize(self, objective: Objective, space: ParamSpace, *, n_trials: int = 25, seed: int = 42) -> SearchResult:
        score = objective({})
        return SearchResult(best_params={}, best_score=float(score), n_trials=1)


class OptunaSearchPolicy:
    """Seeded Bayesian optimization (TPE). The LLM may propose seeds/bounds; Optuna does the search.

    This is the research-backed division of labour: classical HPO owns the search, not the LLM.
    """

    name = "optuna"

    def optimize(self, objective: Objective, space: ParamSpace, *, n_trials: int = 25, seed: int = 42) -> SearchResult:
        if not space:
            score = objective({})
            return SearchResult(best_params={}, best_score=float(score), n_trials=1)

        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _trial_objective(trial: Any) -> float:
            params = {name: _suggest(trial, name, spec) for name, spec in space.items()}
            return objective(params)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(_trial_objective, n_trials=n_trials, show_progress_bar=False)
        return SearchResult(
            best_params=dict(study.best_params), best_score=float(study.best_value), n_trials=len(study.trials)
        )


def _suggest(trial: Any, name: str, spec: object) -> Any:
    if isinstance(spec, IntParam):
        if spec.log:
            return trial.suggest_int(name, spec.low, spec.high, log=True)
        return trial.suggest_int(name, spec.low, spec.high, step=spec.step)
    if isinstance(spec, FloatParam):
        return trial.suggest_float(name, spec.low, spec.high, log=spec.log)
    if isinstance(spec, CategoricalParam):
        return trial.suggest_categorical(name, list(spec.choices))
    raise TypeError(f"Unsupported param spec: {spec!r}")


__all__ = ["DefaultSearchPolicy", "OptunaSearchPolicy"]
