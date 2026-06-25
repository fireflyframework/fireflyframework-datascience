# Copyright 2026 Firefly Software Foundation.
"""GenAI showcase — run the framework with a REAL LLM (Claude / GPT / …).

Unlike the other samples (which use deterministic stand-ins so they run offline), this one calls a real
model for both GenAI feature engineering and the agentic ML-engineering loop. It reads the model and
credentials from the environment — nothing is hard-coded:

    export ANTHROPIC_API_KEY=sk-ant-...                                   # or OPENAI_API_KEY=...
    export FIREFLY_DATASCIENCE_GENAI__DEFAULT_MODEL=anthropic:claude-haiku-4-5   # optional; this is the default
    uv run python samples/genai_llm_showcase.py                          # needs the [tabular] + [genai] extras

See docs/llm-configuration.md for providers, model strings and keys.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

DEFAULT_MODEL = os.getenv("FIREFLY_DATASCIENCE_GENAI__DEFAULT_MODEL", "anthropic:claude-haiku-4-5")


def _credit_dataset(n: int = 800, seed: int = 11) -> Dataset:
    """A credit-risk dataset whose risk is driven by debt-to-income — a ratio withheld from the model."""
    rng = np.random.RandomState(seed)
    income = rng.normal(60_000, 18_000, n).clip(15_000, None)
    loan = rng.normal(18_000, 10_000, n).clip(1_000, None)
    emp = rng.uniform(0, 30, n).round(1)
    prior = rng.poisson(0.6, n)
    logit = -2.6 + 5.0 * (loan / income) + 1.3 * prior - 0.05 * emp + rng.normal(0, 0.25, n)
    y = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    X = pd.DataFrame(
        {"income": income.round(2), "loan_amount": loan.round(2), "employment_years": emp, "num_prior_defaults": prior}
    )
    return Dataset(
        "credit_applicants",
        X,
        pd.Series(y, name="default"),
        task=TaskType.BINARY,
        target_name="default",
        feature_names=list(X.columns),
    )


def genai_feature_engineering(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """The LLM proposes feature code; the gate keeps only what measurably lifts the score."""
    from sklearn.linear_model import LogisticRegression

    from fireflyframework_datascience.features.genai import AgentFeatureProposer, GenAIFeatureEngineer

    train, _ = _credit_dataset().train_test_split(random_state=0)
    engineer = GenAIFeatureEngineer(
        AgentFeatureProposer(model=model),
        scorer_estimator=lambda _t: LogisticRegression(max_iter=1000),
        cv=4,
        max_features=6,
    )
    result = engineer.engineer(train)
    return {
        "accepted": [(a.proposal.name, round(a.gain, 4), a.proposal.code) for a in result.accepted],
        "rejected": [(r.proposal.name, r.proposal.code) for r in result.rejected],
        "summary": result.summary(),
    }


def agentic_loop(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """The LLM reflects on the attempt history to propose the next model/hyperparameters."""
    from fireflyframework_datascience.engineering.loop import AgenticAutoML, AgentSolutionProposer

    train, test = SklearnDatasetLoader().load("breast_cancer").train_test_split(random_state=0)
    run = AgenticAutoML(AgentSolutionProposer(model=model), cv=3, max_iterations=3).solve(train)
    return {
        "attempts": [(a.candidate.trainer, dict(a.candidate.params), round(a.score, 4)) for a in run.attempts],
        "best": (run.best_candidate.trainer if run.best_candidate else None, round(run.best_score, 4)),
        "summary": run.summary(),
        "holdout_predictions": int(len(run.model.predict(test.X))) if run.model else 0,
    }


def main() -> None:
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("No LLM credentials found. Set ANTHROPIC_API_KEY (or OPENAI_API_KEY, …) and re-run.")
        print("See docs/llm-configuration.md.")
        return
    print(f"=== GenAI showcase · model = {DEFAULT_MODEL} ===\n")

    print("[1] GenAI feature engineering — the LLM proposes, the gate decides:")
    fe = genai_feature_engineering()
    for name, gain, code in fe["accepted"]:
        print(f"    ✓ accepted {name:24} gain={gain:+.4f}   {code}")
    for name, code in fe["rejected"]:
        print(f"    ✗ rejected {name:24} (no measured lift)   {code[:64]}")
    print(f"    → {fe['summary']}\n")

    print("[2] Agentic ML-engineering loop — the LLM reflects, the engine verifies:")
    loop = agentic_loop()
    for trainer, params, score in loop["attempts"]:
        print(f"    · {trainer:24} {params}  score={score:.4f}")
    print(f"    → {loop['summary']}")


if __name__ == "__main__":
    main()
