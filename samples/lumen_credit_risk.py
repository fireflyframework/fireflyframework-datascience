# Copyright 2026 Firefly Software Foundation.
"""Lumen Lending — credit-risk AutoML sample.

Demonstrates the full Firefly DataScience stack on a realistic (synthetic) lending dataset:

  1. GenAI feature engineering discovers a domain feature (debt-to-income) and the cost/benefit gate
     keeps it only because it measurably lifts a baseline — no LLM key needed (StaticFeatureProposer).
  2. Classical AutoML selects the best model across trainers via cross-validation.
  3. The winner is served in-process and used to score a new applicant.

Run it:  ``python samples/lumen_credit_risk.py``   (needs the ``tabular`` extra)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.features import FeatureProposal, StaticFeatureProposer
from fireflyframework_datascience.features.genai import GenAIFeatureEngineer
from fireflyframework_datascience.serving import LocalModelServer


def make_lending_dataset(n: int = 800, seed: int = 7) -> Dataset:
    """Synthesize a lending dataset where default risk is driven mostly by debt-to-income."""
    rng = np.random.RandomState(seed)
    income = rng.normal(60_000, 20_000, n).clip(15_000, None)
    loan_amount = rng.normal(20_000, 12_000, n).clip(1_000, None)
    age = rng.randint(21, 70, n)
    employment_years = rng.uniform(0, 30, n).round(1)
    credit_history_length = rng.uniform(0, 25, n).round(1)
    num_prior_defaults = rng.poisson(0.4, n)

    dti = loan_amount / income  # the latent driver — NOT given to the model directly
    logit = (
        -3.0
        + 4.0 * dti
        + 0.8 * num_prior_defaults
        - 0.05 * employment_years
        - 0.02 * credit_history_length
        + rng.normal(0, 0.5, n)
    )
    prob_default = 1.0 / (1.0 + np.exp(-logit))
    default = (rng.uniform(0, 1, n) < prob_default).astype(int)

    X = pd.DataFrame(
        {
            "income": income.round(2),
            "loan_amount": loan_amount.round(2),
            "age": age,
            "employment_years": employment_years,
            "credit_history_length": credit_history_length,
            "num_prior_defaults": num_prior_defaults,
        }
    )
    return Dataset(
        name="lumen_credit_risk",
        X=X,
        y=pd.Series(default, name="default"),
        task=TaskType.BINARY,
        target_name="default",
        feature_names=list(X.columns),
    )


def _logistic_scorer(task: TaskType) -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000)


def run() -> dict[str, Any]:
    """Run the end-to-end Lumen credit-risk pipeline and return a report dict."""
    dataset = make_lending_dataset()
    train, test = dataset.train_test_split(test_size=0.25, random_state=0)

    # 1. GenAI feature engineering — domain features proposed as code, kept only if they lift the score.
    proposer = StaticFeatureProposer(
        [
            FeatureProposal("debt_to_income", "df['debt_to_income'] = df['loan_amount'] / (df['income'] + 1)", "DTI"),
            FeatureProposal(
                "loan_per_year_employed",
                "df['loan_per_year_employed'] = df['loan_amount'] / (df['employment_years'] + 1)",
                "burden",
            ),
            FeatureProposal("noise", "df['noise'] = 0.0", "should be rejected"),
        ]
    )
    fe = GenAIFeatureEngineer(proposer, scorer_estimator=_logistic_scorer, cv=4)
    engineered = fe.engineer(train)

    # 2. Classical AutoML on the engineered features.
    automl = AutoML(cv=4)
    result = automl.fit(engineered.dataset)
    engineered_test = test.with_features(_apply(proposer, test.X, engineered))
    evaluation = result.evaluate(engineered_test)

    # 3. Serve the winner and score one applicant.
    server = LocalModelServer()
    server.load(result.best_model)
    applicant = engineered_test.X.iloc[[0]]
    prediction = server.predict(applicant)

    return {
        "accepted_features": [a.proposal.name for a in engineered.accepted],
        "rejected_features": [r.proposal.name for r in engineered.rejected],
        "fe_lift": engineered.lift,
        "winner": result.best_model.name,
        "leaderboard": result.leaderboard_table(),
        "holdout": evaluation.metrics,
        "sample_prediction": int(prediction[0]),
    }


def _apply(proposer: StaticFeatureProposer, X: pd.DataFrame, engineered: Any) -> pd.DataFrame:
    """Apply the *accepted* feature code to the test frame (keep train/test consistent)."""
    from fireflyframework_datascience.features.executor import FeatureCodeExecutor

    executor = FeatureCodeExecutor()
    working = X.copy()
    for accepted in engineered.accepted:
        working = executor.execute(accepted.proposal.code, working)
    return working


def main() -> None:
    report = run()
    print("=== Lumen Lending — credit-risk AutoML ===")
    print(f"accepted features : {report['accepted_features']}")
    print(f"rejected features : {report['rejected_features']}")
    print(f"feature-eng lift  : {report['fe_lift']:+.4f}")
    print(f"winning model     : {report['winner']}")
    print("leaderboard:")
    print(report["leaderboard"])
    print(f"holdout metrics   : {report['holdout']}")
    print(f"applicant predicted default = {report['sample_prediction']}")


if __name__ == "__main__":
    main()
