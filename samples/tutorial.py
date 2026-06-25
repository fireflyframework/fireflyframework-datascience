# Copyright 2026 Firefly Software Foundation.
"""Firefly DataScience — the end-to-end tutorial.

A single, runnable tour of the whole framework on a realistic (synthetic) credit-risk dataset. It runs
**offline with no LLM key** (the GenAI steps use deterministic stand-in proposers) and prints, at the
end, exactly how to switch on a real LLM.

Run it:
    uv run python samples/tutorial.py            # needs the `tabular` extra

Every step is covered by ``tests/samples/test_tutorial.py`` — the tutorial is guaranteed to work.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fireflyframework_datascience import FireflyDataScienceApplication
from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.engineering import SequenceProposer, SolutionCandidate
from fireflyframework_datascience.engineering.loop import AgenticAutoML
from fireflyframework_datascience.features import FeatureProposal, StaticFeatureProposer
from fireflyframework_datascience.features.genai import GenAIFeatureEngineer
from fireflyframework_datascience.serving import LocalModelServer
from fireflyframework_datascience.validation.adapters import BasicValidator


def make_credit_dataset(n: int = 800, seed: int = 11) -> Dataset:
    """A synthetic credit-risk dataset whose default risk is driven by *debt-to-income* — a ratio that
    is deliberately NOT given to the model, so feature engineering has something real to discover."""
    rng = np.random.RandomState(seed)
    income = rng.normal(60_000, 18_000, n).clip(15_000, None)
    loan_amount = rng.normal(18_000, 10_000, n).clip(1_000, None)
    employment_years = rng.uniform(0, 30, n).round(1)
    num_prior_defaults = rng.poisson(0.6, n)
    dti = loan_amount / income
    logit = -2.6 + 5.0 * dti + 1.3 * num_prior_defaults - 0.05 * employment_years + rng.normal(0, 0.25, n)
    default = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    X = pd.DataFrame(
        {
            "income": income.round(2),
            "loan_amount": loan_amount.round(2),
            "employment_years": employment_years,
            "num_prior_defaults": num_prior_defaults,
        }
    )
    return Dataset(
        "credit_applicants",
        X,
        pd.Series(default, name="default"),
        task=TaskType.BINARY,
        target_name="default",
        feature_names=list(X.columns),
    )


def _logistic_scorer(task: TaskType) -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000)


def step_1_boot() -> dict[str, int]:
    """Boot the application: banner, config, dependency injection, auto-configuration."""
    app = FireflyDataScienceApplication.run(print_output=False)
    return {"beans": app.bean_count, "auto_configs": len(app.applied_auto_configurations)}


def step_2_load_and_validate() -> tuple[Dataset, Any]:
    """Build the dataset and sanity-check it before training."""
    dataset = make_credit_dataset()
    return dataset, BasicValidator().validate(dataset.X, dataset.y)


def step_3_classical_automl(train: Dataset) -> Any:
    """Run classical AutoML — cross-validate candidate models and pick the winner."""
    return AutoML(cv=4).fit(train)


def step_4_genai_feature_engineering(train: Dataset) -> Any:
    """GenAI feature engineering, offline. With a real LLM you would use ``AgentFeatureProposer``; here a
    deterministic proposer stands in so the tutorial runs without a key. The cost/benefit gate keeps a
    feature only if it measurably lifts the score — ``debt_to_income`` (the hidden driver) is accepted,
    the constant ``noise`` feature is rejected."""
    proposer = StaticFeatureProposer(
        [
            FeatureProposal("debt_to_income", "df['debt_to_income'] = df['loan_amount'] / (df['income'] + 1)", "DTI"),
            FeatureProposal("noise", "df['noise'] = 0.0", "should be rejected"),
        ]
    )
    return GenAIFeatureEngineer(proposer, scorer_estimator=_logistic_scorer, cv=4).engineer(train)


def step_5_agentic_loop(train: Dataset) -> Any:
    """The agentic ML-engineering loop, offline. With a real LLM you would use ``AgentSolutionProposer``;
    here a fixed candidate sequence stands in. Each candidate is trained, cross-validated, and verified
    (it must beat a trivial baseline) before selection."""
    proposer = SequenceProposer(
        [SolutionCandidate("linear"), SolutionCandidate("random_forest"), SolutionCandidate("hist_gradient_boosting")]
    )
    return AgenticAutoML(proposer, cv=3, max_iterations=4).solve(train)


def step_6_serve(model: Any, sample_x: Any) -> Any:
    """Serve the winning model in-process and score a sample applicant."""
    server = LocalModelServer()
    server.load(model)
    return server.predict(sample_x)


def run() -> dict[str, Any]:
    """Run the whole tutorial and return a structured report (used by the test)."""
    boot = step_1_boot()
    dataset, validation = step_2_load_and_validate()
    train, test = dataset.train_test_split(test_size=0.25, random_state=0)

    automl = step_3_classical_automl(train)
    automl_eval = automl.evaluate(test)
    engineered = step_4_genai_feature_engineering(train)
    agentic = step_5_agentic_loop(train)
    prediction = step_6_serve(automl.best_model, test.X.iloc[[0]])

    return {
        "boot": boot,
        "validation_ok": validation.ok,
        "automl_winner": automl.best_model.name,
        "automl_roc_auc": automl_eval.metrics["roc_auc"],
        "leaderboard": automl.leaderboard_table(),
        "fe_accepted": [a.proposal.name for a in engineered.accepted],
        "fe_rejected": [r.proposal.name for r in engineered.rejected],
        "fe_lift": engineered.lift,
        "agentic_best": agentic.best_candidate.trainer if agentic.best_candidate else None,
        "agentic_verified": len(agentic.valid_attempts),
        "sample_prediction": int(prediction[0]),
    }


def main() -> None:
    print("=" * 72)
    print("  Firefly DataScience — end-to-end tutorial (credit-risk)")
    print("=" * 72)
    report = run()
    print(f"\n[1] App booted: {report['boot']['beans']} beans, {report['boot']['auto_configs']} auto-configurations")
    print(f"[2] Data validated: ok={report['validation_ok']}")
    print(f"[3] Classical AutoML winner: {report['automl_winner']} (holdout roc_auc={report['automl_roc_auc']:.4f})")
    print("    leaderboard:")
    for line in report["leaderboard"].splitlines():
        print(f"      {line}")
    print(
        f"[4] GenAI features: accepted={report['fe_accepted']} rejected={report['fe_rejected']} (lift {report['fe_lift']:+.4f})"
    )
    print(f"[5] Agentic loop best: {report['agentic_best']} ({report['agentic_verified']} verified candidates)")
    print(f"[6] Served prediction for one applicant: default={report['sample_prediction']}")
    print("\n" + "-" * 72)
    print("  Turn on a REAL LLM (GenAI feature engineering + the agentic loop):")
    print("-" * 72)
    print(
        "  export OPENAI_API_KEY=sk-...                      # or ANTHROPIC_API_KEY=...\n"
        "  export FIREFLY_DATASCIENCE_GENAI__ENABLED=true\n"
        "  export FIREFLY_DATASCIENCE_GENAI__DEFAULT_MODEL=openai:gpt-4o    # or anthropic:claude-sonnet-4-5\n"
        "\n"
        "  Then use the agent-backed proposers instead of the stand-ins:\n"
        "    from fireflyframework_datascience.features.genai import AgentFeatureProposer\n"
        "    from fireflyframework_datascience.engineering.loop import AgentSolutionProposer\n"
        "\n"
        "  Full guide: docs/llm-configuration.md"
    )


if __name__ == "__main__":
    main()
