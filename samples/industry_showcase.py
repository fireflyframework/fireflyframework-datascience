# Copyright 2026 Firefly Software Foundation.
"""Industry showcase — the framework on real, public finance & retail datasets (from OpenML).

No Kaggle account or credentials are needed: these load straight from OpenML over the network. Each
runs the full pipeline — load → validate → AutoML (cross-validated model selection) → holdout
evaluation — on genuine, mixed-type data with categorical features.

    uv run python samples/industry_showcase.py        # needs the [tabular] + [data] extras

To add governed GenAI feature engineering, set an LLM key (see docs/llm-configuration.md) and pass an
``AgentFeatureProposer`` to ``GenAIFeatureEngineer`` as in ``samples/genai_llm_showcase.py``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.datasets.adapters import OpenMLDatasetLoader
from fireflyframework_datascience.validation.adapters import BasicValidator

# Public OpenML datasets (id, human label, a sensible row cap to keep the demo fast).
FINANCE = (31, "credit-g · German credit risk", None)
RETAIL = (1461, "bank-marketing · campaign conversion", 6000)


def _load(source_id: int, max_rows: int | None) -> Dataset:
    """Load an OpenML dataset, encode a non-numeric classification target, and optionally subsample."""
    from sklearn.preprocessing import LabelEncoder

    dataset = OpenMLDatasetLoader().load(f"openml:{source_id}")
    y = dataset.y
    if dataset.task.is_classification() and not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(LabelEncoder().fit_transform(y), name=dataset.target_name)
    X = dataset.X
    if max_rows and len(X) > max_rows:
        idx = X.sample(n=max_rows, random_state=0).index
        X, y = X.loc[idx].reset_index(drop=True), pd.Series(y).loc[idx].reset_index(drop=True)
    return Dataset(
        dataset.name, X, y, task=dataset.task, target_name=dataset.target_name, feature_names=list(X.columns)
    )


def run_case(source_id: int, label: str, max_rows: int | None) -> dict[str, Any]:
    """Run validate → AutoML → evaluate on one dataset and return a structured report."""
    dataset = _load(source_id, max_rows)
    validation = BasicValidator().validate(dataset.X, dataset.y)
    train, test = dataset.train_test_split(test_size=0.25, random_state=0)
    result = AutoML(cv=4).fit(train)
    evaluation = result.evaluate(test)
    return {
        "label": label,
        "rows": dataset.n_rows,
        "features": dataset.n_features,
        "validation_ok": validation.ok,
        "winner": result.best_model.name,
        "metric": result.metric,
        "holdout": round(evaluation.primary_value, 4),
        "leaderboard": result.leaderboard_table(),
    }


def main() -> None:
    for source_id, label, cap in (FINANCE, RETAIL):
        print(f"\n=== {label}  (OpenML {source_id}) ===")
        report = run_case(source_id, label, cap)
        print(f"  rows={report['rows']}  features={report['features']}  validation_ok={report['validation_ok']}")
        print(f"  winner: {report['winner']}   {report['metric']} (holdout) = {report['holdout']}")
        print("  leaderboard:")
        for line in report["leaderboard"].splitlines():
            print(f"    {line}")


if __name__ == "__main__":
    main()
