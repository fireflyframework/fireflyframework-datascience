# Copyright 2026 Firefly Software Foundation.
"""OpenML-CC18 AMLB-style benchmark (Tier-1 of the benchmark strategy).

Runs the classical AutoML facade across real OpenML tasks — exercising the dtype-aware preprocessing on
genuine mixed-type data (e.g. credit-g has categoricals) — and reports apples-to-apples numbers
comparable to AutoGluon/H2O/FLAML on the same datasets. Requires the ``data`` extra (``openml``) and
network access. This runs a small, fast subset; the full AMLB (104 tasks) / CC18 (72) suites plug into
the same ``run_amlb`` shape under a nightly budget.

Run:  python benchmarks/amlb_benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.datasets.adapters import OpenMLDatasetLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from automl_benchmark import BenchmarkResult, format_table  # noqa: E402  (sibling script on sys.path)

# A small, fast OpenML-CC18 binary-classification subset (all < ~1500 rows).
CC18_SUBSET: dict[int, str] = {31: "credit-g", 37: "diabetes", 1464: "blood-transfusion", 1480: "ilpd"}


def _encode_classification_target(dataset: Dataset) -> Dataset:
    """Label-encode non-numeric classification targets (real OpenML data ships string/categorical y)."""
    if dataset.task.is_classification() and not pd.api.types.is_numeric_dtype(dataset.y):
        encoded = pd.Series(LabelEncoder().fit_transform(dataset.y), name=dataset.target_name)
        return Dataset(
            name=dataset.name,
            X=dataset.X,
            y=encoded,
            task=dataset.task,
            target_name=dataset.target_name,
            feature_names=dataset.feature_names,
            metadata=dataset.metadata,
        )
    return dataset


def run_amlb(dataset_ids: list[int] | None = None, *, cv: int = 5) -> list[BenchmarkResult]:
    """Run AutoML across the OpenML dataset ids and return a :class:`BenchmarkResult` per task."""
    loader = OpenMLDatasetLoader()
    results: list[BenchmarkResult] = []
    for data_id in dataset_ids or list(CC18_SUBSET):
        dataset = _encode_classification_target(loader.load(f"openml:{data_id}"))
        train, test = dataset.train_test_split(test_size=0.25, random_state=0)
        started = time.perf_counter()
        outcome = AutoML(cv=cv).fit(train)
        elapsed = time.perf_counter() - started
        holdout = outcome.evaluate(test)
        results.append(
            BenchmarkResult(
                dataset=f"{data_id}:{dataset.name}"[:20],
                task=dataset.task.value,
                metric=outcome.metric,
                cv_score=outcome.best_score,
                holdout_score=holdout.primary_value,
                winner=outcome.best_model.name,
                n_rows=dataset.n_rows,
                n_features=dataset.n_features,
                fit_seconds=round(elapsed, 2),
            )
        )
    return results


def main() -> None:
    print("Firefly DataScience — OpenML-CC18 AMLB-style benchmark (Tier-1 subset)\n")
    print(format_table(run_amlb()))


if __name__ == "__main__":
    main()
