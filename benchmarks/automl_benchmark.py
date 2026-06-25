# Copyright 2026 Firefly Software Foundation.
"""Offline AutoML benchmark harness (Tier-2 of the benchmark strategy).

Runs the classical AutoML facade across a suite of offline scikit-learn datasets and reports the
primary metric, winning model, and wall-clock per dataset. No network required — ideal for CI smoke
and local sanity. The Tier-1 AMLB/OpenML-CC18/CTR23 suites and Tier-3 MLE-bench/DSBench plug in via the
same ``run_suite`` shape with the ``data`` extra and a larger compute budget (see docs/benchmarks.md).

Run:  python benchmarks/automl_benchmark.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

# Tier-2 offline suite: classification + regression, all shipped with scikit-learn.
OFFLINE_SUITE = ["breast_cancer", "iris", "wine", "diabetes", "california_housing"]


@dataclass
class BenchmarkResult:
    """One dataset's benchmark outcome."""

    dataset: str
    task: str
    metric: str
    cv_score: float
    holdout_score: float
    winner: str
    n_rows: int
    n_features: int
    fit_seconds: float


def run_suite(datasets: list[str] | None = None, *, cv: int = 3, test_size: float = 0.25) -> list[BenchmarkResult]:
    """Run AutoML across ``datasets`` and return a :class:`BenchmarkResult` per dataset."""
    loader = SklearnDatasetLoader()
    results: list[BenchmarkResult] = []
    for name in datasets or OFFLINE_SUITE:
        dataset = loader.load(name)
        train, test = dataset.train_test_split(test_size=test_size, random_state=0)
        started = time.perf_counter()
        outcome = AutoML(cv=cv).fit(train)
        elapsed = time.perf_counter() - started
        holdout = outcome.evaluate(test)
        results.append(
            BenchmarkResult(
                dataset=name,
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


def format_table(results: list[BenchmarkResult]) -> str:
    """Render results as a fixed-width table."""
    header = f"{'dataset':<20}{'task':<14}{'metric':<10}{'cv':>8}{'holdout':>9}{'winner':>22}{'secs':>7}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.dataset:<20}{r.task:<14}{r.metric:<10}{r.cv_score:>8.4f}{r.holdout_score:>9.4f}{r.winner:>22}{r.fit_seconds:>7.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    print("Firefly DataScience — offline AutoML benchmark (Tier-2)\n")
    print(format_table(run_suite()))


if __name__ == "__main__":
    main()
