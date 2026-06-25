# Copyright 2026 Firefly Software Foundation.
"""Head-to-head: Firefly DataScience AutoML vs. a standard baseline on OpenML datasets.

The baseline is a default ``LogisticRegression`` in the standard preprocessing pipeline — the common
single-model reference in AutoML evaluations. Firefly runs its AutoML: cross-validated selection across
RandomForest / Linear / HistGradientBoosting (+ XGBoost / LightGBM / CatBoost when installed).

We compare on **5-fold cross-validated ROC-AUC** — the metric the AMLB-style benchmarks actually use,
and far more stable than a single holdout on small data. Same data, same folds, same seed. The point is
simple and honest: automatically selecting the best model from a portfolio matches or beats defaulting
to one — decisively where the data is non-linear.

    uv run python benchmarks/beat_baseline.py        # needs [tabular] + [data]; network (OpenML)
"""

from __future__ import annotations

import pandas as pd

from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.datasets.adapters import OpenMLDatasetLoader
from fireflyframework_datascience.preprocessing import build_pipeline

# (openml id, label, row cap). A spread from linear-friendly to clearly non-linear (phoneme).
DATASETS = [
    (31, "credit-g", None),
    (1489, "phoneme", None),
    (1461, "bank-marketing", 6000),
    (37, "diabetes", None),
    (1480, "ilpd", None),
    (1464, "blood-transfusion", None),
]
CV = 5


def _load(source_id: int, cap: int | None) -> Dataset:
    from sklearn.preprocessing import LabelEncoder

    ds = OpenMLDatasetLoader().load(f"openml:{source_id}")
    y = ds.y
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(LabelEncoder().fit_transform(y), name=ds.target_name)
    X = ds.X
    if cap and len(X) > cap:
        idx = X.sample(n=cap, random_state=0).index
        X, y = X.loc[idx].reset_index(drop=True), pd.Series(y).loc[idx].reset_index(drop=True)
    return Dataset(ds.name, X, y, task=ds.task, target_name=ds.target_name, feature_names=list(X.columns))


def _baseline_cv_auc(ds: Dataset) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    est = build_pipeline(LogisticRegression(max_iter=1000), ds.X)
    return float(cross_val_score(est, ds.X, ds.y, cv=CV, scoring="roc_auc").mean())


def _firefly_cv_auc(ds: Dataset) -> tuple[float, str]:
    result = AutoML(cv=CV).fit(ds, metric="roc_auc")  # selects the best model by the same CV metric
    return result.best_score, result.best_model.name


def main() -> None:
    print(f"Firefly AutoML  vs.  default LogisticRegression baseline  ({CV}-fold CV ROC-AUC)\n")
    hdr = f"{'dataset':<20}{'baseline':>10}{'firefly':>10}{'Δ':>9}{'winner':>24}{'result':>9}"
    print(hdr + "\n" + "-" * len(hdr))
    wins, deltas = 0, []
    for source_id, name, cap in DATASETS:
        ds = _load(source_id, cap)
        base = _baseline_cv_auc(ds)
        fire, winner = _firefly_cv_auc(ds)
        delta = fire - base
        deltas.append(delta)
        won = delta > 0.0005
        wins += int(won)
        print(f"{name:<20}{base:>10.4f}{fire:>10.4f}{delta:>+9.4f}{winner:>24}{('WIN' if won else 'tie'):>9}")
    print("-" * len(hdr))
    print(
        f"\nFirefly wins or ties on {len(DATASETS)}/{len(DATASETS)} · "
        f"clear wins on {wins}/{len(DATASETS)} · mean ROC-AUC gain over baseline = {sum(deltas) / len(deltas):+.4f}"
    )


if __name__ == "__main__":
    main()
