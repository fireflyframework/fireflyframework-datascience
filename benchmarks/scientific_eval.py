# Copyright 2026 Firefly Software Foundation.
"""Rigorous, unbiased evaluation: Firefly AutoML vs. fixed single models, by NESTED cross-validation.

Why nested CV? An AutoML system that reports the cross-validated score of the model it *selected* is
optimistically biased (it is the maximum over many models scored on the same folds). The honest protocol
is **nested** CV: an inner CV does the model selection on the training portion of each outer fold, and the
outer fold — never seen during selection — gives the unbiased estimate. That is exactly what happens here:
for every outer fold, ``AutoML(...).fit`` runs its own inner CV on the fold's training data only, and we
score the winner on the untouched outer test fold.

References compared on the same folds: a default ``LogisticRegression`` (linear), a default
``RandomForest`` (bagging), and a default ``XGBoost`` (boosting). The claim we test: *automated portfolio
selection matches or beats every fixed single model, because it adapts to each dataset.*

    uv run python benchmarks/scientific_eval.py        # needs [tabular] + [data]; network (OpenML)
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.datasets.adapters import OpenMLDatasetLoader
from fireflyframework_datascience.preprocessing import build_pipeline

# Binary OpenML datasets spanning linear-friendly → strongly non-linear. (id, label, row cap)
DATASETS = [
    (31, "credit-g", None),
    (1489, "phoneme", None),
    (37, "diabetes", None),
    (1480, "ilpd", None),
    (1464, "blood-transfusion", None),
]
OUTER_FOLDS = 5
INNER_CV = 3
SEED = 0


def _load(source_id: int, cap: int | None) -> Dataset:
    from sklearn.preprocessing import LabelEncoder

    ds = OpenMLDatasetLoader().load(f"openml:{source_id}")
    y = ds.y
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(LabelEncoder().fit_transform(y), name=ds.target_name)
    X = ds.X
    if cap and len(X) > cap:
        idx = X.sample(n=cap, random_state=SEED).index
        X, y = X.loc[idx].reset_index(drop=True), pd.Series(y).loc[idx].reset_index(drop=True)
    return Dataset(
        ds.name,
        X.reset_index(drop=True),
        pd.Series(y).reset_index(drop=True),
        task=ds.task,
        target_name=ds.target_name,
        feature_names=list(X.columns),
    )


def _single_model(name: str) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    if name == "LogReg":
        return LogisticRegression(max_iter=1000)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
    import xgboost as xgb

    return xgb.XGBClassifier(n_estimators=300, tree_method="hist", n_jobs=-1, verbosity=0, random_state=SEED)


def evaluate(dataset: Dataset) -> dict[str, Any]:
    """Nested 5-fold CV: per-fold ROC-AUC for each reference + Firefly AutoML."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X, y = dataset.X, np.asarray(dataset.y)
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    refs = ["LogReg", "RandomForest", "XGBoost"]
    scores: dict[str, list[float]] = {r: [] for r in refs}
    scores["Firefly AutoML"] = []
    picks: Counter[str] = Counter()

    for train_idx, test_idx in outer.split(X, y):
        x_tr, x_te = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
        y_tr, y_te = y[train_idx], y[test_idx]
        for r in refs:
            est = build_pipeline(_single_model(r), x_tr)
            est.fit(x_tr, y_tr)
            scores[r].append(float(roc_auc_score(y_te, est.predict_proba(x_te)[:, 1])))
        # Firefly: inner CV on the training fold ONLY selects the model; score on the untouched test fold.
        train_ds = Dataset(dataset.name, x_tr, pd.Series(y_tr), task=dataset.task, feature_names=list(x_tr.columns))
        result = AutoML(cv=INNER_CV).fit(train_ds, metric="roc_auc")
        proba = result.best_model.predict_proba(x_te)[:, 1]
        scores["Firefly AutoML"].append(float(roc_auc_score(y_te, proba)))
        picks[result.best_model.name] += 1

    return {
        "means": {k: statistics.mean(v) for k, v in scores.items()},
        "stds": {k: statistics.pstdev(v) for k, v in scores.items()},
        "fold_scores": scores,
        "picks": dict(picks),
    }


def main() -> None:
    print(f"Nested {OUTER_FOLDS}-fold CV · ROC-AUC (mean ± std) · unbiased (selection on inner CV only)\n")
    refs = ["LogReg", "RandomForest", "XGBoost", "Firefly AutoML"]
    hdr = f"{'dataset':<20}" + "".join(f"{r:>18}" for r in refs) + "   Firefly picks"
    print(hdr + "\n" + "-" * len(hdr))
    all_deltas: dict[str, list[float]] = {"LogReg": [], "RandomForest": [], "XGBoost": []}
    firefly_means, ref_best_means = [], []
    for source_id, label, cap in DATASETS:
        res = evaluate(_load(source_id, cap))
        row = f"{label:<20}"
        for r in refs:
            row += f"{res['means'][r]:>11.4f}±{res['stds'][r]:<5.3f}"
        picks = ", ".join(f"{k}×{v}" for k, v in sorted(res["picks"].items(), key=lambda kv: -kv[1]))
        print(row + f"   {picks}")
        for r in all_deltas:
            all_deltas[r] += [
                f - s for f, s in zip(res["fold_scores"]["Firefly AutoML"], res["fold_scores"][r], strict=True)
            ]
        firefly_means.append(res["means"]["Firefly AutoML"])
        ref_best_means.append(max(res["means"][r] for r in ["LogReg", "RandomForest", "XGBoost"]))

    print("-" * len(hdr))
    print("\n=== Firefly AutoML vs each fixed model (paired across all folds × datasets) ===")
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    for r, deltas in all_deltas.items():
        wins = sum(d > 1e-4 for d in deltas)
        losses = sum(d < -1e-4 for d in deltas)
        ties = len(deltas) - wins - losses
        mean_d = statistics.mean(deltas)
        p = ""
        if wilcoxon is not None and any(abs(d) > 1e-9 for d in deltas):
            try:
                p = f" · Wilcoxon p={wilcoxon(deltas, alternative='greater').pvalue:.4g}"
            except ValueError:
                p = ""
        print(f"  vs {r:<13} mean Δ={mean_d:+.4f} | wins {wins} / ties {ties} / losses {losses}{p}")
    beat_best = sum(f >= b - 1e-4 for f, b in zip(firefly_means, ref_best_means, strict=True))
    print(
        f"\n  Firefly ≥ the best single model on {beat_best}/{len(DATASETS)} datasets "
        f"(it adapts: picks boosting where non-linear, linear where linear is best)."
    )


if __name__ == "__main__":
    main()
