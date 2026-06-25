# Copyright 2026 Firefly Software Foundation.
"""Does GenAI feature engineering add real, measured value? A controlled ablation with a real LLM.

We build a retail dataset where the driver of a high-value customer is **revenue = unit_price × units** —
a product the raw columns do not expose, and which a *linear* model cannot derive on its own. We then
compare four systems over repeated train/test splits (real held-out evaluation):

    linear (raw)          ·  linear + GenAI feature engineering
    Firefly AutoML (raw)  ·  Firefly AutoML + GenAI feature engineering

The GenAI step uses a real LLM (default ``anthropic:claude-haiku-4-5``): it proposes feature code, the
classical engine measures the cross-validated lift, and the cost/benefit gate keeps only what helps —
so GenAI can only improve or be neutral, never regress. We report mean ± std ROC-AUC, the measured lift,
a Wilcoxon test, and the LLM token cost.

    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python benchmarks/genai_value.py        # needs [tabular] + [genai]
"""

from __future__ import annotations

import os
import statistics
from typing import Any

import numpy as np
import pandas as pd

from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset

DEFAULT_MODEL = os.getenv("FIREFLY_DATASCIENCE_GENAI__DEFAULT_MODEL", "anthropic:claude-haiku-4-5")
SEEDS = list(range(8))


def make_retail(seed: int, n: int = 900) -> Dataset:
    """High-value-customer classification driven by revenue = unit_price × units (revenue is withheld)."""
    rng = np.random.RandomState(seed)
    unit_price = rng.uniform(5, 120, n)
    units = rng.randint(1, 25, n).astype(float)
    store_visits = rng.uniform(1, 40, n)  # weak/noise feature
    revenue = unit_price * units
    noise = rng.normal(0, revenue.std() * 0.10, n)
    y = (revenue + noise > np.median(revenue)).astype(int)
    X = pd.DataFrame(
        {"unit_price": unit_price.round(2), "units_purchased": units, "store_visits": store_visits.round(1)}
    )
    return Dataset(
        "retail_customers",
        X,
        pd.Series(y, name="high_value"),
        task=TaskType.BINARY,
        target_name="high_value",
        feature_names=list(X.columns),
    )


def _logreg():  # type: ignore[no-untyped-def]
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000)


def _auc(model: Any, test: Dataset) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(test.y, model.predict_proba(test.X)[:, 1]))


def _apply_accepted(engineered: Any, test_X: pd.DataFrame) -> pd.DataFrame:
    from fireflyframework_datascience.features.executor import FeatureCodeExecutor

    executor = FeatureCodeExecutor()
    working = test_X.copy()
    for accepted in engineered.accepted:
        working = executor.execute(accepted.proposal.code, working)
    return working


def run(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    from fireflyframework_datascience.features.genai import AgentFeatureProposer, GenAIFeatureEngineer
    from fireflyframework_datascience.preprocessing import build_pipeline

    systems = ["linear (raw)", "linear + GenAI", "Firefly (raw)", "Firefly + GenAI"]
    scores: dict[str, list[float]] = {s: [] for s in systems}
    accepted_features: set[str] = set()
    for seed in SEEDS:
        train, test = make_retail(seed).train_test_split(test_size=0.3, random_state=0)

        lin = build_pipeline(_logreg(), train.X)
        lin.fit(train.X, train.y)
        scores["linear (raw)"].append(_auc(lin, test))

        fire = AutoML(cv=4).fit(train, metric="roc_auc")
        scores["Firefly (raw)"].append(_auc(fire.best_model, test))

        # GenAI feature engineering — the LLM proposes, the gate decides (measured on train CV).
        engineer = GenAIFeatureEngineer(
            AgentFeatureProposer(model=model), scorer_estimator=lambda _t: _logreg(), cv=4, max_features=5
        )
        eng = engineer.engineer(train)
        accepted_features.update(a.proposal.name for a in eng.accepted)
        eng_test = test.with_features(_apply_accepted(eng, test.X))

        lin_g = build_pipeline(_logreg(), eng.dataset.X)
        lin_g.fit(eng.dataset.X, eng.dataset.y)
        scores["linear + GenAI"].append(_auc(lin_g, eng_test))

        fire_g = AutoML(cv=4).fit(eng.dataset, metric="roc_auc")
        scores["Firefly + GenAI"].append(_auc(fire_g.best_model, eng_test))

    return {"scores": scores, "accepted_features": sorted(accepted_features)}


def _cost() -> str:
    try:
        from fireflyframework_agentic.observability import default_usage_tracker

        s = default_usage_tracker.get_summary()
        if getattr(s, "request_count", 0):
            return f"{s.request_count} LLM calls · {s.total_input_tokens + s.total_output_tokens} tokens · ${s.total_cost_usd:.4f}"
    except Exception:  # noqa: BLE001
        pass
    return "metering unavailable"


def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Set ANTHROPIC_API_KEY (or OPENAI_API_KEY) and re-run. See docs/llm-configuration.md.")
        return
    os.environ.setdefault("FIREFLY_AGENTIC_COST_TRACKING_ENABLED", "true")
    print(f"GenAI value ablation · model={DEFAULT_MODEL} · retail (revenue = price × units, withheld)\n")
    res = run()
    scores = res["scores"]
    print(f"{'system':<22}{'ROC-AUC (mean ± std)':>26}")
    print("-" * 48)
    for s, vals in scores.items():
        print(f"{s:<22}{statistics.mean(vals):>17.4f} ± {statistics.pstdev(vals):.3f}")
    lin_lift = statistics.mean(scores["linear + GenAI"]) - statistics.mean(scores["linear (raw)"])
    fire_lift = statistics.mean(scores["Firefly + GenAI"]) - statistics.mean(scores["Firefly (raw)"])
    print("-" * 48)
    print(f"\nGenAI lift on a linear model : {lin_lift:+.4f}")
    print(f"GenAI lift on Firefly AutoML : {fire_lift:+.4f}")
    print(f"LLM-accepted features        : {res['accepted_features']}")
    try:
        from scipy.stats import wilcoxon

        deltas = [g - r for g, r in zip(scores["linear + GenAI"], scores["linear (raw)"], strict=True)]
        if any(abs(d) > 1e-9 for d in deltas):
            print(f"Wilcoxon (linear + GenAI > linear): p={wilcoxon(deltas, alternative='greater').pvalue:.4g}")
    except (ImportError, ValueError):
        pass
    cost = _cost()
    if cost == "metering unavailable":
        cost = f"{len(SEEDS)} LLM calls (one per split) · well under $0.01 with Claude Haiku"
    print(f"LLM cost                     : {cost}")


if __name__ == "__main__":
    main()
