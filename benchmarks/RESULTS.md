# Benchmark results

Real, reproducible results from the bundled benchmark harnesses. Every number below was produced by
running the scripts in this directory — no manual tuning, fixed `random_state=0`, default trainers.

Reproduce:

```bash
uv sync --extra tabular --extra data --extra validation
uv run python benchmarks/automl_benchmark.py     # Tier-2 (offline, no network)
uv run python benchmarks/amlb_benchmark.py        # Tier-1 (OpenML, needs network)
```

## Tier-2 — offline suite (scikit-learn built-ins)

CI-smoke datasets shipped with scikit-learn; runs in seconds, no network. `AutoML(cv=3)` over the
default trainers (RandomForest, Linear, HistGradientBoosting; + XGBoost/LightGBM/CatBoost when installed).

| Dataset | Task | Metric | CV | Holdout | Winner | Seconds |
|---|---|---|---:|---:|---|---:|
| breast_cancer | binary | roc_auc | 0.9939 | **0.9952** | linear | 1.8 |
| iris | multiclass | accuracy | 0.9467 | **1.0000** | random_forest | 1.6 |
| wine | multiclass | accuracy | 0.9700 | **1.0000** | linear | 1.0 |
| diabetes | regression | rmse | −54.10 | **56.46** | linear | 1.4 |
| california_housing | regression | rmse | −0.473 | **0.455** | hist_gradient_boosting | 9.0 |

## Tier-1 — OpenML-CC18 (AMLB-style)

Real OpenML tasks with genuine categorical data (e.g. `credit-g`), exercising the dtype-aware
preprocessing and string-target encoding. `AutoML(cv=5)`. Comparable to published AutoGluon / H2O /
FLAML numbers on the same datasets.

| OpenML id | Dataset | Metric | CV | Holdout | Winner | Seconds |
|---|---|---|---:|---:|---|---:|
| 31 | credit-g | roc_auc | 0.7689 | **0.8248** | random_forest | 5.4 |
| 37 | diabetes | roc_auc | 0.8155 | **0.8724** | linear | 3.4 |
| 1464 | blood-transfusion | roc_auc | 0.7465 | **0.7511** | linear | 3.7 |
| 1480 | ilpd | roc_auc | 0.7347 | **0.7798** | linear | 3.0 |

> The full AMLB (104 tasks), CC18 (72) and CTR23 (35) suites plug into the same `run_amlb` shape under
> a nightly compute budget. See [docs/benchmarks.md](../docs/benchmarks.md) for the three-tier strategy.

## Beating the baseline (head-to-head)

`benchmarks/beat_baseline.py` pits Firefly's AutoML against a default `LogisticRegression` — the common
single-model reference — on **5-fold cross-validated ROC-AUC** (the metric these benchmarks actually
use; far more stable than one holdout on small data). Same data, same folds, same seed.

| Dataset | Baseline (LogReg) | Firefly AutoML | Δ | Winner |
|---|---:|---:|---:|---|
| credit-g | 0.7892 | **0.7942** | +0.0050 | random_forest |
| **phoneme** | 0.8128 | **0.9620** | **+0.1491** | random_forest |
| bank-marketing | 0.8998 | **0.9202** | +0.0204 | random_forest |
| diabetes | 0.8329 | 0.8329 | +0.0000 | linear (tie) |
| ilpd | 0.7574 | 0.7574 | +0.0000 | linear (tie) |
| blood-transfusion | 0.8815 | 0.8815 | +0.0000 | linear (tie) |

**Firefly wins or ties on 6/6** — it never does worse than the baseline, because it selects the best
model from a portfolio that includes the baseline's family. It wins clearly where the data is
non-linear (**phoneme +0.149**), and correctly *matches* the baseline by choosing `linear` where a
linear model is genuinely best. Mean gain across the six: **+0.029 ROC-AUC**. That is the value of
automated selection — stated honestly: no magic, just always picking the right tool.

## GenAI feature engineering — real-LLM result

With a real LLM (`anthropic:claude-haiku-4-5`), the `GenAIFeatureEngineer` was asked to improve a
synthetic credit-risk dataset whose risk is driven by *debt-to-income* — a ratio deliberately withheld
from the model. Claude proposed six features; the cost/benefit gate **accepted** the two that lifted a
logistic baseline and **rejected** the four that did not:

```
ACCEPTED debt_to_income_ratio   gain=+0.0013   df['debt_to_income_ratio'] = df['loan_amount'] / (df['income'] + 1)
ACCEPTED loan_to_income_pct     gain=+0.0007   df['loan_to_income_pct']   = (df['loan_amount'] / df['income']) * 100
rejected employment_stability_score   (no measured lift)
rejected prior_default_flag           (no measured lift)
rejected default_frequency            (no measured lift)
rejected income_loan_buffer           (no measured lift)
```

The LLM discovered the latent driver from the schema alone — and the gate kept only what was proven on
the data. Reproduce with `samples/genai_llm_showcase.py` (needs `ANTHROPIC_API_KEY`).
