# Use case: Lumen Lending credit risk

**An end-to-end walkthrough of the Firefly DataScience stack — GenAI feature engineering, classical AutoML, and in-process serving — on a realistic synthetic lending dataset, with no LLM API key required.**

The `samples/lumen_credit_risk.py` sample tells one focused story: a credit-risk model where default is *secretly* driven by **debt-to-income (DTI)**, a feature the model is never handed directly. We watch the framework rediscover it from raw columns, reject a useless noise feature, let AutoML pick a winner, and score a live applicant.

The pipeline has three acts:

1. **GenAI feature engineering** proposes domain features as executable code; a cost/benefit gate keeps a feature only if it measurably lifts a cross-validated baseline.
2. **Classical AutoML** trains several models and selects the best by cross-validation.
3. **Serving** loads the winner in-process and scores a new applicant.

!!! firefly "The reproducible pattern — the LLM proposes; the classical engine decides"

    A proposer (an LLM in production, a deterministic stub here) suggests `debt_to_income`,
    `loan_per_year_employed`, and `noise`. None of them is trusted on faith: each is executed,
    cross-validated, and kept only if it clears the [`CostBenefitGate`](genai-features.md). The
    measured score decides — not the proposer.

## Run it

```bash
python samples/lumen_credit_risk.py   # needs the `tabular` extra
```

The script's `run()` function returns a report dict; `main()` prints it. Every step below maps to one block of that function.

## 1. Synthesize the lending data

`make_lending_dataset` builds a `Dataset` of raw applicant columns. Crucially, DTI (`loan_amount / income`) is the *latent* driver of default — it shapes the labels but is **not** a column the model sees.

```python
import numpy as np
import pandas as pd

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset


def make_lending_dataset(n: int = 800, seed: int = 7) -> Dataset:
    rng = np.random.RandomState(seed)
    income = rng.normal(60_000, 20_000, n).clip(15_000, None)
    loan_amount = rng.normal(20_000, 12_000, n).clip(1_000, None)
    # ... age, employment_years, credit_history_length, num_prior_defaults ...

    dti = loan_amount / income  # latent driver — NOT given to the model
    logit = -3.0 + 4.0 * dti + 0.8 * num_prior_defaults - 0.05 * employment_years  # (1)!
    prob_default = 1.0 / (1.0 + np.exp(-logit))
    default = (rng.uniform(0, 1, n) < prob_default).astype(int)

    X = pd.DataFrame({"income": income, "loan_amount": loan_amount, ...})
    return Dataset(
        name="lumen_credit_risk",
        X=X,
        y=pd.Series(default, name="default"),
        task=TaskType.BINARY,
        target_name="default",
        feature_names=list(X.columns),
    )
```

1. The real logit also adds `- 0.02 * credit_history_length` and a small `rng.normal(0, 0.5, n)` noise term. DTI carries the largest coefficient (`4.0`), so it dominates the label — yet it never appears as a column in `X`.

The six raw columns handed to the model are `income`, `loan_amount`, `age`, `employment_years`, `credit_history_length`, and `num_prior_defaults`. A `Dataset` carries its `X`, `y`, `task`, and `feature_names` together. Split it the usual way:

```python
dataset = make_lending_dataset()
train, test = dataset.train_test_split(test_size=0.25, random_state=0)
```

!!! note "The task is `TaskType.BINARY`"

    Because the task is binary classification, the framework's default selection metric is `roc_auc`
    — that is the score the gate and AutoML maximize throughout this run.

## 2. GenAI feature engineering — discover DTI, reject noise

A *feature proposer* emits `FeatureProposal`s: a name, a line of Python that mutates a DataFrame `df`, and a rationale. In production an `AgentFeatureProposer` asks an LLM; the sample uses `StaticFeatureProposer` so it runs with **no API key**, while exercising the exact same gate.

```python
from fireflyframework_datascience.features import FeatureProposal, StaticFeatureProposer
from fireflyframework_datascience.features.genai import GenAIFeatureEngineer

proposer = StaticFeatureProposer(
    [
        FeatureProposal(
            "debt_to_income",
            "df['debt_to_income'] = df['loan_amount'] / (df['income'] + 1)",
            "DTI",
        ),
        FeatureProposal(
            "loan_per_year_employed",
            "df['loan_per_year_employed'] = df['loan_amount'] / (df['employment_years'] + 1)",
            "burden",
        ),
        FeatureProposal("noise", "df['noise'] = 0.0", "should be rejected"),
    ]
)
```

`GenAIFeatureEngineer` runs the propose → execute → measure → gate loop. For each proposal it executes the code (via `FeatureCodeExecutor`), measures the cross-validated score against the current baseline, and **accepts the feature only if the gate accepts the lift**:

```python
from sklearn.linear_model import LogisticRegression


def _logistic_scorer(task):
    return LogisticRegression(max_iter=1000)


fe = GenAIFeatureEngineer(proposer, scorer_estimator=_logistic_scorer, cv=4)  # (1)!
engineered = fe.engineer(train)

print([a.proposal.name for a in engineered.accepted])   # ['debt_to_income', ...]
print([r.proposal.name for r in engineered.rejected])   # ['noise']
print(f"lift = {engineered.lift:+.4f}")
```

1. `scorer_estimator` swaps the default `HistGradientBoosting*` scorer for a fast `LogisticRegression`, and `cv=4` sets the number of cross-validation folds used to measure each proposal's lift.

The gate is a `CostBenefitGate` with a default `min_gain` of `0.0`: a candidate is accepted only if `candidate_score - current_score > min_gain`. That is why a feature must *strictly improve* the cross-validated score to be kept.

The result object exposes:

- `engineered.dataset` — the train set with accepted features added.
- `engineered.accepted` / `engineered.rejected` — `accepted` items wrap `.proposal` plus `.score` and `.gain`; `rejected` items wrap `.proposal` plus a `.reason` (and `.score`).
- `engineered.lift` — the net cross-validated improvement (`final_score - baseline_score`).
- `engineered.summary()` — a one-line audit string of the whole step.

`debt_to_income` is accepted because it reconstructs the latent driver and lifts the score; `noise` (a constant `0.0` column) executes fine but is rejected by the gate because it adds no lift.

!!! warning "Proposed code runs through a safety analysis first"

    `FeatureCodeExecutor` is not `eval`-on-faith. Before any snippet runs it goes through a static
    safety analysis that denies imports, dunder access, and dangerous builtins (`eval`, `exec`,
    `open`, `__import__`, ...). The code then runs in a restricted namespace exposing only `df`,
    `pd`, and `np`. A snippet that is unsafe, errors, adds no new column, or produces a non-numeric
    column is turned into a `RejectedFeature` rather than crashing the loop. See
    [Security](security.md).

## 3. Classical AutoML on the engineered features

`AutoML.fit` trains its trainers over the engineered dataset and ranks them by cross-validation. The default trainer set is `random_forest`, `linear`, and `hist_gradient_boosting`, so `result.best_model.name` is one of those three.

```python
from fireflyframework_datascience.automl import AutoML

automl = AutoML(cv=4)
result = automl.fit(engineered.dataset)

print(result.best_model.name)        # the winning trainer (e.g. 'hist_gradient_boosting')
print(result.leaderboard_table())    # ranked comparison
```

To evaluate on the held-out test set, the **accepted** feature code must be re-applied so train and test stay consistent. The sample reuses `FeatureCodeExecutor` for this, then evaluates with `AutoMLResult.evaluate`:

```python
from fireflyframework_datascience.features.executor import FeatureCodeExecutor

executor = FeatureCodeExecutor()
working = test.X.copy()
for accepted in engineered.accepted:
    working = executor.execute(accepted.proposal.code, working)

engineered_test = test.with_features(working)
evaluation = result.evaluate(engineered_test)
print(evaluation.metrics)            # holdout metrics dict
```

!!! tip "Why re-apply the code, not the values"

    The accepted features were measured on the train fold. Re-executing the *same code* on the test
    frame is what keeps train and test schemas identical — the model trained on `debt_to_income`
    would fail on a test frame that does not have it. `engineered.accepted` is the audit trail that
    makes this replay exact.

For a binary task the holdout `evaluation.metrics` dict contains `accuracy`, `f1`, `precision`, and `recall`, plus `roc_auc` and `log_loss` when the winning model exposes `predict_proba`.

## 4. Serve the winner and score an applicant

`LocalModelServer` runs the winning model in-process — no network, no container — so you can score immediately.

```python
from fireflyframework_datascience.serving import LocalModelServer

server = LocalModelServer()
server.load(result.best_model)

applicant = engineered_test.X.iloc[[0]]
prediction = server.predict(applicant)
print(int(prediction[0]))            # 0 = no default, 1 = default
```

=== "In-process (the sample)"

    `LocalModelServer` is the default, dependency-free server. `load` holds the fitted `Model`;
    `predict` (and `predict_proba`) delegate straight to it. Nothing leaves the host process.

=== "Heavier servers"

    For production deployment, BentoML/KServe (and vLLM/TGI for LLMs) adapters live behind the
    `serving` extra. The port (`ModelServerPort`) is identical, so swapping the server does not
    change calling code. See [Serving](serving.md).

## Expected output

Running the sample prints a report like:

!!! success "Expected"

    ```text
    === Lumen Lending — credit-risk AutoML ===
    accepted features : ['debt_to_income', 'loan_per_year_employed']
    rejected features : ['noise']
    feature-eng lift  : +0.0XYZ
    winning model     : <trainer name>
    leaderboard:
    <ranked leaderboard table>
    holdout metrics   : {'accuracy': ..., 'roc_auc': ...}
    applicant predicted default = 0
    ```

Exact numbers vary with your scikit-learn version, but the shape is stable: **`debt_to_income` is accepted, `noise` is rejected, the lift is positive, and a winner is served.** That is the whole point — the framework rediscovers the signal you deliberately hid, and the cost/benefit gate throws away the feature that adds nothing.

## See also

- [GenAI feature engineering](genai-features.md)
- [AutoML](automl.md)
- [Datasets](datasets.md)
- [Serving](serving.md)
- [Getting started](quickstart.md)
