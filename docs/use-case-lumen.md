# Use Case: Lumen Lending Credit Risk

**An end-to-end walkthrough of the Firefly DataScience stack — GenAI feature engineering, classical AutoML, and in-process serving — on a realistic synthetic lending dataset.**

The `samples/lumen_credit_risk.py` sample tells one focused story: a credit-risk model where default is *secretly* driven by **debt-to-income (DTI)**, a feature the model is never handed directly. We watch the framework rediscover it from raw columns, reject a useless noise feature, let AutoML pick a winner, and score a live applicant — all without an LLM API key.

The pipeline has three acts:

1. **GenAI feature engineering** proposes domain features as executable code; a cost/benefit gate keeps a feature only if it measurably lifts a cross-validated baseline.
2. **Classical AutoML** trains several models and selects the best by cross-validation.
3. **Serving** loads the winner in-process and scores a new applicant.

## Run it

```bash
python samples/lumen_credit_risk.py   # needs the `tabular` extra
```

## 1. Synthetic lending data

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
    logit = -3.0 + 4.0 * dti + 0.8 * num_prior_defaults - 0.05 * employment_years
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

A `Dataset` carries its `X`, `y`, `task`, and `feature_names` together. Split it the usual way:

```python
dataset = make_lending_dataset()
train, test = dataset.train_test_split(test_size=0.25, random_state=0)
```

## 2. GenAI feature engineering — discover DTI, reject noise

A *feature proposer* emits `FeatureProposal`s: a name, a line of Python that mutates a DataFrame `df`, and a rationale. In production a `GenAIFeatureProposer` asks an LLM; the sample uses `StaticFeatureProposer` so it runs with **no API key**, while exercising the exact same gate.

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

`GenAIFeatureEngineer` executes each proposal's code, measures the cross-validated lift against a baseline scorer, and **accepts a feature only if it helps**:

```python
from sklearn.linear_model import LogisticRegression


def _logistic_scorer(task):
    return LogisticRegression(max_iter=1000)


fe = GenAIFeatureEngineer(proposer, scorer_estimator=_logistic_scorer, cv=4)
engineered = fe.engineer(train)

print([a.proposal.name for a in engineered.accepted])   # ['debt_to_income', ...]
print([r.proposal.name for r in engineered.rejected])   # ['noise']
print(f"lift = {engineered.lift:+.4f}")
```

The result object exposes:

- `engineered.dataset` — the train set with accepted features added.
- `engineered.accepted` / `engineered.rejected` — each wraps `.proposal` (so `.proposal.name`, `.proposal.code`).
- `engineered.lift` — the net cross-validated improvement.

`debt_to_income` is accepted because it reconstructs the latent driver and lifts the score; `noise` (a constant column) is rejected because it adds nothing.

## 3. Classical AutoML on the engineered features

`AutoML.fit` trains its trainers over the engineered dataset and ranks them by cross-validation.

```python
from fireflyframework_datascience.automl import AutoML

automl = AutoML(cv=4)
result = automl.fit(engineered.dataset)

print(result.best_model.name)        # the winning trainer
print(result.leaderboard_table())    # ranked comparison
```

To evaluate on the held-out test set, the **accepted** feature code must be re-applied so train and test stay consistent. The sample uses `FeatureCodeExecutor` for this, then evaluates:

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

## Expected output

Running the sample prints a report like:

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

Exact numbers vary with your scikit-learn version, but the shape is stable: **`debt_to_income` is accepted, `noise` is rejected, the lift is positive, and a winner is served.** That is the whole point — the framework rediscovers the signal you deliberately hid.

## See also

- [GenAI Feature Engineering](genai-feature-engineering.md)
- [AutoML](automl.md)
- [Datasets](datasets.md)
- [Serving](serving.md)
- [Getting Started](getting-started.md)
