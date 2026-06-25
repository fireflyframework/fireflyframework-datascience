# Tutorial

**A guided, end-to-end tour of Firefly DataScience — from booting the app to serving a model.**

This tutorial mirrors the runnable script [`samples/tutorial.py`](https://github.com/fireflyframework/fireflyframework-datascience/blob/main/samples/tutorial.py),
which is covered by a test (`tests/samples/test_tutorial.py`), so everything here is guaranteed to
work. It runs **offline with no LLM key** — the GenAI steps use deterministic stand-ins, and we show
how to switch on a real LLM at the end.

```bash
uv add 'fireflyframework-datascience[tabular]'
uv run python samples/tutorial.py
```

We use a synthetic **credit-risk** dataset whose default risk is driven by *debt-to-income* — a ratio
deliberately withheld from the model, so feature engineering has something real to discover.

!!! firefly "The pattern every step rests on — the LLM proposes; the classical engine decides"

    Generative AI only ever *proposes* candidates here: feature code (step 4) and model choices
    (step 5). A deterministic classical engine cross-validates each one and a cost-benefit gate keeps
    it only if it measurably beats the current baseline. The LLM never touches the score — the data
    does. That is why the tour runs identically with or without an API key.

## 1. Boot the application

```python
from fireflyframework_datascience import FireflyDataScienceApplication

app = FireflyDataScienceApplication.run()         # (1)!
```

1. Pass `print_output=False` to suppress the banner (the script does this so its test output stays clean).

This prints the banner and a wiring summary, loads configuration, builds the dependency-injection
container, and discovers every adapter via entry-point auto-configuration. `app.bean_count` and
`app.applied_auto_configurations` tell you what got wired.

!!! success "Expected"

    On a fresh `[tabular]` install the container wires a couple dozen beans from roughly a dozen
    auto-configurations (exact counts depend on which extras are installed):

    ```text
    [1] App booted: 21 beans, 12 auto-configurations
    ```

See [Architecture](architecture.md).

## 2. Build, load, and validate the data

The script generates the credit dataset with `make_credit_dataset()`. Default risk is a logistic
function of `debt_to_income = loan_amount / income`, plus prior defaults and employment, but only the
four raw columns are handed to the model — `debt_to_income` is the hidden driver feature engineering
must rediscover.

```python
import numpy as np
import pandas as pd

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.validation.adapters import BasicValidator


def make_credit_dataset(n: int = 800, seed: int = 11) -> Dataset:
    rng = np.random.RandomState(seed)
    income = rng.normal(60_000, 18_000, n).clip(15_000, None)
    loan_amount = rng.normal(18_000, 10_000, n).clip(1_000, None)
    employment_years = rng.uniform(0, 30, n).round(1)
    num_prior_defaults = rng.poisson(0.6, n)
    dti = loan_amount / income                                            # (1)!
    logit = -2.6 + 5.0 * dti + 1.3 * num_prior_defaults - 0.05 * employment_years + rng.normal(0, 0.25, n)
    default = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    X = pd.DataFrame(
        {
            "income": income.round(2),
            "loan_amount": loan_amount.round(2),
            "employment_years": employment_years,
            "num_prior_defaults": num_prior_defaults,                      # (2)!
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


dataset = make_credit_dataset()
report = BasicValidator().validate(dataset.X, dataset.y)
assert report.ok                                  # no all-null columns, no null target, etc.
train, test = dataset.train_test_split(test_size=0.25, random_state=0)     # (3)!
```

1. `dti` drives the label but is **never** put into `X` — that is the signal step 4 has to recover.
2. Only these four raw columns reach the model; `debt_to_income` is deliberately absent.
3. `train_test_split` stratifies on the target for classification; here it yields 600 train / 200 test rows.

The `BasicValidator` catches empty data, all-null/constant columns, duplicate rows, and null targets
before you waste time training.

!!! success "Expected"

    ```text
    [2] Data validated: ok=True
    ```

    The dataset is 800 rows × 4 features, a `TaskType.BINARY` task; the split gives 600 train / 200 test.

See [Datasets](datasets.md).

## 3. Classical AutoML

```python
from fireflyframework_datascience.automl import AutoML

result = AutoML(cv=4).fit(train)
print(result.leaderboard_table())
print(result.evaluate(test))                      # holdout metrics
```

`AutoML` cross-validates each candidate trainer (`RandomForestTrainer`, `LinearTrainer`,
`HistGradientBoostingTrainer`, plus the boosting libraries if installed), ranks them on a
task-appropriate metric (`roc_auc` for binary), and refits the winner on the full training set. Each
candidate is wrapped in an impute-and-scale preprocessing pipeline before scoring.

!!! success "Expected"

    A leaderboard topped by `linear`, and a holdout `roc_auc ≈ 0.85`:

    ```text
    linear                   roc_auc=0.7867
    random_forest            roc_auc=0.7493
    hist_gradient_boosting   roc_auc=0.7335
    EvaluationResult(primary=roc_auc=0.8498; accuracy=0.7900, f1=0.7853, precision=0.7851, recall=0.7900, roc_auc=0.8498, log_loss=0.4500)
    ```

!!! note

    The leaderboard prints **cross-validation** scores on the training data, while `evaluate(test)`
    reports metrics on the untouched holdout — so the headline `roc_auc` (≈0.85) is higher than the
    CV figure (≈0.79). Both are real; they measure different things.

See [Classical AutoML](automl.md).

## 4. GenAI feature engineering

```python
from fireflyframework_datascience.features import StaticFeatureProposer, FeatureProposal
from fireflyframework_datascience.features.genai import GenAIFeatureEngineer

proposer = StaticFeatureProposer([
    FeatureProposal("debt_to_income", "df['debt_to_income'] = df['loan_amount'] / (df['income'] + 1)", "DTI"),  # (1)!
    FeatureProposal("noise", "df['noise'] = 0.0", "should be rejected"),                                       # (2)!
])
engineered = GenAIFeatureEngineer(proposer, cv=4).engineer(train)
print(engineered.summary())
```

1. The hidden driver: its CV lift clears `CostBenefitGate(min_gain=0.0)`, so it is **accepted**.
2. A constant column adds nothing, so the gate **rejects** it — the LLM never overrides that decision.

The loop is **propose → execute (safely) → measure CV lift → gate**. `debt_to_income` is accepted
because it lifts the score; the constant `noise` feature is rejected. The LLM never decides — the
measured score does.

!!! success "Expected"

    ```text
    GenAI feature engineering: 1 accepted, 1 rejected; roc_auc 0.7875 -> 0.7889 (lift +0.0013)
    ```

    `engineered.accepted` lists `debt_to_income`; `engineered.rejected` lists `noise` with the reason
    `no lift (0.7889 <= 0.7889)`. The lift is small but **positive and real** — the gate rejects
    anything that does not strictly beat the running baseline.

=== "Static (no LLM)"

    `StaticFeatureProposer` stands in for the LLM so the tutorial runs offline with a fixed,
    reproducible set of proposals — exactly what the snippet above uses.

=== "Agent (LLM)"

    With a real model you swap in `AgentFeatureProposer`, which wraps a `FireflyAgent` and is built
    lazily (no LLM client is created at startup):

    ```python
    from fireflyframework_datascience.features.genai import AgentFeatureProposer, GenAIFeatureEngineer

    proposer = AgentFeatureProposer(model="openai:gpt-4o")
    engineered = GenAIFeatureEngineer(proposer, cv=4).engineer(train)
    ```

    See [Configuring the LLM](llm-configuration.md).

See [GenAI Feature Engineering](genai-features.md).

## 5. The agentic ML-engineering loop

```python
from fireflyframework_datascience.engineering import SequenceProposer, SolutionCandidate
from fireflyframework_datascience.engineering.loop import AgenticAutoML

proposer = SequenceProposer([SolutionCandidate("linear"), SolutionCandidate("random_forest"),
                             SolutionCandidate("hist_gradient_boosting")])
run = AgenticAutoML(proposer, cv=3, max_iterations=4).solve(train)        # (1)!
print(run.summary())
```

1. `AgenticAutoML` seeds the population, then reflects on the attempt history up to `max_iterations`
   times; a `patience` budget (default 3) stops the search once it stalls.

Each candidate is trained, cross-validated, and **verified** by a `DeterministicVerifier` — it must
beat a trivial `DummyClassifier(strategy="prior")` baseline, not merely run (the "correctness ≠ ran"
principle) — before the best one is selected. `run.attempts` is the full audited trail and
`run.valid_attempts` are the ones that passed verification.

<p align="center"><img src="img/agentic-loop.svg" alt="The agentic loop: the LLM proposes a candidate, the classical engine trains and cross-validates it, and a separate verifier decides whether it beats the trivial baseline" width="100%"></p>

!!! success "Expected"

    ```text
    Agentic AutoML: 3 attempts (3 verified); best=linear roc_auc=0.7897 (baseline 0.5000)
    ```

    All three seeded candidates clear the `roc_auc=0.5000` trivial baseline, so all three are verified;
    `linear` wins.

See [Agentic Loop](agentic-loop.md).

## 6. Serve the model

```python
from fireflyframework_datascience.serving import LocalModelServer

server = LocalModelServer()
server.load(result.best_model)
prediction = server.predict(test.X.iloc[[0]])     # score one applicant
print(int(prediction[0]))
```

`LocalModelServer` is the default, dependency-free server: it loads a fitted `Model` in the host
process and answers `predict` / `predict_proba`. Heavier servers (e.g. `BentoMLModelServer`) live
behind the `serving` extra.

!!! success "Expected"

    The first holdout applicant is scored as a non-default:

    ```text
    [6] Served prediction for one applicant: default=0
    ```

See [Serving & Lineage](serving.md).

## Turn on a real LLM

Steps 4 and 5 ran offline with deterministic stand-ins. To let a real model do the proposing, set
your key and enable GenAI, then swap in the agent-backed proposers:

```bash
export OPENAI_API_KEY=sk-...                      # or ANTHROPIC_API_KEY=...
export FIREFLY_DATASCIENCE_GENAI__ENABLED=true
export FIREFLY_DATASCIENCE_GENAI__DEFAULT_MODEL=openai:gpt-4o   # or anthropic:claude-sonnet-4-5
```

```python
from fireflyframework_datascience.features.genai import AgentFeatureProposer
from fireflyframework_datascience.engineering.loop import AgentSolutionProposer
```

Use `AgentFeatureProposer` in place of `StaticFeatureProposer` (step 4) and `AgentSolutionProposer` in
place of `SequenceProposer` (step 5). Nothing else changes — the cost-benefit gate and the verifier
still decide. The full guide, including providers, keys, cost gating, and secure execution, is in
[Configuring the LLM](llm-configuration.md).

## See also

- [Quick Start](quickstart.md)
- [Samples](samples.md)
- [Configuration](configuration.md)
- [Agentic Loop](agentic-loop.md)
- [Use Case: Lumen](use-case-lumen.md)
