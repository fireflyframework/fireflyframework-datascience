# Tutorial

**A guided, end-to-end tour of Firefly DataScience — from booting the app to serving a model.**

This tutorial mirrors the runnable script [`samples/tutorial.py`](https://github.com/fireflyframework/fireflyframework-datascience/blob/main/samples/tutorial.py), which is
covered by a test, so everything here is guaranteed to work. It runs **offline with no LLM key** — the
GenAI steps use deterministic stand-ins, and we show how to switch on a real LLM at the end.

```bash
uv add 'fireflyframework-datascience[tabular]'
uv run python samples/tutorial.py
```

We use a synthetic **credit-risk** dataset whose default risk is driven by *debt-to-income* — a ratio
deliberately withheld from the model, so feature engineering has something real to discover.

## 1. Boot the application

```python
from fireflyframework_datascience import FireflyDataScienceApplication

app = FireflyDataScienceApplication.run()
```

This prints the banner and a wiring summary, loads configuration, builds the dependency-injection
container, and discovers every adapter via entry-point auto-configuration. `app.bean_count` and
`app.applied_auto_configurations` tell you what got wired. See [Architecture](architecture.md).

## 2. Load and validate the data

```python
from fireflyframework_datascience.validation.adapters import BasicValidator

dataset, validation = ...                       # build the credit dataset (see the script)
report = BasicValidator().validate(dataset.X, dataset.y)
assert report.ok                                 # no all-null columns, no null target, etc.
train, test = dataset.train_test_split(test_size=0.25, random_state=0)
```

The `BasicValidator` catches empty data, all-null/constant columns, duplicate rows, and null targets
before you waste time training. See [Datasets](datasets.md).

## 3. Classical AutoML

```python
from fireflyframework_datascience.automl import AutoML

result = AutoML(cv=4).fit(train)
print(result.leaderboard_table())
print(result.evaluate(test))                     # holdout metrics
```

AutoML cross-validates each candidate trainer (RandomForest, Linear, HistGradientBoosting, and the
boosting libraries if installed), ranks them on a task-appropriate metric (`roc_auc` for binary), and
refits the winner. Expected: a leaderboard topped by `linear` at **roc_auc ≈ 0.85** on holdout. See
[Classical AutoML](automl.md).

## 4. GenAI feature engineering

```python
from fireflyframework_datascience.features import StaticFeatureProposer, FeatureProposal
from fireflyframework_datascience.features.genai import GenAIFeatureEngineer

proposer = StaticFeatureProposer([
    FeatureProposal("debt_to_income", "df['debt_to_income'] = df['loan_amount'] / (df['income'] + 1)", "DTI"),
    FeatureProposal("noise", "df['noise'] = 0.0", "should be rejected"),
])
engineered = GenAIFeatureEngineer(proposer, cv=4).engineer(train)
print(engineered.summary())
```

The loop is **propose → execute (safely) → measure CV lift → gate**. `debt_to_income` (the hidden
driver) is **accepted** because it lifts the score; the constant `noise` feature is **rejected**. The
LLM never decides — the measured score does. See [GenAI Feature Engineering](genai-features.md).

> Here a `StaticFeatureProposer` stands in for the LLM so the tutorial runs offline. With a real model
> you'd use `AgentFeatureProposer(model="openai:gpt-4o")` — see [Configuring the LLM](llm-configuration.md).

## 5. The agentic ML-engineering loop

```python
from fireflyframework_datascience.engineering import SequenceProposer, SolutionCandidate
from fireflyframework_datascience.engineering.loop import AgenticAutoML

proposer = SequenceProposer([SolutionCandidate("linear"), SolutionCandidate("random_forest"),
                             SolutionCandidate("hist_gradient_boosting")])
run = AgenticAutoML(proposer, cv=3).solve(train)
print(run.summary())
```

Each candidate is trained, cross-validated, and **verified** — it must beat a trivial baseline, not
merely run (the "correctness ≠ ran" principle) — before the best one is selected. `run.attempts` is the
full audited trail. See [Agentic Loop](agentic-loop.md).

## 6. Serve the model

```python
from fireflyframework_datascience.serving import LocalModelServer

server = LocalModelServer()
server.load(result.best_model)
prediction = server.predict(test.X.iloc[[0]])    # score one applicant
```

See [Serving & Lineage](serving.md).

## Turn on a real LLM

```bash
export OPENAI_API_KEY=sk-...                       # or ANTHROPIC_API_KEY=...
export FIREFLY_DATASCIENCE_GENAI__ENABLED=true
export FIREFLY_DATASCIENCE_GENAI__DEFAULT_MODEL=openai:gpt-4o
```

Then use `AgentFeatureProposer` / `AgentSolutionProposer` in place of the stand-ins. The full guide,
including providers, keys, cost gating, and secure execution, is in
[Configuring the LLM](llm-configuration.md).

## See also

- [Quick Start](quickstart.md) · [Configuration](configuration.md) · [Use Case: Lumen](use-case-lumen.md)
