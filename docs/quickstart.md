# Quick Start

**Boot a Firefly DataScience application and run AutoML in minutes — AutoML that fuses GenAI with classical ML and Deep Learning.**

Firefly DataScience is a hexagonal, secure-by-default Python metaframework. The core stays import-light: heavy libraries (pandas, scikit-learn, XGBoost, MLflow, …) live behind optional extras and are loaded lazily, so you only install what you use. This page gets you from `uv add` to a fitted model and the `firefly-ds` CLI.

## Install

Firefly DataScience requires **Python 3.13+**. The only hard dependency is the Firefly Agentic GenAI substrate; everything else is an extra.

```bash
# core only (ports, application, DI — no heavy ML libs)
uv add fireflyframework-datascience

# classical tabular ML: pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna
uv add "fireflyframework-datascience[tabular]"

# the curated AutoML bundle: tabular + tabfm + autogluon + tracking + validation + data
uv add "fireflyframework-datascience[automl-stack]"

# GenAI accelerators (script execution, embeddings, vector stores via Firefly Agentic)
uv add "fireflyframework-datascience[genai]"

# everything (tabular, DL, NLP, mlops, serving, lineage, orchestration, genai)
uv add "fireflyframework-datascience[full]"
```

Extras compose, e.g. `uv add "fireflyframework-datascience[tabular,tracking,genai]"`.

## Boot the application

`FireflyDataScienceApplication` mirrors the PyFly / Spring Boot lifecycle: load config → print banner → build the DI container → discover and apply auto-configurations → eagerly initialize singletons → return a ready `ApplicationContext`.

```python
from fireflyframework_datascience import FireflyDataScienceApplication

# Construct and start in one call.
app = FireflyDataScienceApplication.run()

print(app.bean_count)                    # number of wired beans
print(app.config.default_ml_framework)   # active ML framework from config
print(app.applied_auto_configurations)   # the auto-configs whose conditions matched
```

`run(**kwargs)` forwards to the constructor. Common options:

```python
app = FireflyDataScienceApplication.run(
    config_dir="./config",        # directory containing firefly-datascience.yaml
    profiles=["local"],           # active configuration profiles
    print_output=False,           # silence the banner + wiring summary
)
```

Resolve wired components from the container by type:

```python
from fireflyframework_datascience.models import TrainerPort

trainers = app.container.resolve_all(TrainerPort)   # all registered trainers
config = app.get(type(app.config))                  # or app.config directly
```

## Run AutoML

`AutoML` is classical tabular AutoML: it cross-validates a set of trainers (optionally tuning each), then fits the winner. It works two ways.

### Imperative (notebook style)

Build a `Dataset` from any sklearn-style dataset and call `fit`. With no arguments, `AutoML()` uses the default trainers (random forest, linear, hist gradient boosting), a default evaluator, and a default search policy.

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer

from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset

raw = load_breast_cancer(as_frame=True)
X: pd.DataFrame = raw.data
y = raw.target

dataset = Dataset(
    name="breast_cancer",
    X=X,
    y=y,
    task=TaskType.CLASSIFICATION,
    target_name="target",
    feature_names=list(X.columns),
)

# Cross-validate candidates and fit the winner.
result = AutoML(cv=5, n_trials=20, random_state=42).fit(dataset)

print(result.best_model.name)   # winning trainer
print(result.best_score)        # winner's CV score
for entry in result.leaderboard:
    print(entry)                # "<model>  <metric>=<score>"

# Predict with the fitted winner.
preds = result.predict(dataset.X)
```

`fit` accepts overrides — `AutoML().fit(dataset, task=TaskType.REGRESSION, metric="r2")` — otherwise the task comes from `dataset.task` and the metric from the evaluator's default for that task.

Hold out a test split the usual way:

```python
train, test = dataset.train_test_split(test_size=0.25, random_state=42)
result = AutoML().fit(train)
report = result.evaluate(test)
```

### Declarative (DI-wired)

`AutoML.from_context` pulls its trainers, evaluator, search policy, validator, and tracker straight from the application container, so an app's auto-configured (or custom) adapters are used automatically.

```python
from fireflyframework_datascience import FireflyDataScienceApplication
from fireflyframework_datascience.automl import AutoML

app = FireflyDataScienceApplication.run()

# Components are resolved from the DI container; kwargs override engine settings.
automl = AutoML.from_context(app, cv=5, n_trials=20)
result = automl.fit(dataset)
```

## The `firefly-ds` CLI

Installing the package exposes the `firefly-ds` command (run with `uv run firefly-ds <cmd>`).

```bash
# Print the framework version.
firefly-ds version

# Check the environment and report which adapter extras are installed.
firefly-ds doctor

# Boot the app and list applied auto-configurations + registered beans.
firefly-ds introspect

# introspect with explicit config and profiles.
firefly-ds introspect --config-dir ./config --profile local --profile gpu
```

`doctor` verifies that the required Firefly Agentic substrate is present and prints an installed/partial/not-installed status for every optional extra (`tabular`, `automl`, `dl`, `nlp`, `tracking`, `genai`, …) — the fastest way to confirm your environment before a run.

## See also

- [Configuration](configuration.md) — `firefly-datascience.yaml`, profiles, and `FireflyDataScienceConfig`
- [Datasets](datasets.md) — the `Dataset` container and `DatasetLoaderPort`
- [AutoML](automl.md) — trainers, search policies, evaluators, and the leaderboard
- [GenAI](genai.md) — fusing Firefly Agentic with classical ML
- [CLI reference](cli.md) — every `firefly-ds` command
