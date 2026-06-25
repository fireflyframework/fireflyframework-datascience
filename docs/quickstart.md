# Quick Start

**Go from `uv add` to a fitted model and a working `firefly-ds` CLI in minutes — AutoML that fuses GenAI with classical ML and Deep Learning.**

Firefly DataScience is a hexagonal, secure-by-default Python metaframework. The core stays import-light: heavy libraries (pandas, scikit-learn, XGBoost, MLflow, …) live behind optional extras and are loaded lazily, so you only install what you use. This page walks the shortest path: install an extra, boot the application, run AutoML two ways, and verify your environment with the CLI.

!!! firefly "The reproducible pattern — the LLM proposes; the classical engine decides"

    Everything below is classical-first by default. GenAI is **off** unless you enable it, and when
    enabled it is a governed, cost-benefit-gated accelerator over a deterministic classical engine —
    never a black box. The defaults you boot with reflect that: `genai` disabled, `sandbox = monty`.

## Install

Firefly DataScience requires **Python 3.13+**. The only hard dependency is the Firefly Agentic GenAI substrate; everything else is an optional extra. Pick the extra that matches what you are doing:

=== "Core only"

    No heavy ML libraries — just ports, the application bootstrap, and the DI container.

    ```bash
    uv add fireflyframework-datascience
    ```

=== "Classical tabular"

    pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna.

    ```bash
    uv add "fireflyframework-datascience[tabular]"
    ```

=== "AutoML stack"

    The curated bundle: tabular + tabfm + autogluon + tracking + validation + data.

    ```bash
    uv add "fireflyframework-datascience[automl-stack]"
    ```

=== "GenAI"

    GenAI accelerators (script execution, embeddings, vector stores via Firefly Agentic).

    ```bash
    uv add "fireflyframework-datascience[genai]"
    ```

=== "Everything"

    tabular, DL, NLP, mlops, serving, lineage, orchestration, genai.

    ```bash
    uv add "fireflyframework-datascience[full]"
    ```

!!! tip "Extras compose"

    Combine extras in a single brackets clause, e.g. `uv add "fireflyframework-datascience[tabular,tracking,genai]"`.

## Boot the application

`FireflyDataScienceApplication` mirrors the PyFly / Spring Boot lifecycle: load config → print banner → build the DI container → discover and apply auto-configurations → eagerly initialize singletons → return a ready `ApplicationContext`.

```python
from fireflyframework_datascience import FireflyDataScienceApplication

# Construct and start in one call.
app = FireflyDataScienceApplication.run()  # (1)!

print(app.bean_count)                    # (2)!
print(app.config.default_ml_framework)   # (3)!
print(app.applied_auto_configurations)   # (4)!
```

1. `run(**kwargs)` constructs the application and immediately calls `start()`, returning a started `ApplicationContext`.
2. `bean_count` is the number of wired beans (`len(app.container)`).
3. The active ML framework from config — `"sklearn"` by default.
4. The auto-configuration classes whose conditions matched and were applied, in `@order`.

`run(**kwargs)` forwards to the constructor. Common options:

```python
app = FireflyDataScienceApplication.run(
    config_dir="./config",        # directory containing firefly-datascience.yaml
    profiles=["local"],           # active configuration profiles
    print_output=False,           # silence the banner + wiring summary
)
```

When `print_output` is left on (the default), the application prints a short wiring summary after start — your `profiles`, `beans`, `auto-config` count, `ml framework`, `genai` state, and `sandbox`. Resolve wired components from the container by type:

```python
from fireflyframework_datascience.models import TrainerPort

trainers = app.container.resolve_all(TrainerPort)   # all registered trainers
config = app.get(type(app.config))                  # or app.config directly
```

## Run AutoML

`AutoML` is classical tabular AutoML: it validates the data (if a validator is wired), cross-validates a set of trainers (optionally tuning each), then fits the winner and ranks every candidate in a leaderboard. It works two ways — the framework serves both notebook-driven data scientists and DI-wired app developers.

### Imperative (notebook style)

Build a `Dataset` from any sklearn-style dataset and call `fit`. With no arguments, `AutoML()` uses the default trainers (`RandomForestTrainer`, `LinearTrainer`, `HistGradientBoostingTrainer`), a default evaluator (`SklearnMetricsEvaluator`), and a default search policy (`DefaultSearchPolicy`).

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
    task=TaskType.BINARY,                # breast cancer is a binary task -> roc_auc by default
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

The leaderboard is sorted best-first, and each entry stringifies as the model name padded to a column followed by `<metric>=<score>`:

!!! success "Representative output"

    ```text
    RandomForestTrainer
    0.9789
    RandomForestTrainer      roc_auc=0.9789
    HistGradientBoostingTrainer roc_auc=0.9761
    LinearTrainer            roc_auc=0.9743
    ```

    Exact scores depend on the data, CV splits, and trial budget; the format is fixed.

`fit` accepts overrides — `AutoML().fit(dataset, task=TaskType.REGRESSION, metric="r2")` — otherwise the task comes from `dataset.task` and the metric from the evaluator's default for that task.

Hold out a test split the usual way:

```python
train, test = dataset.train_test_split(test_size=0.25, random_state=42)
result = AutoML().fit(train)
report = result.evaluate(test)
```

### Declarative (DI-wired)

`AutoML.from_context` pulls its trainers, evaluator, search policy, validator, and tracker straight from the application container, so an app's auto-configured (or custom) adapters are used automatically. Each component falls back to its default when not registered, and `**overrides` set the engine knobs (`cv`, `n_trials`, `random_state`).

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

`doctor` verifies that the required Firefly Agentic substrate is present, then prints an installed / partial / not-installed status for every optional extra (`tabular`, `tabfm`, `automl`, `dl`, `nlp`, `tracking`, `validation`, `featurestore`, `serving`, `lineage`, `orchestration`, `data`, `genai`) — the fastest way to confirm your environment before a run.

!!! success "Expected — `firefly-ds doctor`"

    ```text
    Firefly DataScience doctor — v0.1.0
      python : 3.13.1 (macOS-15.5-arm64-arm-64bit)
      agentic: ok (required)
                       Optional adapter extras
    ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┓
    ┃ extra           ┃ status        ┃ modules ┃
    ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━┩
    │ tabular         │ installed     │ 7/7     │
    │ tabfm           │ not installed │ 0/1     │
    │ automl          │ not installed │ 0/1     │
    │ dl              │ not installed │ 0/4     │
    │ nlp             │ not installed │ 0/4     │
    │ tracking        │ installed     │ 1/1     │
    │ validation      │ installed     │ 1/1     │
    │ featurestore    │ not installed │ 0/1     │
    │ serving         │ not installed │ 0/1     │
    │ lineage         │ not installed │ 0/1     │
    │ orchestration   │ not installed │ 0/1     │
    │ data            │ partial       │ 1/2     │
    │ genai           │ installed     │ 2/2     │
    └─────────────────┴───────────────┴─────────┘
    ```

    `installed` means every representative module for the extra resolves; `partial` means some do; `not installed` means none. Your rows depend on which extras you installed.

!!! warning "If `agentic` is MISSING"

    The Firefly Agentic substrate is the one hard dependency. If `doctor` reports `agentic: MISSING`,
    the application will not boot — reinstall the package (the base install pulls Agentic in).

## See also

- [Configuration](configuration.md) — `firefly-datascience.yaml`, profiles, and `FireflyDataScienceConfig`
- [Datasets](datasets.md) — the `Dataset` container and `DatasetLoaderPort`
- [AutoML](automl.md) — trainers, search policies, evaluators, and the leaderboard
- [Architecture](architecture.md) — the hexagonal ports, DI container, and bootstrap lifecycle
- [GenAI features](genai-features.md) — fusing Firefly Agentic with classical ML
