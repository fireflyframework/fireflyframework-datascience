# Serving & Lineage

**Serve a trained model in-process by default, track experiments, and emit lineage — all behind narrow ports you can swap.**

Firefly DataScience keeps the core dependency-free. A fitted `Model` is served by a `ModelServerPort`; experiment runs go through a `TrackerPort`; data/model lineage flows through a `LineagePort`. Each port ships a zero-dependency default and an opt-in adapter behind an extra.

<p align="center">
  <img src="img/ecosystem.svg" alt="Firefly ecosystem" width="85%">
</p>

## The model-server port

```python
from typing import Any, Protocol, runtime_checkable
from fireflyframework_datascience.models import Model

@runtime_checkable
class ModelServerPort(Protocol):
    name: str
    def load(self, model: Model) -> None: ...
    def predict(self, X: Any) -> Any: ...
```

Any object with a `name`, `load`, and `predict` satisfies the port — no base class to inherit.

## LocalModelServer (default)

`LocalModelServer` serves a fitted `Model` in the host process. It is the default and pulls in no external dependency.

```python
from fireflyframework_datascience.serving import LocalModelServer

server = LocalModelServer()      # server.name == "local"
server.load(model)               # model: a fitted Model
preds = server.predict(X_test)
proba = server.predict_proba(X_test)   # if the estimator supports it
```

Calling `predict` before `load` raises `FireflyDataScienceError`. The loaded model is available via `server.model`.

### Loading a trained model and serving predictions

A `Model` persists with `joblib`; reload it and hand it to the server.

```python
from fireflyframework_datascience.models import Model
from fireflyframework_datascience.serving import LocalModelServer

# After training elsewhere, the Model was saved:
#   model.save("artifacts/churn.joblib")

# Load it back (only from trusted, first-party locations — joblib uses pickle):
model = Model.load("artifacts/churn.joblib")

server = LocalModelServer()
server.load(model)

predictions = server.predict(X_new)
print(predictions)
```

Because `LocalModelServer` simply delegates to `Model.predict` / `Model.predict_proba`, the served output matches what the estimator produces directly.

## BentoMLModelServer (gated)

For packaging/deployment to a BentoML service, use `BentoMLModelServer` from the adapters module. It requires the `serving` extra.

```bash
pip install "fireflyframework-datascience[serving]"
```

```python
from fireflyframework_datascience.serving.adapters import BentoMLModelServer

server = BentoMLModelServer()    # raises AdapterUnavailableError without the extra
server.load(model)
preds = server.predict(X_new)
```

Without `bentoml` installed, construction raises `AdapterUnavailableError("BentoMLModelServer", "serving")`. Both servers expose the same `name`/`load`/`predict` surface, so swapping is a one-line change.

## Experiment tracking

The `TrackerPort` records params, metrics, and model artifacts for a run.

```python
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable
from fireflyframework_datascience.tracking import RunHandle

@runtime_checkable
class TrackerPort(Protocol):
    name: str
    def start_run(self, run_name: str | None = None) -> RunHandle: ...
    def log_params(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None: ...
    def log_model(self, model: Any, artifact_name: str = "model") -> None: ...
    def end_run(self) -> None: ...
```

### NoOpTracker (default)

Records nothing (logs at debug level) and keeps the core dependency-free.

```python
from fireflyframework_datascience.tracking.adapters import NoOpTracker

tracker = NoOpTracker()          # tracker.name == "noop"
run = tracker.start_run("baseline")     # -> RunHandle(run_id="noop", name="baseline")
tracker.log_params({"max_depth": 6, "n_estimators": 200})
tracker.log_metrics({"auc": 0.91, "accuracy": 0.88})
tracker.log_model(model.estimator, artifact_name="churn")
tracker.end_run()
```

### MLflowTracker (opt-in)

Logs to an MLflow backend. Requires the `tracking` extra.

```bash
pip install "fireflyframework-datascience[tracking]"
```

```python
from fireflyframework_datascience.tracking.adapters import MLflowTracker

tracker = MLflowTracker(
    tracking_uri="http://localhost:5000",
    experiment="firefly-datascience",
)
run = tracker.start_run("rf-sweep")
tracker.log_params({"max_depth": 6})
tracker.log_metrics({"auc": 0.93}, step=1)
tracker.log_model(model.estimator, artifact_name="churn")
tracker.end_run()
```

Construction raises `AdapterUnavailableError("MLflowTracker", "tracking")` when `mlflow` is not installed. The API is identical to `NoOpTracker`, so code written against the port runs unchanged with either tracker.

## Lineage

The `LineagePort` emits a `LineageEvent` — a named run with input/output dataset references and metadata — to a backend.

```python
from fireflyframework_datascience.lineage import LineageEvent, NoOpLineage

event = LineageEvent(
    name="train-churn",
    inputs=["s3://lake/customers.parquet"],
    outputs=["artifacts/churn.joblib"],
    metadata={"rows": 120_000},
)

lineage = NoOpLineage()          # default; lineage.name == "noop"
lineage.emit(event)
```

### OpenLineageEmitter (gated)

Emits to an OpenLineage backend such as Marquez. Requires the `lineage` extra.

```bash
pip install "fireflyframework-datascience[lineage]"
```

```python
from fireflyframework_datascience.lineage.adapters import OpenLineageEmitter

lineage = OpenLineageEmitter(
    url="http://localhost:5000",
    namespace="firefly-datascience",
)
lineage.emit(event)
```

Without the `openlineage` client installed, construction raises `AdapterUnavailableError("OpenLineageEmitter", "lineage")`.

## See also

- [Models & Training](automl.md)
- [Tuning](automl.md)
- [Getting Started](quickstart.md)
