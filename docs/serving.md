# Serving & Lineage

**Serve a trained model in-process by default, track experiments, and emit lineage — all behind narrow ports you can swap without touching your code.**

Firefly DataScience keeps the core dependency-free. A fitted `Model` is served by a `ModelServerPort`; experiment runs go through a `TrackerPort`; data and model lineage flows through a `LineagePort`. Each port ships a zero-dependency default — registered automatically by the container — and an opt-in adapter behind an extra. Because every adapter implements the same port, moving from the in-process default to MLflow, BentoML, or OpenLineage is a configuration change, not a rewrite.

!!! firefly "The same gate, applied to operations"

    The training loop trusts only measured improvement; serving and lineage extend that discipline to
    production. The defaults are deterministic and dependency-free, so a run that scored well in
    development behaves identically when served. Heavier backends are opt-in and swapped behind a port —
    you adopt MLflow or OpenLineage when they earn their keep, never by default.

<p align="center">
  <img src="img/ecosystem.svg" alt="Firefly DataScience ecosystem: in-process defaults with opt-in adapters behind ports" width="85%">
</p>

## The model-server port

A `ModelServerPort` loads a fitted `Model` and answers prediction requests. It is a `runtime_checkable` `Protocol`, so any object with a `name` attribute plus `load` and `predict` methods satisfies it — there is no base class to inherit.

```python
from typing import Any, Protocol, runtime_checkable
from fireflyframework_datascience.models import Model

@runtime_checkable
class ModelServerPort(Protocol):
    name: str
    def load(self, model: Model) -> None: ...
    def predict(self, X: Any) -> Any: ...
```

The container registers a server for you. `ServingAutoConfiguration` provides `LocalModelServer` as the primary `ModelServerPort` bean, and only when no other bean of that type is already present (`@conditional_on_missing_bean`). Register your own adapter and it wins; otherwise the in-process default applies.

## LocalModelServer (default)

`LocalModelServer` serves a fitted `Model` in the host process. It is the default and pulls in no external dependency.

```python
from fireflyframework_datascience.serving import LocalModelServer

server = LocalModelServer()      # server.name == "local"
server.load(model)               # model: a fitted Model
preds = server.predict(X_test)
proba = server.predict_proba(X_test)   # if the estimator supports it
```

Calling `predict` (or `predict_proba`) before `load` raises `FireflyDataScienceError("No model loaded — call load(model) first")`. The loaded model is available via the read-only `server.model` property, which is `None` until you call `load`.

!!! note "predict_proba is not part of the port"

    `predict_proba` is an extra on `LocalModelServer`, not on `ModelServerPort`. Code written against the
    port should rely on `name`, `load`, and `predict` only. Under the hood `Model.predict_proba` raises
    `AttributeError` if the wrapped estimator does not implement it.

### Loading a trained model and serving predictions

A `Model` persists with `joblib`; reload it and hand it to the server.

```python
from fireflyframework_datascience.models import Model
from fireflyframework_datascience.serving import LocalModelServer

# After training elsewhere, the Model was saved:
#   model.save("artifacts/churn.joblib")

# Load it back (only from trusted, first-party locations — joblib uses pickle):  # (1)!
model = Model.load("artifacts/churn.joblib")

server = LocalModelServer()
server.load(model)

predictions = server.predict(X_new)
print(predictions)
```

1.  `Model.load` uses `joblib`, which uses `pickle` — and `pickle` executes arbitrary code on load. Load
    models only from trusted, first-party locations (your own registry or artifact store), never from
    untrusted input. See [Security](security.md) for the threat model.

Because `LocalModelServer` simply delegates to `Model.predict` / `Model.predict_proba`, the served output matches what the estimator produces directly.

## BentoMLModelServer (gated)

For packaging and deployment to a BentoML service, use `BentoMLModelServer` from the adapters module. It requires the `serving` extra.

=== "In-process (default)"

    ```python
    from fireflyframework_datascience.serving import LocalModelServer

    server = LocalModelServer()      # name == "local", no extra dependency
    server.load(model)
    preds = server.predict(X_new)
    ```

=== "BentoML (gated)"

    ```bash
    pip install "fireflyframework-datascience[serving]"
    ```

    ```python
    from fireflyframework_datascience.serving.adapters import BentoMLModelServer

    server = BentoMLModelServer()    # name == "bentoml"; raises AdapterUnavailableError without the extra
    server.load(model)
    preds = server.predict(X_new)
    ```

Without `bentoml` installed, construction raises `AdapterUnavailableError("BentoMLModelServer", "serving")`. Both servers expose the same `name`/`load`/`predict` surface, so swapping is a one-line change. Calling `predict` before `load` on `BentoMLModelServer` raises `FireflyDataScienceError("No model loaded")`.

!!! warning "BentoML packaging is a deployment concern"

    `BentoMLModelServer` wraps a fitted model and integrates with BentoML's runner API when available;
    full service packaging (bentos, runners, the HTTP server) lives in your deployment pipeline, outside
    this reference adapter.

## Experiment tracking

The `TrackerPort` records params, metrics, and model artifacts for a run. `start_run` returns a `RunHandle` — an opaque dataclass with `run_id` and `name`.

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

`TrackingAutoConfiguration` registers `NoOpTracker` by default (`@conditional_on_missing_bean`), and swaps in `MLflowTracker` as the primary bean only when the `tracking_enabled` config property is `True` (it defaults to `False`). You opt in to MLflow through configuration; nothing in your training code changes.

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

!!! note "RunHandle naming"

    `NoOpTracker.start_run` always returns `run_id="noop"`. The handle's `name` is the `run_name` you pass,
    falling back to `"run"` when you pass none.

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

Construction raises `AdapterUnavailableError("MLflowTracker", "tracking")` when `mlflow` is not installed. The API is identical to `NoOpTracker`, so code written against the port runs unchanged with either tracker. `tracking_uri` defaults to `None` (MLflow's local store) and `experiment` defaults to `"firefly-datascience"`; under the hood `MLflowTracker` calls `mlflow.set_experiment(...)` on construction and `mlflow.sklearn.log_model(...)` for `log_model`.

!!! tip "Same code, two backends"

    Write against `TrackerPort`, develop with `NoOpTracker`, then flip `tracking_enabled` to `True` (and
    install the `tracking` extra) to capture the very same runs in MLflow — no call-site edits.

## Model registry

The `RegistryPort` persists and retrieves models by name and version — a separate port from tracking, so a registry adapter can be swapped independently.

```python
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class RegistryPort(Protocol):
    name: str
    def register(self, model: Any, name: str) -> str: ...
    def load(self, name: str, version: str | None = None) -> Any: ...
```

`register` returns the assigned version identifier; `load` resolves the latest version when `version` is `None`. Treat the registry as the trusted source for `Model.load` — only first-party artifact stores are safe to deserialize, because `joblib` uses `pickle`.

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

`LineageEvent` is a dataclass whose `inputs`, `outputs`, and `metadata` all default to empty — only `name` is required. `LineageAutoConfiguration` registers `NoOpLineage` as the primary `LineagePort` bean by default, so lineage is always on but emits nowhere (it logs at debug level) until you provide a real backend.

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
lineage.emit(event)            # lineage.name == "openlineage"
```

Without the `openlineage` client installed, construction raises `AdapterUnavailableError("OpenLineageEmitter", "lineage")`. `url` defaults to `"http://localhost:5000"` and `namespace` to `"firefly-datascience"`; the emitter constructs an `OpenLineageClient(url=url)` and forwards events to it.

!!! success "Expected"

    With the default `NoOpLineage`, `emit` returns `None` and writes a debug log line — your pipeline runs
    unchanged whether or not a lineage backend is wired up. Swap in `OpenLineageEmitter` (same `emit`
    surface) to send the same events to Marquez.

## See also

- [Architecture](architecture.md) — the ports-and-adapters design these servers, trackers, and emitters plug into.
- [AutoML](automl.md) — how a `Model` is trained, scored, and selected before you serve it.
- [Configuration](configuration.md) — toggles such as `tracking_enabled` that pick the adapter.
- [Security](security.md) — why `Model.load` must read only from trusted, first-party locations.
- [Quickstart](quickstart.md) — train and serve a first model end to end.
