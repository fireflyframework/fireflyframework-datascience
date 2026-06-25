# Deep Learning & Tabular Foundation Models

**Neural and tabular-foundation-model training behind two import-light ports — with a verified sklearn reference and gated PyTorch / TabPFN adapters.**

The `dl` module defines two ports for non-classical-ML training. `DLTrainerPort` covers neural trainers; `TabFMPort` covers tabular foundation models (in-context fit/predict, e.g. TabPFN). Both are runtime-checkable `Protocol`s and share the same shape as the rest of the framework: `supports(task)` plus `fit(dataset) -> Model`.

The module itself is import-light — it pulls in no heavy dependencies. A verified `MLPTrainer` (scikit-learn) ships as the reference adapter and needs only the `tabular` extra. The heavy adapters (`TabPFNPredictor`, `TorchTabularTrainer`) are **gated behind extras** and raise a clear error when those extras are missing.

## The ports

Both live in `fireflyframework_datascience.dl`:

```python
from fireflyframework_datascience.dl import DLTrainerPort, TabFMPort
from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.models import Model


class DLTrainerPort(Protocol):
    name: str
    def supports(self, task: TaskType) -> bool: ...
    def fit(self, dataset: Dataset) -> Model: ...


class TabFMPort(Protocol):
    name: str
    def supports(self, task: TaskType) -> bool: ...
    def fit(self, dataset: Dataset) -> Model: ...
```

Because they are `@runtime_checkable`, any object with the right attributes satisfies them:

```python
from fireflyframework_datascience.dl import DLTrainerPort
from fireflyframework_datascience.dl.adapters import MLPTrainer

assert isinstance(MLPTrainer(), DLTrainerPort)  # True
```

## MLPTrainer — the verified neural reference

`MLPTrainer` wraps scikit-learn's `MLPClassifier` / `MLPRegressor` in the standard preprocessing pipeline and returns a fitted `Model`. It is verified in the PR gate and needs only the `tabular` extra — no PyTorch required.

```bash
pip install "fireflyframework-datascience[tabular]"
```

```python
import pandas as pd
from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.dl.adapters import MLPTrainer

X = pd.DataFrame({"age": [25, 41, 33, 52], "income": [40_000, 88_000, 61_000, 120_000]})
y = pd.Series([0, 1, 0, 1])
dataset = Dataset(
    name="credit",
    X=X,
    y=y,
    task=TaskType.BINARY,
    target_name="approved",
    feature_names=["age", "income"],
)

trainer = MLPTrainer()
assert trainer.supports(dataset.task)

model = trainer.fit(dataset)   # -> fireflyframework_datascience.models.Model
preds = model.predict(X)
proba = model.predict_proba(X)
```

`supports` is honest about scope — `MLPTrainer` accepts classification (`BINARY`, `MULTICLASS`, `CLASSIFICATION`) and `REGRESSION`:

```python
trainer.supports(TaskType.MULTICLASS)   # True
trainer.supports(TaskType.REGRESSION)   # True
```

The returned `Model` is the same wrapper the rest of the framework uses, so you get `predict` / `predict_proba` / `save` / `load` for free.

## TabPFNPredictor — tabular foundation model (`tabfm` extra)

`TabPFNPredictor` implements `TabFMPort` over [TabPFN](https://github.com/PriorLabs/TabPFN): an in-context transformer that fits and predicts without conventional gradient training. It is **gated behind the `tabfm` extra**.

```bash
pip install "fireflyframework-datascience[tabfm]"
```

```python
from fireflyframework_datascience.dl.adapters import TabPFNPredictor

predictor = TabPFNPredictor()
predictor.name              # "tabpfn"
predictor.supports(TaskType.BINARY)   # True

model = predictor.fit(dataset)        # in-context fit -> Model
preds = model.predict(dataset.X)
```

If the `tabfm` extra is not installed, `fit` raises a clear, actionable error rather than an opaque `ImportError`:

```python
from fireflyframework_datascience.core.exceptions import AdapterUnavailableError

try:
    TabPFNPredictor().fit(dataset)
except AdapterUnavailableError as exc:
    # names the adapter ("TabPFNPredictor") and the missing extra ("tabfm")
    print(exc)
```

## TorchTabularTrainer — the PyTorch / Lightning integration point (`dl` extra)

`TorchTabularTrainer` is the `DLTrainerPort` adapter where full deep-learning workloads plug in: PyTorch Lightning, HuggingFace Accelerate, distributed training (FSDP/DDP), and PEFT/TRL all share this contract. It is **gated behind the `dl` extra** and verified under the nightly/integration suite rather than the PR gate.

```bash
pip install "fireflyframework-datascience[dl]"
```

```python
from fireflyframework_datascience.dl.adapters import TorchTabularTrainer

trainer = TorchTabularTrainer(epochs=50, hidden=64, lr=1e-3)
trainer.name                       # "torch_tabular"
trainer.supports(TaskType.REGRESSION)   # True
```

Be honest about current state: in the published package the training loop lives behind the `dl` extra and the nightly suite. Without `torch` installed, `fit` raises `AdapterUnavailableError("TorchTabularTrainer", "dl")`; with `torch` present, the reference build raises `NotImplementedError` pointing at the nightly DL suite. Treat it as the **integration seam** — the contract (`supports` + `fit(dataset) -> Model`) is identical to `MLPTrainer`, so your own Lightning/HF trainer can drop straight in:

```python
class MyLightningTrainer:
    name = "lightning_tabular"

    def supports(self, task):
        return task.is_classification() or task is TaskType.REGRESSION

    def fit(self, dataset):
        # ... train with PyTorch Lightning / Accelerate ...
        return Model("lightning_tabular", estimator, dataset.task, list(dataset.feature_names))

# isinstance(MyLightningTrainer(), DLTrainerPort) -> True
```

## Choosing an adapter

| Adapter | Port | Extra | Status |
| --- | --- | --- | --- |
| `MLPTrainer` | `DLTrainerPort` | `tabular` | Verified (PR gate) |
| `TabPFNPredictor` | `TabFMPort` | `tabfm` | Gated; runs with the extra |
| `TorchTabularTrainer` | `DLTrainerPort` | `dl` | Integration seam; nightly suite |

Start with `MLPTrainer` for a dependency-light neural baseline, reach for `TabPFNPredictor` on small/medium tables where a foundation model shines, and use `TorchTabularTrainer` as the entry point for full PyTorch-based deep learning.

## See also

- [Datasets](./datasets.md) — the `Dataset` container and `DatasetLoaderPort`
- [Models](automl.md) — the fitted `Model` wrapper and `TrainerPort`
- [Preprocessing](automl.md) — `build_pipeline`, shared by every adapter
- [Core Types](configuration.md) — `TaskType` and friends
