# Deep Learning & Tabular Foundation Models

**Neural and tabular-foundation-model training behind two import-light ports — with a verified sklearn reference and gated PyTorch / TabPFN adapters that share the exact contract of the classical engine.**

The `dl` module defines two ports for non-classical-ML training. `DLTrainerPort` covers neural trainers; `TabFMPort` covers tabular foundation models (in-context fit/predict, e.g. TabPFN). Both are runtime-checkable `Protocol`s and share the same shape as the rest of the framework: `name`, `supports(task)`, and `fit(dataset) -> Model`.

That parity is the point. A deep-learning adapter is not a special case — it is the same `supports` + `fit` seam the classical trainers expose, so AutoML can rank a `MLPTrainer` against a gradient-boosted tree without any extra wiring. The framework's discipline still holds across modalities:

!!! firefly "The LLM proposes; the classical engine decides"
    Deep learning, text, and vision adapters widen *what* can be proposed — neural nets, transformers, CNNs, in-context foundation models. They do not change *who decides*. Every adapter returns a measured `Model`, and the same cost-benefit gate that scores classical models scores these, on held-out data. A heavyweight neural trainer earns its place only when it beats the dependency-light baseline on the metric, not because it is fashionable.

The module itself is import-light — it pulls in no heavy dependencies at import time. A verified `MLPTrainer` (scikit-learn) ships as the reference adapter and needs only the `tabular` extra. The heavy adapters (`TabPFNPredictor`, `TorchTabularTrainer`) are **gated behind extras** and raise a clear error when those extras are missing.

## The ports

Both live in `fireflyframework_datascience.dl`:

```python
from fireflyframework_datascience.dl import DLTrainerPort, TabFMPort
from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.models import Model


class DLTrainerPort(Protocol):       # (1)!
    name: str
    def supports(self, task: TaskType) -> bool: ...
    def fit(self, dataset: Dataset) -> Model: ...


class TabFMPort(Protocol):           # (2)!
    name: str
    def supports(self, task: TaskType) -> bool: ...
    def fit(self, dataset: Dataset) -> Model: ...
```

1. Neural trainers. The verified sklearn-MLP reference ships here; PyTorch Lightning / HuggingFace adapters plug in behind the `dl` / `nlp` extras.
2. Tabular foundation models — in-context fit/predict (e.g. TabPFN), behind the `tabfm` extra.

Because they are `@runtime_checkable`, any object with the right attributes satisfies them — no base class, no registration:

```python
from fireflyframework_datascience.dl import DLTrainerPort
from fireflyframework_datascience.dl.adapters import MLPTrainer

assert isinstance(MLPTrainer(), DLTrainerPort)  # True
```

This is the same structural-typing trick the classical `TrainerPort` uses, which is why a DL adapter slots into the engine with zero glue code.

## MLPTrainer — the verified neural reference

`MLPTrainer` wraps scikit-learn's `MLPClassifier` / `MLPRegressor` in the standard preprocessing pipeline and returns a fitted `Model`. It is verified in the PR gate and needs only the `tabular` extra — no PyTorch required. The estimator is built with `build_pipeline(...)`, so it shares the exact preprocessing every other trainer gets.

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
trainer.name                   # "mlp"
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

Under the hood the network is fixed and reproducible — `hidden_layer_sizes=(64, 32)`, `max_iter=400`, `random_state=42` — chosen as a sensible baseline rather than a tuning target.

The returned `Model` is the same wrapper the rest of the framework uses, so you get `predict` / `predict_proba` / `save` / `load` for free.

## TabPFNPredictor — tabular foundation model (`tabfm` extra)

`TabPFNPredictor` implements `TabFMPort` over [TabPFN](https://github.com/PriorLabs/TabPFN): an in-context transformer that fits and predicts without conventional gradient training. It is **gated behind the `tabfm` extra**.

```bash
pip install "fireflyframework-datascience[tabfm]"
```

```python
from fireflyframework_datascience.dl.adapters import TabPFNPredictor

predictor = TabPFNPredictor()
predictor.name                        # "tabpfn"
predictor.supports(TaskType.BINARY)   # True

model = predictor.fit(dataset)        # in-context fit -> Model
preds = model.predict(dataset.X)
```

Like `MLPTrainer`, it dispatches on the task — `TabPFNClassifier` for classification, `TabPFNRegressor` for regression — and wraps the result through the same `build_pipeline` / `Model` path, so the foundation model is just another scored candidate from AutoML's point of view.

If the `tabfm` extra is not installed, `fit` raises a clear, actionable error rather than an opaque `ImportError`:

```python
from fireflyframework_datascience.core.exceptions import AdapterUnavailableError

try:
    TabPFNPredictor().fit(dataset)
except AdapterUnavailableError as exc:
    # names the adapter ("TabPFNPredictor") and the missing extra ("tabfm")
    print(exc)
```

!!! success "Expected"
    ```text
    Adapter 'TabPFNPredictor' requires the optional dependency group 'tabfm'.
    Install it with: pip install 'fireflyframework-datascience[tabfm]'
    ```
    The exception carries `.adapter` and `.extra` attributes too, so callers can branch programmatically instead of parsing the message.

## TorchTabularTrainer — the PyTorch / Lightning integration point (`dl` extra)

`TorchTabularTrainer` is the `DLTrainerPort` adapter where full deep-learning workloads plug in: PyTorch Lightning, HuggingFace Accelerate, distributed training (FSDP/DDP), and PEFT/TRL all share this contract. It is **gated behind the `dl` extra** and verified under the nightly/integration suite rather than the PR gate.

```bash
pip install "fireflyframework-datascience[dl]"
```

```python
from fireflyframework_datascience.dl.adapters import TorchTabularTrainer

trainer = TorchTabularTrainer(epochs=50, hidden=64, lr=1e-3)   # (1)!
trainer.name                            # "torch_tabular"
trainer.supports(TaskType.REGRESSION)   # True
```

1. These are the constructor defaults — `epochs=50`, `hidden=64`, `lr=1e-3` — surfaced explicitly here for clarity. With the `dl` extra present, `fit` builds the preprocessor, encodes labels, and trains a small MLP via the bundled torch implementation.

!!! warning "Gated behind extras"
    Without `torch` installed, `fit` raises `AdapterUnavailableError("TorchTabularTrainer", "dl")`. The training loop runs only with the `dl` extra, under the nightly suite — not the PR gate.

Treat it as the **integration seam**: the contract (`supports` + `fit(dataset) -> Model`) is identical to `MLPTrainer`, so your own Lightning/HF trainer can drop straight in with no inheritance and no registration:

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

## Beyond tabular: text and vision share the shape

The same ports-parity discipline extends past the `dl` module. Each modality lives in its own import-light package, defines a port plus a result model, and exposes a single `fit(...)` method — the only thing that varies is the input type.

=== "Text (`nlp`)"

    `HFTextClassifier` fine-tunes a HuggingFace sequence-classification model on `(texts, labels)` and returns a `TextModel`. It defaults to DistilBERT and is **gated behind the `nlp` extra** (transformers + torch).

    ```python
    from fireflyframework_datascience.nlp.adapters import HFTextClassifier

    clf = HFTextClassifier()                      # model_name="distilbert-base-uncased"
    clf.name                                      # "hf_text"
    model = clf.fit(["great product", "broke on day one"], ["pos", "neg"])
    model.predict(["love it"])                    # -> ["pos"] (a TextModel)
    ```

    Swap `model_name` for any other sequence-classification checkpoint (RoBERTa, DeBERTa, …). Defaults: `epochs=3`, `lr=5e-5`, `max_length=64`, `batch_size=8`. Without the extra, `fit` raises `AdapterUnavailableError("HFTextClassifier", "nlp")`.

=== "Vision (`dl`)"

    `TorchCNNClassifier` trains a small CNN on `(N, C, H, W)` image arrays and returns an `ImageModel`. It is **gated behind the `dl` extra**.

    ```python
    from fireflyframework_datascience.vision.adapters import TorchCNNClassifier

    clf = TorchCNNClassifier()        # epochs=15, lr=1e-3, batch_size=16
    clf.name                          # "torch_cnn"
    model = clf.fit(images, labels)   # images: (N, C, H, W) float array
    model.predict(images)             # -> list of labels (an ImageModel)
    ```

    Without the `dl` extra, `fit` raises `AdapterUnavailableError("TorchCNNClassifier", "dl")`.

The text and vision ports (`TextClassifierPort`, `ImageClassifierPort`) take `(inputs, labels)` rather than a `Dataset`, because text and image inputs are not the tabular `Dataset` the `dl` ports consume — but the fit/predict rhythm, the gated-extra discipline, and the structural-typing contract are identical.

## Modalities at a glance

The framework names five modalities in the `Modality` enum (`fireflyframework_datascience.core.types`). Three have shipping adapters today; two are reserved in the type system but have no built-in adapter yet — you would supply your own against the relevant port.

| Modality | Port | Result | Reference adapter | Extra | Input |
| --- | --- | --- | --- | --- | --- |
| `TABULAR` | `DLTrainerPort` / `TabFMPort` | `Model` | `MLPTrainer`, `TabPFNPredictor`, `TorchTabularTrainer` | `tabular` / `tabfm` / `dl` | `Dataset` |
| `TEXT` | `TextClassifierPort` | `TextModel` | `HFTextClassifier` | `nlp` | `(texts, labels)` |
| `VISION` | `ImageClassifierPort` | `ImageModel` | `TorchCNNClassifier` | `dl` | `(images, labels)` |
| `TIMESERIES` | — | — | none built in | — | — |
| `MULTIMODAL` | — | — | none built in | — | — |

!!! note "Reserved, not implemented"
    `TIMESERIES` and `MULTIMODAL` exist in the `Modality` enum (and `TaskType.FORECASTING` exists for forecasting tasks), but the published package ships no adapter for them. The enum names the design space; bring your own adapter on the same `supports` + `fit` contract and it will satisfy the port like any other.

## Choosing an adapter

| Adapter | Port | Extra | Status |
| --- | --- | --- | --- |
| `MLPTrainer` | `DLTrainerPort` | `tabular` | Verified (PR gate) |
| `TabPFNPredictor` | `TabFMPort` | `tabfm` | Gated; runs with the extra |
| `TorchTabularTrainer` | `DLTrainerPort` | `dl` | Integration seam; nightly suite |
| `HFTextClassifier` | `TextClassifierPort` | `nlp` | Gated; runs with the extra |
| `TorchCNNClassifier` | `ImageClassifierPort` | `dl` | Gated; runs with the extra |

!!! tip "Where to start"
    Reach for `MLPTrainer` for a dependency-light neural baseline, `TabPFNPredictor` on small/medium tables where a foundation model shines, and `TorchTabularTrainer` as the entry point for full PyTorch-based deep learning. For text or images, start with `HFTextClassifier` and `TorchCNNClassifier`. In every case the adapter only earns its keep if it beats the baseline on the held-out metric — the cost-benefit gate, not the adapter, makes the call.

## See also

- [Datasets](datasets.md) — the `Dataset` container and loader ports
- [AutoML](automl.md) — the fitted `Model` wrapper, `TrainerPort`, and the cost-benefit gate that scores every adapter
- [GenAI features](genai-features.md) — where the LLM proposes and the classical engine decides
- [Configuration](configuration.md) — `TaskType`, `Modality`, and how extras toggle adapters on
- [Architecture](architecture.md) — the hexagonal ports-and-adapters design these trainers plug into
