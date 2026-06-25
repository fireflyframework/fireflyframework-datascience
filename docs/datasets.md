# Datasets

**A small, dependency-light container for tabular data — plus pluggable loaders for getting it.**

The `datasets` module gives you one thing to pass around: a `Dataset` holding features `X`, an
optional target `y`, the `task`, and metadata. The module itself is import-light (pandas and
scikit-learn are imported lazily), so the `Dataset` type and the `DatasetLoaderPort` protocol are
usable without the `tabular` extra installed. Concrete loaders live in
`fireflyframework_datascience.datasets.adapters`.

## The `Dataset` container

`Dataset` is a dataclass. The only required fields are `name` and `X`.

```python
from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset

ds = Dataset(
    name="my_data",
    X=frame_x,                 # a pandas DataFrame (or array-like)
    y=series_y,                # optional target; defaults to None
    task=TaskType.BINARY,      # defaults to TaskType.CLASSIFICATION
    target_name="label",
    feature_names=list(frame_x.columns),
    metadata={"source": "csv"},
)

ds.n_rows       # int — len(X)
ds.n_features   # int — X.shape[1]
ds.has_target   # bool — y is not None
```

### `train_test_split`

Splits into `(train, test)` datasets. For classification tasks with a target present, the split is
stratified on `y` automatically. The returned datasets carry the same `task`, `target_name`,
`feature_names`, and a copy of `metadata`; their names are suffixed `[train]` / `[test]`.

```python
train, test = ds.train_test_split(test_size=0.25, random_state=42)

train.name   # "my_data[train]"
test.name    # "my_data[test]"
```

Both arguments are keyword-only with the defaults shown above.

### `with_features`

Returns a copy with the feature matrix `X` replaced (used by feature engineering). The new
`feature_names` are taken from the DataFrame's columns; `y` and the rest are preserved.

```python
engineered = ds.with_features(new_frame_x)
engineered.feature_names == list(new_frame_x.columns)  # True
```

## `infer_task`

Infers a `TaskType` from a target series or array:

```python
from fireflyframework_datascience.datasets import infer_task

infer_task([0, 1, 0, 1])              # TaskType.BINARY        (2 unique values)
infer_task(["a", "b", "c"])           # TaskType.MULTICLASS    (>2 categorical)
infer_task([0.1, 0.2, ... , 3.4])     # TaskType.REGRESSION    (float, >20 unique)
```

The rules: float or integer targets with more than 20 distinct values are `REGRESSION`; exactly two
unique values are `BINARY`; everything else is `MULTICLASS`.

## `DatasetLoaderPort`

A `@runtime_checkable` `Protocol`. Implement it to teach the framework about a new data source.

```python
from typing import Any
from fireflyframework_datascience.datasets import Dataset, DatasetLoaderPort

class DatasetLoaderPort(Protocol):
    name: str
    def can_load(self, source: str) -> bool: ...
    def load(self, source: str, *, target: str | None = None, **kwargs: Any) -> Dataset: ...
```

A loader inspects a string `source` (a name, an id, or a URI), reports whether it `can_load` it, and
returns a fully-populated `Dataset`.

## `SklearnDatasetLoader` (offline)

Loads scikit-learn's built-in toy/real datasets by name — no network needed. Accepts bare names
(`breast_cancer`) or the `sklearn:` prefix (`sklearn:breast_cancer`).

```python
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

loader = SklearnDatasetLoader()
loader.name                       # "sklearn"
loader.can_load("iris")           # True
loader.can_load("openml:31")      # False

ds = loader.load("breast_cancer")
ds.task          # TaskType.BINARY
ds.n_features    # 30
ds.metadata      # {"source": "sklearn", "n_rows": ..., "n_features": ...}
```

Available names and their tasks:

| Source | Task |
| --- | --- |
| `breast_cancer` | `BINARY` |
| `iris`, `wine`, `digits` | `MULTICLASS` |
| `diabetes`, `california_housing` | `REGRESSION` |

An unknown name raises `ValueError` listing the available datasets.

## `OpenMLDatasetLoader` (`data` extra)

Loads datasets from [OpenML](https://www.openml.org/) by id (`openml:31`) or name
(`openml:credit-g`). Requires the `data` extra (`openml`) and network access. The `task` is inferred
via `infer_task` from the resolved target.

```bash
pip install "fireflyframework-datascience[data]"
```

```python
from fireflyframework_datascience.datasets.adapters import OpenMLDatasetLoader

loader = OpenMLDatasetLoader()
loader.name                       # "openml"
loader.can_load("openml:31")      # True

ds = loader.load("openml:credit-g")          # by name
ds = loader.load("openml:31", target="class") # override the default target
ds.metadata                                   # {"source": "openml", "openml_id": ...}
```

If the `openml` package is not installed, `load` raises `AdapterUnavailableError("OpenMLDatasetLoader", "data")`.

## See also

- [Core types](architecture.md) — `TaskType` and the rest of the core enums
- [Adapters](architecture.md) — the adapter pattern and the `data` / `tabular` extras
- [Feature engineering](genai-features.md) — consumers of `Dataset.with_features`
- [Getting started](quickstart.md)
