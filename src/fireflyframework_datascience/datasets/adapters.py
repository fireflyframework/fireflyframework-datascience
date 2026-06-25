# Copyright 2026 Firefly Software Foundation.
"""Dataset loader adapters (require the ``tabular`` / ``data`` extras)."""

from __future__ import annotations

from typing import Any

from fireflyframework_datascience.core.exceptions import AdapterUnavailableError
from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset, infer_task

# sklearn loader name -> (loader callable name, task)
_SKLEARN_DATASETS: dict[str, TaskType] = {
    "breast_cancer": TaskType.BINARY,
    "iris": TaskType.MULTICLASS,
    "wine": TaskType.MULTICLASS,
    "digits": TaskType.MULTICLASS,
    "diabetes": TaskType.REGRESSION,
    "california_housing": TaskType.REGRESSION,
}


class SklearnDatasetLoader:
    """Loads scikit-learn's built-in toy/real datasets by name (no network).

    Sources are bare names (``breast_cancer``) or prefixed (``sklearn:breast_cancer``).
    """

    name = "sklearn"

    def _key(self, source: str) -> str:
        return source.removeprefix("sklearn:").strip()

    def can_load(self, source: str) -> bool:
        return self._key(source) in _SKLEARN_DATASETS

    def load(self, source: str, *, target: str | None = None, **kwargs: Any) -> Dataset:
        import sklearn.datasets as skd

        key = self._key(source)
        if key not in _SKLEARN_DATASETS:
            raise ValueError(f"Unknown sklearn dataset {source!r}. Available: {sorted(_SKLEARN_DATASETS)}")
        loader = getattr(skd, f"load_{key}", None) or getattr(skd, f"fetch_{key}")
        bunch = loader(as_frame=True)
        frame_x = bunch.data
        series_y = bunch.target
        return Dataset(
            name=key,
            X=frame_x,
            y=series_y,
            task=_SKLEARN_DATASETS[key],
            target_name=getattr(series_y, "name", "target"),
            feature_names=list(frame_x.columns),
            metadata={"source": "sklearn", "n_rows": len(frame_x), "n_features": frame_x.shape[1]},
        )


class OpenMLDatasetLoader:
    """Loads datasets from OpenML by id (``openml:31``) or name (``openml:credit-g``).

    Requires the ``data`` extra (``openml``) and network access.
    """

    name = "openml"

    def can_load(self, source: str) -> bool:
        return source.startswith("openml:")

    def load(self, source: str, *, target: str | None = None, **kwargs: Any) -> Dataset:
        try:
            import openml
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise AdapterUnavailableError("OpenMLDatasetLoader", "data") from exc

        ref = source.removeprefix("openml:").strip()
        dataset = openml.datasets.get_dataset(int(ref)) if ref.isdigit() else openml.datasets.get_dataset(ref)
        default_target = target or dataset.default_target_attribute
        data = dataset.get_data(target=default_target, dataset_format="dataframe")
        frame_x: Any = data[0]
        series_y: Any = data[1]
        task = infer_task(series_y) if series_y is not None else TaskType.CLASSIFICATION
        return Dataset(
            name=dataset.name,
            X=frame_x,
            y=series_y,
            task=task,
            target_name=default_target,
            feature_names=list(frame_x.columns),
            metadata={"source": "openml", "openml_id": dataset.dataset_id},
        )


__all__ = ["OpenMLDatasetLoader", "SklearnDatasetLoader"]
