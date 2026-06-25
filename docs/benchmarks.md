# Benchmarks & Datasets

**A three-tier evaluation strategy: credible public benchmarks, fast CI smoke datasets, and agentic capability suites — fed by pluggable dataset loaders.**

Firefly DataScience separates *how we prove the framework is good* from *how we load data day-to-day*. The same `DatasetLoaderPort` that powers a quick `iris` smoke test in CI also pulls real OpenML benchmark suites for credibility runs. This page describes the evaluation roadmap and shows how to load datasets through the loaders.

## The three tiers

| Tier | Purpose | Sources | When it runs |
|------|---------|---------|--------------|
| **Tier 1 — Credibility** | Compare against the literature on standard suites | AMLB, OpenML‑CC18, OpenML‑CTR23 | Offline / scheduled (network) |
| **Tier 2 — CI smoke** | Fast, deterministic, no-network correctness | `breast_cancer`, `iris`, `wine`, `digits`, `diabetes`, `california_housing` | Every PR |
| **Tier 3 — Agentic capability** | Measure end-to-end agent problem solving | MLE‑bench, DSBench | Periodic, sandboxed |

Tier 2 is the only tier that runs without network access, which is why it backs the default CI gate. Tiers 1 and 3 are the *roadmap* — they define how the framework earns external credibility over time.

## Loading datasets

Two loaders ship today, both implementing `DatasetLoaderPort` (`name`, `can_load`, `load`).

### Tier 2 — scikit-learn (offline, no network)

`SklearnDatasetLoader` resolves bare names or `sklearn:`-prefixed names against scikit-learn's bundled datasets. No download, fully deterministic.

```python
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

loader = SklearnDatasetLoader()
loader.can_load("breast_cancer")        # True
loader.can_load("sklearn:diabetes")     # True (prefix is stripped)

ds = loader.load("breast_cancer")
print(ds.name, ds.task, ds.n_rows, ds.n_features)
# breast_cancer TaskType.BINARY 569 30
```

The built-in Tier 2 names map to fixed task types:

```python
# binary       -> breast_cancer
# multiclass   -> iris, wine, digits
# regression   -> diabetes, california_housing
```

Each `load` returns a `Dataset` dataclass with `X`, `y`, `task`, `target_name`, `feature_names`, and a `metadata` dict (`source`, `n_rows`, `n_features`).

### Tier 1 — OpenML (benchmark suites, network)

`OpenMLDatasetLoader` fetches by numeric id or by name using the `openml:` prefix. It needs the `data` extra (`openml`) and network access; without the extra it raises `AdapterUnavailableError`.

```python
from fireflyframework_datascience.datasets.adapters import OpenMLDatasetLoader

loader = OpenMLDatasetLoader()
loader.can_load("openml:31")            # True
loader.can_load("breast_cancer")        # False (no openml: prefix)

ds = loader.load("openml:31")           # by id, e.g. the 'credit-g' task
ds = loader.load("openml:credit-g")     # by name
ds = loader.load("openml:31", target="class")  # override the default target

print(ds.metadata["openml_id"], ds.task)
```

OpenML dataset ids are how Tier 1 suites (OpenML‑CC18, OpenML‑CTR23, AMLB) are addressed — each suite is a curated set of these ids, so a credibility run is "load each id, fit, score, compare".

Install the extra:

```bash
pip install "fireflyframework-datascience[data]"
```

### Working with a loaded `Dataset`

`Dataset` carries split and feature helpers used by the rest of the framework:

```python
ds = SklearnDatasetLoader().load("iris")

train, test = ds.train_test_split(test_size=0.25, random_state=42)
# classification targets are stratified automatically
print(train.name, test.name)            # iris[train] iris[test]

ds.has_target                           # True
ds.task.is_classification()             # True
```

When the target is unknown (OpenML without a declared task type), the loader infers it:

```python
from fireflyframework_datascience.datasets import infer_task

infer_task([0, 1, 1, 0])                # TaskType.BINARY
infer_task([0.1, 2.3, 9.9, 4.2, ...])   # TaskType.REGRESSION (float, many uniques)
```

## Auto-configuration

When scikit-learn is on the path, the loaders are registered as beans automatically — no manual wiring. The OpenML bean only appears when `openml` is also importable.

```python
# DatasetsAutoConfiguration registers:
#   sklearn_dataset_loader  (conditional_on_class "sklearn")
#   openml_dataset_loader   (conditional_on_class "openml")
```

Both beans are typed as `DatasetLoaderPort`, so downstream code can depend on the port and let `can_load` route a source string to the right loader.

## Tier 3 — agentic capability (roadmap)

Tier 3 measures the *agent*, not a single estimator: given a task description and raw data, can the system produce a working, scoring solution end to end? The target suites are **MLE‑bench** and **DSBench**. These run in a sandbox on a periodic schedule rather than per-PR. As they land, they reuse the same `DatasetLoaderPort` contract — a new loader (e.g. a `mlebench:` adapter) plugs in exactly like `SklearnDatasetLoader` and `OpenMLDatasetLoader` without changing callers.

## Results (real, executed)

These are produced by running the harnesses — fixed `random_state=0`, default trainers, no manual
tuning. Full table and reproduction steps: [`benchmarks/RESULTS.md`](https://github.com/fireflyframework/fireflyframework-datascience/blob/main/benchmarks/RESULTS.md).

**Tier-1 — OpenML-CC18 (AMLB-style), holdout ROC-AUC:**

| credit-g | diabetes | blood-transfusion | ilpd |
|---:|---:|---:|---:|
| 0.825 | 0.872 | 0.751 | 0.780 |

Comparable to published AutoGluon / H2O / FLAML numbers on the same datasets — out of the box, on real
data with categorical features.

**On real finance & retail data** (`samples/industry_showcase.py`): German credit risk (`credit-g`)
reaches **0.82** holdout ROC-AUC and bank-marketing campaign conversion reaches **0.92** — each a full
load → validate → AutoML → evaluate run on public OpenML data, no Kaggle account required.

### Unbiased comparison — nested cross-validation

`benchmarks/scientific_eval.py` uses **nested 5-fold CV** (inner CV selects the model; the untouched
outer fold gives the unbiased estimate) to compare Firefly AutoML against fixed single models on
identical folds, with a Wilcoxon signed-rank test:

| Firefly AutoML vs… | mean Δ ROC-AUC | Wilcoxon p |
|---|---:|---:|
| LogReg (linear) | **+0.029** | **0.046** |
| RandomForest | +0.012 | 0.051 (on par) |
| XGBoost | **+0.030** | **7.5e-6** |

Firefly **significantly beats** single LogReg and single XGBoost and is **statistically on par with**
RandomForest — because it *adapts* per dataset (boosting on non-linear data, linear where linear wins).
On 2 of 5 small datasets a fixed model edges it out by ~0.01 (selection variance) — reported honestly.

### GenAI value — controlled ablation (real LLM)

`benchmarks/genai_value.py` isolates the GenAI contribution on a retail task whose driver
(`revenue = price × units`) is withheld. Over 8 splits with `anthropic:claude-haiku-4-5`, GenAI feature
engineering lifts a **linear model by +0.0205 ROC-AUC** (0.975 → 0.996, **Wilcoxon p = 0.0039**) — Claude
rediscovered `total_revenue` from the schema alone. On Firefly's tree-based AutoML the lift is smaller
(+0.002) and the **gate guarantees no regression**. Cost: 8 calls, **< $0.01**. GenAI is a *Pareto-safe
accelerator* — significant value where structure exists, never a regression.

## See also

- [Datasets API](./datasets.md)
- [Container & auto-configuration](index.md)
- [Task types](index.md)
- [Getting started](quickstart.md)
