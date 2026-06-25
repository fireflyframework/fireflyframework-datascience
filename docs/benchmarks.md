# Benchmarks & Datasets

**A three-tier evaluation strategy — credible public benchmarks, fast CI smoke datasets, and agentic capability suites — fed by pluggable dataset loaders, with every published number produced by a bundled, runnable harness.**

Firefly DataScience separates *how we prove the framework is good* from *how we load data day-to-day*. The same `DatasetLoaderPort` that powers a quick `iris` smoke test in CI also pulls real OpenML benchmark suites for credibility runs. This page describes the evaluation strategy, shows how to load datasets through the loaders, and lists the real, reproducible results. Every figure here was produced by running a script in `benchmarks/` with no manual tuning — see the full table in [`benchmarks/RESULTS.md`](https://github.com/fireflyframework/fireflyframework-datascience/blob/main/benchmarks/RESULTS.md).

!!! firefly "The recurring thesis — the LLM proposes; the classical engine decides"

    GenAI proposes feature code; a deterministic classical engine measures the cross-validated lift;
    and a **cost/benefit gate keeps only what is proven on the data**. That is why the GenAI ablation
    below can only improve or stay neutral — never regress. The benchmarks measure both the classical
    core and the gated accelerator on the same footing.

## The three tiers

| Tier | Purpose | Sources | When it runs |
|------|---------|---------|--------------|
| **Tier 1 — Credibility** | Compare against the literature on standard suites | AMLB, OpenML‑CC18, OpenML‑CTR23 | Offline / scheduled (network) |
| **Tier 2 — CI smoke** | Fast, deterministic, no-network correctness | `breast_cancer`, `iris`, `wine`, `digits`, `diabetes`, `california_housing` | Every PR |
| **Tier 3 — Agentic capability** | Measure end-to-end agent problem solving | MLE‑bench, DSBench | Periodic, sandboxed |

Tier 2 is the only tier that runs without network access, which is why it backs the default CI gate. Tiers 1 and 3 are the *roadmap* — they define how the framework earns external credibility over time.

## Loading datasets

Two loaders ship today, both implementing `DatasetLoaderPort` (`name`, `can_load`, `load`).

=== "Tier 2 — scikit-learn (offline, no network)"

    `SklearnDatasetLoader` resolves bare names or `sklearn:`-prefixed names against scikit-learn's
    bundled datasets. No download, fully deterministic.

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

    Each `load` returns a `Dataset` dataclass with `X`, `y`, `task`, `target_name`, `feature_names`,
    and a `metadata` dict (`source`, `n_rows`, `n_features`).

=== "Tier 1 — OpenML (benchmark suites, network)"

    `OpenMLDatasetLoader` fetches by numeric id or by name using the `openml:` prefix. It needs the
    `data` extra (`openml`) and network access; without the extra it raises `AdapterUnavailableError`.

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

    OpenML dataset ids are how Tier 1 suites (OpenML‑CC18, OpenML‑CTR23, AMLB) are addressed — each
    suite is a curated set of these ids, so a credibility run is "load each id, fit, score, compare".

    Install the extra:

    ```bash
    pip install "fireflyframework-datascience[data]"
    ```

### Working with a loaded `Dataset`

`Dataset` carries split and feature helpers used by the rest of the framework:

```python
ds = SklearnDatasetLoader().load("iris")

train, test = ds.train_test_split(test_size=0.25, random_state=42)  # (1)!
print(train.name, test.name)            # iris[train] iris[test]

ds.has_target                           # True
ds.task.is_classification()             # True
```

1.  Classification targets are stratified automatically, so each split preserves the class balance —
    important on the small (~1000-row) datasets used in the Tier‑1 runs below.

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

These are produced by running the harnesses — fixed `random_state=0`, default trainers, no manual tuning. Full table and reproduction steps: [`benchmarks/RESULTS.md`](https://github.com/fireflyframework/fireflyframework-datascience/blob/main/benchmarks/RESULTS.md).

To reproduce locally:

```bash
uv sync --extra tabular --extra data --extra validation
uv run python benchmarks/automl_benchmark.py     # Tier-2 (offline, no network)
uv run python benchmarks/amlb_benchmark.py        # Tier-1 (OpenML, needs network)
```

!!! success "Expected — Tier-2 offline suite (`automl_benchmark.py`)"

    `AutoML(cv=3)` over the default trainers (RandomForest, Linear, HistGradientBoosting; + XGBoost /
    LightGBM / CatBoost when installed). Runs in seconds, no network.

    | Dataset | Task | Metric | CV | Holdout | Winner | Seconds |
    |---|---|---|---:|---:|---|---:|
    | breast_cancer | binary | roc_auc | 0.9939 | **0.9952** | linear | 1.8 |
    | iris | multiclass | accuracy | 0.9467 | **1.0000** | random_forest | 1.6 |
    | wine | multiclass | accuracy | 0.9700 | **1.0000** | linear | 1.0 |
    | diabetes | regression | rmse | −54.10 | **56.46** | linear | 1.4 |
    | california_housing | regression | rmse | −0.473 | **0.455** | hist_gradient_boosting | 9.0 |

### Tier-1 — OpenML-CC18 (AMLB-style)

`amlb_benchmark.py` runs `AutoML(cv=5)` across real OpenML tasks with genuine categorical data (e.g. `credit-g`), exercising the dtype-aware preprocessing and string-target encoding. Holdout ROC-AUC:

| OpenML id | Dataset | CV | Holdout | Winner |
|---:|---|---:|---:|---|
| 31 | credit-g | 0.7689 | **0.825** | random_forest |
| 37 | diabetes | 0.8155 | **0.872** | linear |
| 1464 | blood-transfusion | 0.7465 | **0.751** | linear |
| 1480 | ilpd | 0.7347 | **0.780** | linear |

Comparable to published AutoGluon / H2O / FLAML numbers on the same datasets — out of the box, on real data with categorical features.

!!! note "On real finance & retail data (`samples/industry_showcase.py`)"

    German credit risk (`credit-g`) reaches **0.82** holdout ROC-AUC and bank-marketing campaign
    conversion reaches **0.92** — each a full load → validate → AutoML → evaluate run on public OpenML
    data, no Kaggle account required.

### Unbiased comparison — nested cross-validation

`benchmarks/scientific_eval.py` uses **nested 5-fold CV** (inner CV selects the model on each outer fold's *training* data only; the untouched outer fold gives the unbiased estimate) to compare Firefly AutoML against fixed single models on identical folds, with a one-sided Wilcoxon signed-rank test over all 25 paired deltas (5 folds × 5 datasets):

| Firefly AutoML vs… | mean Δ ROC-AUC | wins / ties / losses | Wilcoxon p |
|---|---:|---|---:|
| LogReg (linear) | **+0.029** | 8 / 14 / 3 | **0.046** |
| RandomForest | +0.012 | 16 / 2 / 7 | 0.051 (on par) |
| XGBoost | **+0.030** | 22 / 1 / 2 | **7.5e-6** |

Firefly **significantly beats** single LogReg and single XGBoost and is **statistically on par with** RandomForest — because it *adapts* per dataset (boosting/bagging on non-linear data like `phoneme`, linear where linear genuinely wins, e.g. `blood-transfusion` and `ilpd`). On 2 of 5 small datasets a fixed model edges it out by ~0.01–0.02 (selection variance on ~1000-row data) — reported honestly.

!!! note "Why nested CV"

    An AutoML system that reports the cross-validated score of the model it *selected* is
    optimistically biased — it is the maximum over many models scored on the same folds. Nested CV
    removes that bias: model selection happens on the inner CV of each outer fold's training data, and
    the outer fold — never seen during selection — gives the honest estimate.

### GenAI value — controlled ablation (real LLM)

`benchmarks/genai_value.py` isolates the GenAI contribution on a retail "high-value customer" task whose true driver (`revenue = unit_price × units`) is withheld from the model — a multiplicative interaction a *linear* learner cannot derive on its own. Four systems, 8 repeated train/test splits, real `anthropic:claude-haiku-4-5`:

| System | ROC-AUC (mean ± std) |
|---|---:|
| linear (raw) | 0.9752 ± 0.006 |
| **linear + GenAI** | **0.9957 ± 0.002** |
| Firefly AutoML (raw) | 0.9929 ± 0.003 |
| Firefly AutoML + GenAI | 0.9950 ± 0.003 |

GenAI feature engineering lifts the **linear model by +0.0205 ROC-AUC** (0.975 → 0.996, **Wilcoxon p = 0.0039**) — Claude proposed and the gate accepted `total_revenue` / `price_volume_ratio`, rediscovering the withheld multiplicative driver from the schema alone. On Firefly's tree-based AutoML the lift is smaller (+0.002): trees already approximate the interaction, so there is less to add — and the **gate guarantees no regression**. Cost: 8 LLM calls, **well under $0.01** with Claude Haiku.

!!! tip "Pareto-safe accelerator"

    GenAI feature engineering adds measurable, significant value where the data has structure a model
    cannot reach on its own, surfaces interpretable domain features, and is gated to never hurt — at
    negligible cost. See [GenAI features](genai-features.md) and the [agentic loop](agentic-loop.md)
    for the propose-measure-gate mechanism.

## See also

- [Datasets API](datasets.md)
- [Classical AutoML](automl.md)
- [GenAI features](genai-features.md)
- [Configuration](configuration.md)
- [Getting started](quickstart.md)
