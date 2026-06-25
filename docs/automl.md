# Classical AutoML

**Cross-validate a panel of tabular models, tune the contenders, and fit the winner — in three lines.**

The `AutoML` engine is the front door of Firefly DataScience. It validates a dataset, cross-validates every
trainer that supports the task (optionally tuning each one), and returns a fitted winner together with the full
leaderboard. It is import-light: scikit-learn is only loaded when you actually call `fit`, so
`from fireflyframework_datascience.automl import AutoML` stays cheap.

!!! firefly "The LLM proposes; the classical engine decides"

    `AutoML` is pure classical machine learning — deterministic, seeded, and reproducible. Where GenAI
    enters elsewhere in the framework, it only ever *proposes* (seeds, bounds, candidate features);
    this engine *decides* by cross-validated score. The search is owned by Optuna and scikit-learn,
    never by a language model.

<p align="center"><img src="img/automl-loop.svg" alt="The AutoML loop: validate, cross-validate candidates, tune, refit the winner" width="100%"></p>

## Quick start

```python
from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

dataset = SklearnDatasetLoader().load("breast_cancer")
train, test = dataset.train_test_split(test_size=0.25)

result = AutoML().fit(train)            # task & metric inferred from the dataset

print(result.best_model.name)           # e.g. "hist_gradient_boosting"
print(result.leaderboard_table())
print(result.evaluate(test))            # EvaluationResult on held-out data
```

`fit` infers the task from `dataset.task` and the default metric from the evaluator (`roc_auc` for binary,
`accuracy` for multiclass, `rmse` for regression). Override either explicitly:

```python
from fireflyframework_datascience.core.types import TaskType

result = AutoML().fit(train, task=TaskType.BINARY, metric="f1")
```

## The fit loop

For each trainer that `supports(task)`, `AutoML`:

1. Builds the trainer's hyperparameter search space — but only when `n_trials > 1`; with `n_trials <= 1`
   the space is empty and the search collapses to a single default-hyperparameter evaluation.
2. Runs the search policy, whose objective wraps the estimator in a preprocessing pipeline and
   cross-validates it (`cross_val_score`, `cv` folds).
3. Records a `LeaderboardEntry` with the best CV score.
4. After all candidates are scored, refits the highest-scoring trainer on the full training data and
   wraps it as a `Model`.

```python
space = trainer.param_space(task) if n_trials > 1 else {}   # (1)!
result = search_policy.optimize(objective, space, n_trials=n_trials, seed=random_state)  # (2)!
leaderboard.append(LeaderboardEntry(trainer.name, dict(result.best_params), result.best_score, metric))
if best is None or result.best_score > best[0]:             # (3)!
    best = (result.best_score, trainer, dict(result.best_params))
```

1. No tuning budget means no space to search — the policy evaluates the estimator's defaults once.
2. The CV objective returns the mean fold score. A candidate that raises during CV is logged and scored
   `-inf`, so one broken estimator never aborts the whole run.
3. Selection is strictly by CV score (greater is better, always). The winning trainer is then refit on
   `dataset.X, dataset.y` inside the same preprocessing pipeline.

The preprocessing pipeline is built automatically from the column dtypes: numeric columns get median
imputation + `StandardScaler`; categorical columns get most-frequent imputation + `OneHotEncoder`
(`handle_unknown="ignore"`). Scaling is harmless for trees and essential for linear models, so the same
pipeline serves every trainer.

## Configuring the engine

```python
from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.models.adapters import (
    RandomForestTrainer,
    HistGradientBoostingTrainer,
    XGBoostTrainer,
)
from fireflyframework_datascience.search.adapters import OptunaSearchPolicy

automl = AutoML(
    trainers=[RandomForestTrainer(), HistGradientBoostingTrainer(), XGBoostTrainer()],
    search_policy=OptunaSearchPolicy(),
    cv=5,
    n_trials=40,
    random_state=42,
)
result = automl.fit(train)
```

The constructor accepts `trainers`, `evaluator`, `search_policy`, `validator`, `tracker`, plus the
`cv`, `n_trials`, and `random_state` knobs (defaults `cv=5`, `n_trials=20`, `random_state=42`). Anything
left as `None` falls back to sensible defaults: `[RandomForestTrainer(), LinearTrainer(), HistGradientBoostingTrainer()]`,
the `SklearnMetricsEvaluator`, and the `DefaultSearchPolicy`. A `validator` and `tracker` stay `None`
unless you supply them — when present, the validator runs first and raises on failure, and the tracker
logs the winner's params, CV score, and model artifact.

!!! tip "`cv` accepts a splitter, not just a fold count"

    `cv` is passed straight to scikit-learn's `cross_val_score`, so beyond an `int` you can hand it any
    splitter to control *how* folds are drawn — and to avoid silent leakage:

    ```python
    from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedKFold

    AutoML(cv=TimeSeriesSplit(n_splits=5))          # temporal data: forward-chaining, no future leakage
    AutoML(cv=StratifiedKFold(5, shuffle=True))     # explicit stratification + shuffling control
    AutoML(cv=GroupKFold(n_splits=5))               # grouped data: keep a group out of train and test
    ```

    The same splitter drives every candidate's cross-validation, so the leaderboard stays comparable.

### Wiring it up

=== "Imperative / notebook"

    Construct the engine by hand and call `fit` directly — ideal for exploration:

    ```python
    automl = AutoML(cv=10, n_trials=50)
    result = automl.fit(train)
    ```

=== "Declarative / DI"

    In an application, resolve the components from a started `ApplicationContext` instead of wiring
    them by hand:

    ```python
    automl = AutoML.from_context(app, cv=10, n_trials=50)
    ```

    `from_context` pulls every registered `TrainerPort` from the container and resolves the optional
    evaluator, search policy, validator, and tracker. Keyword `overrides` win over the resolved
    components, and missing evaluator/search policy fall back to the same defaults as the constructor.

## Trainers

Each trainer is a thin adapter that builds an *unfitted* estimator and declares a `ParamSpace`. All of them
support classification and regression and select the right estimator class per task.

| Trainer | `name` | Estimator (clf / reg) | Extra dependency |
| --- | --- | --- | --- |
| `RandomForestTrainer` | `random_forest` | `RandomForest{Classifier,Regressor}` | — (sklearn) |
| `LinearTrainer` | `linear` | `LogisticRegression` / `Ridge` | — (sklearn) |
| `HistGradientBoostingTrainer` | `hist_gradient_boosting` | `HistGradientBoosting{Classifier,Regressor}` | — (sklearn) |
| `XGBoostTrainer` | `xgboost` | `XGB{Classifier,Regressor}` | `xgboost` |
| `LightGBMTrainer` | `lightgbm` | `LGBM{Classifier,Regressor}` | `lightgbm` |
| `CatBoostTrainer` | `catboost` | `CatBoost{Classifier,Regressor}` | `catboost` |

The boosting-library trainers (`xgboost`, `lightgbm`, `catboost`) import their backend lazily, so you only
pay for the extra you install. A trainer exposes three methods used by the engine:

```python
trainer.supports(task)                 # -> bool
trainer.make_estimator(task, params)   # -> unfitted estimator (sensible defaults merged with params)
trainer.param_space(task)              # -> ParamSpace
```

!!! note "Defaults are baked in, not magic"

    `make_estimator` merges your `params` over each trainer's defaults — for example `RandomForestTrainer`
    starts from `n_estimators=200, n_jobs=-1, random_state=42`, and `HistGradientBoostingTrainer` from
    `learning_rate=0.1, max_iter=200, random_state=42`. The `param_space` only widens the dimensions worth
    tuning (e.g. `n_estimators`, `max_depth`, `max_features` for random forest).

## Search policies

A search policy optimizes the cross-validation objective over a trainer's `ParamSpace`. Scores are always
"greater is better" (the evaluator maps loss-style metrics to negated sklearn scorers).

- **`DefaultSearchPolicy`** (`name="default"`) evaluates the estimator's default hyperparameters once — fast
  and fully deterministic. This is the engine default, and it reports `n_trials=1`.
- **`OptunaSearchPolicy`** (`name="optuna"`) runs seeded Bayesian optimization (TPE). The space spec drives
  the suggestions; if the space is empty it degrades to a single default evaluation (`n_trials=1`).

```python
from fireflyframework_datascience.search.adapters import OptunaSearchPolicy
from fireflyframework_datascience.tuning import IntParam, FloatParam, CategoricalParam

space = {
    "n_estimators": IntParam(100, 500, step=50),
    "max_depth": IntParam(3, 24),
    "max_features": CategoricalParam(("sqrt", "log2", None)),
}
result = OptunaSearchPolicy().optimize(objective, space, n_trials=40, seed=42)
print(result.best_params, result.best_score, result.n_trials)
```

Both policies return a `SearchResult(best_params, best_score, n_trials)`. The seeded TPE sampler
(`TPESampler(seed=seed)`) keeps the search reproducible — classical HPO owns the search, not an LLM.

## Metrics

The default `SklearnMetricsEvaluator` (`name="sklearn"`) supplies CV scoring names and a panel of held-out
metrics:

- **Classification**: `accuracy`, `f1` (weighted), `precision` (weighted), `recall` (weighted), plus
  `roc_auc` and `log_loss` when probabilities are available. For **binary** tasks the panel also reports
  `average_precision` (PR-AUC) and `brier_score` (probability quality / calibration).
- **Regression**: `rmse`, `mae`, `r2`.

```python
ev = result.evaluator
ev.default_metric(result.task)        # "roc_auc" for binary, "accuracy" multiclass, "rmse" regression
ev.scoring_name(result.task, "f1")    # "f1_weighted" (the CV scorer)
ev.greater_is_better("rmse")          # False
```

The leaderboard and CV objective use the *scoring* name, not the raw metric: `f1` maps to the
`f1_weighted` scorer, `rmse` to `neg_root_mean_squared_error`, and binary `roc_auc` stays `roc_auc` while
multiclass `roc_auc` becomes `roc_auc_ovr_weighted`. This is why CV scores are always maximized — a lower
RMSE shows up as a larger (less negative) `neg_root_mean_squared_error`.

!!! tip "Select on PR-AUC for imbalanced binary problems"

    ROC-AUC over-credits a classifier on heavily imbalanced data. Pass `metric="average_precision"` to
    select the winner on **PR-AUC** instead — it is a first-class CV scorer, so the leaderboard, the
    refit winner, and `result.cv_scoring` all reflect it:

    ```python
    result = AutoML().fit(train, metric="average_precision")   # winner chosen by PR-AUC, not accuracy
    ```

!!! tip "Two scores, one winner"

    `result.metric` is the human-facing metric name (e.g. `roc_auc`); `result.cv_scoring` is the sklearn
    scoring string actually used for cross-validation. The leaderboard's `cv_score` is the mean CV score
    under that scorer, and `evaluate(test)` recomputes the full panel on held-out data.

## The `AutoMLResult` API

`fit` returns an `AutoMLResult` carrying the fitted winner, the sorted leaderboard, and the evaluator used:

```python
result.best_model        # Model: name, estimator, task, feature_names, params
result.best_score        # leaderboard[0].cv_score (the top CV score)
result.leaderboard       # list[LeaderboardEntry] sorted best-first
result.metric            # primary metric name
result.task              # TaskType
result.cv_scoring        # sklearn scoring string used during CV

result.predict(test.X)               # winner predictions
result.predict_proba(test.X)         # class probabilities (classification)
eval_result = result.evaluate(test)  # EvaluationResult on a held-out Dataset
print(eval_result.primary_metric, eval_result.primary_value)

print(result.leaderboard_table())    # one line per candidate
```

Each `LeaderboardEntry` holds `model_name`, `params`, `cv_score`, and `metric`, and prints as a tidy
`model_name  metric=score` line (the name is left-padded to 24 columns, the score to 4 decimals).
`best_score` is a property that reads the top entry, so it always agrees with the first row of the table.
`evaluate` automatically passes probabilities through for classification when the winning estimator exposes
`predict_proba`.

!!! success "Expected"

    `result.leaderboard_table()` on the breast-cancer quick-start prints one line per candidate, sorted
    best CV score first. Each line is the `LeaderboardEntry.__str__` format — the trainer `name`
    left-padded to 24 columns, then `metric=score` to 4 decimals (values vary slightly by environment
    and library versions):

    ```text
    hist_gradient_boosting   roc_auc=0.9921
    random_forest            roc_auc=0.9907
    linear                   roc_auc=0.9886
    ```

## Probability calibration

Tree and boosting models often produce over-confident probabilities. For risk- or cost-sensitive
decisions, calibrate the winner so its probabilities are trustworthy — pass `calibrate=True`:

```python
result = AutoML(calibrate=True).fit(train)        # wraps the winner in cross-validated calibration
report = result.evaluate(test)
print(report.metrics["brier_score"])              # lower is better-calibrated
```

`calibrate` wraps the selected classifier in a `CalibratorPort` (default: scikit-learn
`CalibratedClassifierCV`, isotonic, cross-fit) after model selection — classification only, off by
default. The evaluator also reports **`average_precision`** (PR-AUC, important on imbalanced data) and
**`brier_score`** (probability quality) for binary tasks alongside `roc_auc`/`accuracy`.

## Ensembling

Single-best selection leaves accuracy on the table. Pass `ensemble=True` to stack the **top-k**
leaderboard candidates into one model via a cross-fit meta-learner:

```python
result = AutoML(ensemble=True, ensemble_size=3).fit(train)
print(result.best_model.name)              # "stacking_ensemble"
print(result.best_model.params["members"]) # the base learners that were stacked
```

The winner becomes an `EnsemblePort` (default: scikit-learn `StackingClassifier`/`StackingRegressor`
over the top-`ensemble_size` trainers, with a logistic / ridge meta-learner). Off by default;
`ensemble` and `calibrate` compose (the stack can itself be calibrated). DI-resolvable via `from_context`.

## See also

- [Datasets and loaders](datasets.md) — build the `Dataset` you feed to `fit`.
- [GenAI + classical fusion](genai-features.md) — how the LLM proposes and this engine decides.
- [The agentic loop](agentic-loop.md) — the cost-benefit gate around GenAI proposals.
- [Serving the winner](serving.md) — deploy `result.best_model`.
- [Benchmarks](benchmarks.md) — measured leaderboard results on real datasets.
