# Changelog

All notable changes to `fireflyframework-datascience` are documented here. The project uses CalVer
(`YY.MM.PATCH`).

## [Unreleased]

### SP1 — Hexagonal ports + classical tabular core

The classical AutoML heart: real, benchmarkable tabular AutoML behind swappable ports.

- **Ports** (`@runtime_checkable` Protocols): `DatasetLoaderPort`, `TrainerPort`, `MetricsEvaluatorPort`,
  `SearchPolicyPort`, `ValidatorPort`, `TrackerPort` / `RegistryPort`, `AutoMLBackendPort`.
- **Datasets** — `Dataset` container (+ stratified `train_test_split`, `infer_task`); `SklearnDatasetLoader`
  (offline toy/real sets) and `OpenMLDatasetLoader` (`data` extra).
- **Trainers** — RandomForest, Linear (LogReg/Ridge), HistGradientBoosting always; XGBoost, LightGBM,
  CatBoost when installed. Each declares a declarative hyperparameter `ParamSpace`.
- **Evaluation** — `SklearnMetricsEvaluator` (accuracy/f1/precision/recall/roc_auc/log_loss; rmse/mae/r2)
  with task-aware default metrics and CV scoring names.
- **Search** — `DefaultSearchPolicy` (evaluate defaults) and `OptunaSearchPolicy` (seeded TPE — the LLM
  may propose seeds/bounds, but classical HPO owns the search).
- **Validation** — `BasicValidator` (nulls/constants/duplicates/target checks); optional `PanderaValidator`.
- **Tracking** — `NoOpTracker` default; `MLflowTracker` (`tracking` extra).
- **AutoML facade** — `AutoML.fit/predict` with leaderboard + cross-validation selection +
  impute/scale/one-hot preprocessing; usable imperatively (`AutoML().fit`) and declaratively
  (`AutoML.from_context(app)`). All adapters auto-wired via the `firefly_datascience.auto_configuration`
  entry-point group.

Gate: ruff clean, pyright 0 errors, 64 tests passing; real end-to-end AutoML on breast-cancer
(roc_auc ≈ 0.98 holdout) and diabetes regression.

## [26.6.0] — 2026-06-25

### SP0 — Foundation & Firefly DNA

The first slice: a self-contained, Firefly-native foundation that boots like pyfly and depends on
`fireflyframework-agentic` for the GenAI substrate.

- **Application bootstrap** — `FireflyDataScienceApplication` with the pyfly-style lifecycle
  (banner → config → DI container → entry-point auto-configuration discovery → condition evaluation →
  eager singleton init → wiring summary → ready `ApplicationContext`).
- **Dependency injection** — a lean, type-hint-driven `Container` (constructor & factory injection,
  singleton/transient scopes, `resolve` / `resolve_all` / interface binding, `@order` sorting,
  circular-dependency detection, `@primary` disambiguation).
- **Conditional auto-configuration** — `@auto_configuration`, `@configuration`, `@bean`, `@component`,
  and `@conditional_on_class` / `@conditional_on_property` / `@conditional_on_missing_bean` /
  `@conditional_on_bean`, discovered via the `firefly_datascience.auto_configuration` entry-point group.
- **Configuration** — `FireflyDataScienceConfig` (pydantic-settings) with env (`FIREFLY_DATASCIENCE_*`,
  nested via `__`), `.env`, and YAML + profile-overlay precedence.
- **Startup banner** — `BannerPrinter` (`TEXT` / `MINIMAL` / `OFF`) + teal `assets/banner.svg`.
- **CLI** — `firefly-ds` with `version`, `doctor` (environment & adapter-extra check), and `introspect`.
- **Engineering DNA** — uv + hatchling (src layout), CalVer, ruff + pyright, pytest (`asyncio_mode=auto`,
  `nightly`/`integration` markers), CI workflow, Apache-2.0 headers, `Copyright 2026 Firefly Software
  Foundation`.

Quality gate at release: ruff clean, pyright 0 errors, 36 tests passing (88% coverage).
