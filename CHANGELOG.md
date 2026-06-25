# Changelog

All notable changes to `fireflyframework-datascience` are documented here. The project uses CalVer
(`YY.MM.PATCH`).

## [Unreleased]

### SP4 — Deep learning & tabular foundation models

- **`DLTrainerPort` / `TabFMPort`** ports; **`MLPTrainer`** (scikit-learn MLP) is a *verified* neural
  reference. **`TabPFNPredictor`** (`tabfm` extra) and **`TorchTabularTrainer`** (`dl` extra) are gated
  adapters on the same contract — the integration point for PyTorch Lightning + HuggingFace +
  distributed (Accelerate/FSDP/DDP) + PEFT/TRL, verified under the nightly suite, not the PR gate.

### SP5 — Serving, MLOps breadth & the Lumen sample

- **`ModelServerPort`** with a verified in-process **`LocalModelServer`** (default) and a gated
  **`BentoMLModelServer`** (`serving` extra).
- **`LineagePort`** with a `NoOpLineage` default and a gated **`OpenLineageEmitter`** (`lineage` extra).
- **Lumen Lending credit-risk sample** (`samples/lumen_credit_risk.py`) — the end-to-end showcase:
  GenAI feature engineering discovers `debt_to_income` (and the gate rejects a noise feature), classical
  AutoML selects the winner, and the model is served. Runs offline; covered by tests (holdout
  accuracy ≈ 0.85). Demonstrates both the imperative and DI-wired (`AutoML.from_context`) entry points
  for the framework's two audiences.

### SP3 — Agentic ML-engineering loop

The headline agentic capability, grounded on the classical executor: **propose → train/CV → verify →
reflect → select**, with verification as a stage distinct from execution-success.

- **`AgenticAutoML`** — seeds a population (each trainer at defaults), then greedily reflects on the
  attempt history to propose better candidates, bounded by an iteration + patience budget. Every
  candidate is cross-validated by the classical engine; the best *verified* one is refit.
- **`DeterministicVerifier`** — requires a finite score that beats a trivial (Dummy) baseline; a model
  that "ran" but doesn't beat the baseline is rejected (the DS-STAR correctness-≠-ran principle).
- **`AgentSolutionProposer`** — a `FireflyAgent` that reflects on history to propose the next
  (trainer, params); the LLM client is built lazily (the app boots GenAI-on without an API key).
  `SequenceProposer` gives deterministic, LLM-free runs.
- Opt-in via `genai.enabled`; `EngineeringRun` carries the full audited attempt trace.

Gate: ruff clean, pyright 0 errors, 81 tests passing; the loop selects the best verified candidate,
records invalid attempts, and fits a usable model. Also fixed eager LLM-client construction at startup.

### SP2 — GenAI feature engineering (flagship hybrid)

Proves the core thesis in code: **the LLM proposes feature code; classical cross-validation measures
the lift; a `CostBenefitGate` decides.** Built on `fireflyframework-agentic`.

- **`FeatureCodeExecutor`** — vets LLM-generated pandas code with agentic's static safety analysis
  (deny imports/dunder/dangerous builtins), then runs it in a restricted namespace (`df`/`pd`/`np`
  only, minimal builtins allowlist). The CAAFE pattern, secure-by-default.
- **`GenAIFeatureEngineer`** — the propose → execute → measure → gate loop with greedy forward
  acceptance; injectable proposer (fully testable without an LLM) and injectable scorer.
- **`AgentFeatureProposer`** — wraps a `FireflyAgent` (Pydantic AI) to propose features as structured
  output; `StaticFeatureProposer` for deterministic / LLM-free pipelines.
- **`CostBenefitGate`** — accepts a GenAI contribution only if it beats the current score by `min_gain`.
- Opt-in via `genai.enabled` + `@enable_genai_ds_stack`-style auto-configuration.

Gate: ruff clean, pyright 0 errors, 76 tests passing (useful interaction feature accepted with measured
lift; useless/unsafe proposals rejected; agentic integration verified via pydantic-ai `TestModel`).

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
