<p align="center">
  <img src="img/banner.svg" alt="Firefly DataScience" width="100%">
</p>

# Firefly DataScience Documentation

**AutoML that fuses GenAI with classical ML & Deep Learning — hexagonal, secure-by-default, native to the Firefly Framework.**

> New here? Jump to the **[Tutorial](tutorial.md)** for a guided, runnable walkthrough, or
> **[Configuring the LLM](llm-configuration.md)** to wire up GenAI.

`fireflyframework-datascience` is a state-of-the-art Python metaframework for AutoML. It pairs **GenAI**
— built on [`fireflyframework-agentic`](https://github.com/fireflyframework/fireflyframework-agentic),
which wraps [Pydantic AI](https://ai.pydantic.dev/) — with **traditional ML and Deep Learning**, so any
team can apply data science to any project quickly, with production governance, hexagonal
swappability, and security by default.

The reproducible pattern: **the LLM proposes; a deterministic classical engine decides.** GenAI
proposes code, features, pipelines and seeds; a classical engine trains, scores and selects; and every
GenAI step is gated behind a measured improvement over a seeded classical baseline. GenAI is a
governed, measurably-gated accelerator over a battle-tested classical core — never a black box.

<p align="center">
  <img src="img/architecture.svg" alt="Firefly DataScience architecture" width="100%">
</p>

## The 7 pillars

1. **Classical-first AutoML.** A deterministic engine trains, scores and selects models across
   scikit-learn, XGBoost, LightGBM, CatBoost, AutoGluon and TabPFN — reproducible from a seed.
2. **GenAI as a gated accelerator.** The LLM proposes features and pipelines; nothing ships unless it
   beats the seeded classical baseline (`genai.cost_benefit_gate` is on by default).
3. **The agentic ML-engineering loop.** Propose → train → score → select, driven by the agentic
   runtime, with measured improvement at every step.
4. **Deep Learning, swappable.** PyTorch Lightning and HuggingFace sit behind the same ports as the
   classical adapters — tabular, text, vision, timeseries and multimodal.
5. **Hexagonal & swappable.** Every ML/MLOps library (MLflow, Feast, BentoML, …) is a swappable
   adapter behind a `Protocol` port; the core stays library-agnostic.
6. **Secure by default.** LLM-generated code runs in a sandbox (`monty` by default) with timeouts and
   approval gates; GenAI is **off** until you enable it.
7. **Firefly-native.** Auto-configuration, dependency injection, a startup banner + wiring summary,
   CalVer, and the same CI gates as the rest of the Firefly Framework.

## Install

```bash
uv add fireflyframework-datascience                    # core
uv add 'fireflyframework-datascience[automl-stack]'    # + classical AutoML + tracking
```

## End-to-end example

Booting an application returns a started `ApplicationContext`: the loaded config plus the wired DI
container.

```python
from fireflyframework_datascience import (
    FireflyDataScienceApplication,
    FireflyDataScienceConfig,
    Modality,
    TaskType,
)

# Boot: load config -> banner -> wire container -> wiring summary -> ready context
app = FireflyDataScienceApplication.run()

print(app.config.default_ml_framework)   # "sklearn"
print(app.bean_count)                    # number of wired beans
print(app.applied_auto_configurations)   # discovered auto-configurations

# Core domain types stay importable with zero ML extras installed
task = TaskType.BINARY
assert task.is_classification()
assert Modality.TABULAR in Modality
```

Configuration is a `pydantic-settings` model. Values resolve (highest precedence first) from
constructor kwargs → `FIREFLY_DATASCIENCE_*` env vars → `.env` → profile YAML overlays →
`firefly-datascience.yaml` → field defaults. GenAI is classical-first and **off by default**:

```python
config = FireflyDataScienceConfig(
    app_name="lumen-credit-risk",
    default_ml_framework="lightgbm",
    profiles=["prod"],
)
config.genai.enabled = True              # opt in to the GenAI accelerator
config.genai.cost_benefit_gate = True    # require a measured win over baseline
config.execution.sandbox = "docker"      # sandbox LLM-generated code

app = FireflyDataScienceApplication.run(config=config)
```

You can also boot from a config directory and active profiles directly:

```python
app = FireflyDataScienceApplication.run(config_dir="./config", profiles=["prod"])
```

## CLI

```bash
firefly-ds doctor       # check your environment & installed adapters
firefly-ds introspect   # boot the app and show discovered auto-configurations
```

## Documentation

| Page | What it covers |
| --- | --- |
| [Architecture](architecture.md) | Hexagonal ports/adapters, the DI container, auto-configuration. |
| [Quickstart](quickstart.md) | Install, boot an `ApplicationContext`, run your first AutoML job. |
| [Configuration](configuration.md) | `FireflyDataScienceConfig`, profiles, env vars, YAML overlays. |
| [Datasets](datasets.md) | Dataset backends (pandas, …) and `Modality`. |
| [AutoML](automl.md) | The classical-first engine: train, score, select. |
| [GenAI features](genai-features.md) | The gated GenAI accelerator and the cost-benefit gate. |
| [Agentic loop](agentic-loop.md) | Propose → train → score → select on the agentic runtime. |
| [Deep Learning](deep-learning.md) | PyTorch Lightning & HuggingFace behind the ports. |
| [Serving](serving.md) | Model registry, feature store, and BentoML serving. |
| [Security](security.md) | Sandboxed code execution, approval gates, secure defaults. |
| [Benchmarks](benchmarks.md) | Reproducible measurement of GenAI vs. classical baselines. |
| [Use case: Lumen](use-case-lumen.md) | End-to-end lending vertical worked example. |

## See also

- [Architecture](architecture.md)
- [Quickstart](quickstart.md)
- [Configuration](configuration.md)
- [AutoML](automl.md)
- [GenAI features](genai-features.md)
