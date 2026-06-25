<p align="center">
  <img src="assets/banner.svg" alt="Firefly DataScience — AutoML that fuses GenAI with classical ML and Deep Learning, built on Firefly Agentic" width="100%">
</p>

<h1 align="center">Firefly DataScience</h1>

<p align="center">
  <strong>AutoML that fuses GenAI with classical ML &amp; Deep Learning — hexagonal, secure-by-default, native to the Firefly Framework.</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.13%2B-blue.svg" alt="Python 3.13+"></a> &nbsp;·&nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a> &nbsp;·&nbsp;
  <a href="https://github.com/fireflyframework/fireflyframework-agentic"><img src="https://img.shields.io/badge/built%20on-Firefly%20Agentic-22d3ee.svg" alt="Built on Firefly Agentic"></a> &nbsp;·&nbsp;
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/lint-ruff-261230.svg" alt="ruff"></a> &nbsp;·&nbsp;
  <a href="https://microsoft.github.io/pyright/"><img src="https://img.shields.io/badge/types-pyright-blue.svg" alt="pyright"></a>
</p>

<p align="center">
  <em>The LLM proposes; a deterministic classical engine decides. GenAI is a governed, measurably-gated
  accelerator over a battle-tested classical core — never a black box.</em>
</p>

<p align="center">
  <sub>Copyright 2026 Firefly Software Foundation · Licensed under the Apache License 2.0</sub>
</p>

---

> **Status:** all sub-projects delivered and green (ruff · pyright · 90+ tests). Classical tabular
> AutoML · GenAI feature engineering · the agentic ML-engineering loop · deep learning (PyTorch
> Lightning) + NLP (HuggingFace) + vision · TabFM · serving · the OpenML-AMLB benchmark harness.
> **New here? Start with the [Tutorial](docs/tutorial.md)** or browse the
> **[documentation site](https://fireflyframework.github.io/fireflyframework-datascience/)**.

## What is this?

`fireflyframework-datascience` is a state-of-the-art Python **metaframework for AutoML**. It combines
**GenAI** (built on [`fireflyframework-agentic`](https://github.com/fireflyframework/fireflyframework-agentic),
which wraps [Pydantic AI](https://ai.pydantic.dev/)) with **traditional ML and Deep Learning**, so any
team can apply data science to any project quickly — with production governance, hexagonal
swappability, and security by default.

- **One reproducible pattern.** The LLM proposes code/features/pipelines/seeds; a deterministic
  classical engine trains, scores, and selects; every GenAI step is gated behind a measured
  improvement over a seeded classical baseline.
- **Hexagonal & swappable.** Every ML/MLOps library (scikit-learn, XGBoost, LightGBM, CatBoost,
  AutoGluon, TabPFN, PyTorch Lightning, HuggingFace, MLflow, Feast, BentoML, …) is a swappable adapter
  behind a `Protocol` port. The core stays library-agnostic.
- **Firefly-native.** Auto-configuration, dependency injection, a startup banner + wiring summary,
  CalVer, and the same CI gates as the rest of the Firefly Framework.

## Quick start

```bash
uv add 'fireflyframework-datascience[tabular]'        # classical AutoML
# or:  uv add 'fireflyframework-datascience[automl-stack]'   # + TabPFN, MLflow, OpenML
```

Train, rank, and evaluate models in five lines:

```python
from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

train, test = SklearnDatasetLoader().load("breast_cancer").train_test_split()
result = AutoML().fit(train)               # cross-validates candidates, picks the winner
print(result.leaderboard_table())          # random_forest / linear / hist_gradient_boosting …
print(result.evaluate(test))               # holdout roc_auc ≈ 0.98
```

Boot it as a Firefly application (auto-configuration + dependency injection), or use the CLI:

```bash
firefly-ds doctor       # check your environment & installed adapters
firefly-ds introspect   # boot the app and show discovered auto-configurations
```

Add a real LLM for GenAI feature engineering and the agentic loop — see
[Configuring the LLM](docs/llm-configuration.md). The full guided walkthrough is the
[Tutorial](docs/tutorial.md).

## Architecture

Five acyclic layers, mirroring `fireflyframework-agentic` with a **DataScience** layer inserted. Every
ML/MLOps library is a swappable adapter behind a `Protocol` port, registered by **entry-point
auto-configuration** and resolved through a type-hint **dependency-injection container**.

<p align="center">
  <img src="docs/img/architecture.svg" alt="Firefly DataScience layered architecture" width="70%">
</p>

```
Core → Agent (reused: agentic) → DataScience → Intelligence → Orchestration
```

The GenAI ↔ classical fusion is governed: the LLM proposes code; the classical engine measures; a
cost/benefit gate keeps only what beats the baseline.

<p align="center">
  <img src="docs/img/genai-classical-fusion.svg" alt="Governed GenAI and classical fusion" width="70%">
</p>

## Documentation

📖 **Full docs site:** <https://fireflyframework.github.io/fireflyframework-datascience/>

| Guide | |
|---|---|
| [Tutorial](docs/tutorial.md) | the guided end-to-end walkthrough (runs offline; tested) |
| [Quick Start](docs/quickstart.md) | install, boot, first AutoML run, the `firefly-ds` CLI |
| [Configuring the LLM](docs/llm-configuration.md) | providers, API keys, model selection, cost gating |
| [Architecture](docs/architecture.md) | layers, hexagonal ports, auto-configuration, the DI container |
| [Configuration](docs/configuration.md) | env / `.env` / YAML / profiles precedence |
| [Datasets](docs/datasets.md) | the `Dataset` container and loaders |
| [Classical AutoML](docs/automl.md) | the `AutoML` facade, trainers, search, metrics |
| [GenAI Feature Engineering](docs/genai-features.md) | propose → execute → measure → gate |
| [Agentic ML-Engineering Loop](docs/agentic-loop.md) | propose → verify → reflect → select |
| [Deep Learning & TabFM](docs/deep-learning.md) | MLP, TabPFN, the PyTorch integration point |
| [Serving & Lineage](docs/serving.md) | in-process and gated servers, lineage |
| [Security Model](docs/security.md) | secure code execution, sandbox tiers, prompt-injection defense |
| [Benchmarks](docs/benchmarks.md) | the three-tier AMLB-anchored evaluation strategy |
| [Use Case: Lumen Lending](docs/use-case-lumen.md) | the end-to-end credit-risk walkthrough |

## License

Apache-2.0. Copyright 2026 Firefly Software Foundation.
