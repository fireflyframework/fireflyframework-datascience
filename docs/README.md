# Firefly DataScience — Documentation

**The complete documentation set.** Browse it as a rendered site at
**<https://fireflyframework.github.io/fireflyframework-datascience/>**, or read the Markdown here.

<p align="center">
  <img src="img/banner.svg" alt="Firefly DataScience" width="100%">
</p>

## Table of contents

### Getting started
| Page | What it covers |
|---|---|
| [Home / Overview](index.md) | what the framework is, the 7 pillars, the architecture at a glance |
| [Quick Start](quickstart.md) | install, boot, your first AutoML run, the `firefly-ds` CLI |
| [Tutorial](tutorial.md) | the guided, runnable end-to-end walkthrough (offline, tested) |
| [Configuration](configuration.md) | env vars, `.env`, YAML, and profile precedence |
| [Configuring the LLM](llm-configuration.md) | providers, API keys, model selection, cost & budget gating |

### Concepts
| Page | What it covers |
|---|---|
| [Architecture](architecture.md) | the five layers, hexagonal ports/adapters, the DI container, auto-configuration |
| [Datasets](datasets.md) | the `Dataset` container, loaders, `train_test_split`, task inference |
| [Classical AutoML](automl.md) | the `AutoML` facade, trainers, search policies, metrics, the leaderboard |
| [GenAI Feature Engineering](genai-features.md) | propose → execute → measure → gate; the `CostBenefitGate` |
| [Agentic ML-Engineering Loop](agentic-loop.md) | propose → train → verify → reflect → select |
| [Deep Learning & TabFM](deep-learning.md) | sklearn-MLP, PyTorch Lightning, HuggingFace, TabPFN |
| [Serving & Lineage](serving.md) | the in-process server, gated backends, lineage |
| [Security Model](security.md) | secure code execution, sandbox tiers, prompt-injection defense |
| [Benchmarks](benchmarks.md) | the three-tier AMLB/OpenML-anchored evaluation strategy |

### Use case
| Page | What it covers |
|---|---|
| [Lumen Lending — Credit Risk](use-case-lumen.md) | a full, realistic walkthrough end to end |

## Diagrams

All diagrams are generated (WeasyPrint-safe SVG, teal palette) by
[`assets/tools/gen_diagrams.py`](../assets/tools/gen_diagrams.py) into [`img/`](img):

| Diagram | |
|---|---|
| [Architecture](img/architecture.svg) | the five-layer design |
| [Hexagonal ports](img/hexagonal.svg) | ports & adapters around a library-agnostic core |
| [Auto-configuration](img/auto-configuration.svg) | entry-point discovery → conditions → beans |
| [AutoML pipeline](img/automl-loop.svg) | the classical AutoML flow |
| [GenAI × classical fusion](img/genai-classical-fusion.svg) | the governed fusion |
| [Agentic loop](img/agentic-loop.svg) | propose → verify → reflect → select |
| [Security tiers](img/security.svg) | the secure-by-default execution model |
| [Ecosystem](img/ecosystem.svg) | how this sits beside Agentic and PyFly |

---

<sub>Copyright 2026 Firefly Software Foundation · Licensed under the Apache License 2.0</sub>
