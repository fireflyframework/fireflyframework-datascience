# Configuring the LLM

**Point Firefly DataScience at a real LLM for GenAI feature engineering and the agentic loop — in three settings, with no key required to boot.**

Firefly DataScience is classical-first. GenAI is an *optional accelerator*: everything except the
explicit GenAI steps runs with no LLM at all. When you do switch it on, the agent is powered by
[`fireflyframework-agentic`](https://github.com/fireflyframework/fireflyframework-agentic), which wraps
[Pydantic AI](https://ai.pydantic.dev/) — so any provider Pydantic AI supports works here, selected by a
single model string.

!!! warning "GenAI is off by default"
    `GenAIConfig.enabled` defaults to `False` (see `core/config.py`). Nothing reaches an LLM until you
    set `genai.enabled=true` **and** install the `genai` extra. A fresh install is fully deterministic.

This page covers the four things you need: enable GenAI, choose a provider/model, supply keys, and
govern cost. The same propose → execute → measure → gate loop runs whether the proposer is a real LLM or
a deterministic stand-in, so you can develop and test offline.

## 1. Enable GenAI

Two switches turn it on: the `genai.enabled` flag, and the installed `genai` extra.

```bash
uv add 'fireflyframework-datascience[genai]'        # installs the agentic GenAI accelerators
export FIREFLY_DATASCIENCE_GENAI__ENABLED=true       # turn the GenAI auto-configurations on
```

Or in `firefly-datascience.yaml`:

```yaml
genai:
  enabled: true
  default_model: openai:gpt-4o    # (1)!
  cost_benefit_gate: true         # (2)!
```

1. The Pydantic AI model string. Field default is `openai:gpt-4o` (`GenAIConfig.default_model`).
2. Keep proposals only if they measurably beat the baseline. Default is `true`.

With `genai.enabled=true`, `FeaturesAutoConfiguration` (GenAI feature engineering) and
`EngineeringAutoConfiguration` (the agentic loop) register their agent-backed beans — each wiring an
`AgentFeatureProposer`/`AgentSolutionProposer` built from `config.genai.default_model`. The agent, and
its API client, is built **lazily on first use**, so the application still boots without a key.

!!! note "Environment-variable mapping"
    Config keys map to env vars with the prefix `FIREFLY_DATASCIENCE_` and the nested delimiter `__`
    (double underscore). So `genai.enabled` becomes `FIREFLY_DATASCIENCE_GENAI__ENABLED`. Precedence is
    constructor kwargs → env vars → `.env` → profile YAML → base YAML → field defaults. See
    [Configuration](configuration.md).

## 2. Choose a provider and model

Set `genai.default_model` to a Pydantic AI model string, `"<provider>:<model>"`. Use a content tab for
the two first-class providers; the table below lists the rest.

=== "OpenAI"

    ```bash
    export FIREFLY_DATASCIENCE_GENAI__DEFAULT_MODEL=openai:gpt-4o   # or openai:gpt-4o-mini
    export OPENAI_API_KEY=sk-...
    ```

    `openai:gpt-4o` is the framework default, so OpenAI works with only a key set.

=== "Anthropic"

    ```bash
    export FIREFLY_DATASCIENCE_GENAI__DEFAULT_MODEL=anthropic:claude-sonnet-4-5
    export ANTHROPIC_API_KEY=sk-ant-...
    ```

    Other Claude strings work the same way, e.g. `anthropic:claude-opus-4-1`.

| Provider | Model string (example) | API key env var |
|---|---|---|
| OpenAI | `openai:gpt-4o`, `openai:gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic:claude-sonnet-4-5`, `anthropic:claude-opus-4-1` | `ANTHROPIC_API_KEY` |
| Google | `google-gla:gemini-2.0-flash` | `GEMINI_API_KEY` |
| Groq | `groq:llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| Mistral | `mistral:mistral-large-latest` | `MISTRAL_API_KEY` |
| Ollama (local) | `openai:llama3.2` via a local base URL | — (runs locally) |

## 3. Where to put API keys

Keys are read from the environment (Pydantic AI's convention). In order of convenience:

```bash
# 1. Shell environment
export OPENAI_API_KEY=sk-...

# 2. A local .env file (loaded automatically; real env vars always win)
echo 'OPENAI_API_KEY=sk-...' >> .env
```

!!! warning "Never commit API keys"
    Keep keys in `.env` (git-ignored) or a secrets manager. The framework never logs keys, and
    agentic's `OutputGuard` redacts secrets from model output. The agent is built lazily, so a missing
    key surfaces only on the *first* GenAI call — not at startup.

## 4. Use it

Once enabled, swap the deterministic stand-in proposers (used in tests and the tutorial) for the
agent-backed ones — both take a keyword-only `model` and pick up `genai.default_model` automatically
when wired from config.

```python
from fireflyframework_datascience.features.genai import AgentFeatureProposer, GenAIFeatureEngineer
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

train, _ = SklearnDatasetLoader().load("breast_cancer").train_test_split()

# The LLM proposes feature code; classical CV measures the lift; the gate decides.
engineer = GenAIFeatureEngineer(AgentFeatureProposer(model="openai:gpt-4o"))
result = engineer.engineer(train)
print(result.summary())
```

!!! success "Expected"
    ```text
    GenAI feature engineering: 3 accepted, 5 rejected; roc_auc 0.9700 -> 0.9800 (lift +0.0100)
    ```
    Counts and scores vary by dataset and model; the shape is fixed by `EngineeringResult.summary()`.

The agentic loop works the same way — the LLM reflects on the attempt history while the engine trains
and verifies each candidate:

```python
from fireflyframework_datascience.engineering.loop import AgenticAutoML, AgentSolutionProposer

run = AgenticAutoML(AgentSolutionProposer(model="anthropic:claude-sonnet-4-5")).solve(train)
print(run.summary())
```

!!! success "Expected"
    ```text
    Agentic AutoML: 6 attempts (4 verified); best=HistGradientBoostingClassifier roc_auc=0.9850 (baseline 0.9600)
    ```
    From `EngineeringRun.summary()`. The trainer, counts, and scores depend on the registry and run.

Or wire everything from the application context, where the model comes from config:

```python
from fireflyframework_datascience import FireflyDataScienceApplication

app = FireflyDataScienceApplication.run()   # genai.enabled -> agent beans registered
engineer = app.get(...)                      # resolve the FeatureEngineerPort bean, wired with your model
```

## 5. Cost, budget, and governance

GenAI here is a **measurably-gated accelerator**, never a black box.

!!! firefly "The LLM proposes; the classical engine decides"
    A proposal is accepted only when a deterministic cross-validation score beats the seeded baseline by
    at least `CostBenefitGate.min_gain` (default `0.0`). The model's confidence is irrelevant — the
    measured score is the sole arbiter, and a step that produces no lift can be disabled outright.

```yaml
genai:
  enabled: true
  cost_benefit_gate: true     # auto-disable a GenAI step that does not beat the seeded baseline
  budget_usd: 5.0             # optional hard spend ceiling for a run (default: null = no ceiling)
```

- The **`CostBenefitGate`** accepts a proposed feature/candidate only when it improves the
  cross-validation score: `accepts(current, candidate)` returns `True` iff
  `candidate - current > min_gain`. The LLM never decides; the measured score does.
- `budget_usd` defaults to `None` (no ceiling). When set, agentic's usage tracking enforces it as a hard
  spend cap for the run.

## 6. Secure code execution

LLM-proposed feature code never runs raw. The `FeatureCodeExecutor` first reuses agentic's static safety
analysis (deny imports / dunder access / dangerous builtins) and rejects anything unsafe with a
`FeatureExecutionError`; vetted snippets then run in a restricted namespace exposing only `df`, `pd`, and
`np`. Choose the sandbox tier under `execution`:

```yaml
execution:
  sandbox: monty          # monty (default, deny-by-default) | docker | e2b | local
  require_approval: true  # human-in-the-loop before non-sandboxed execution (default: true)
  timeout_seconds: 60     # per-execution wall clock (default: 60)
```

See the [Security Model](security.md) for the full trust model.

## 7. Offline and testing (no key required)

For development, tests, and the [tutorial](tutorial.md), use the deterministic stand-ins. They exercise
the exact same propose → execute → measure → gate loop without any LLM:

```python
from fireflyframework_datascience.features import StaticFeatureProposer, FeatureProposal
from fireflyframework_datascience.engineering import SequenceProposer, SolutionCandidate
```

To unit-test the *real* agent integration without a network, pass Pydantic AI's `TestModel`:

```python
from pydantic_ai.models.test import TestModel
from fireflyframework_datascience.features.genai import AgentFeatureProposer

proposer = AgentFeatureProposer(model=TestModel(custom_output_args={"features": [...]}))
```

!!! tip "Same loop, different proposer"
    `AgentFeatureProposer` only implements the `FeatureProposer` protocol — `GenAIFeatureEngineer` calls
    `propose(...)` and gates the results identically regardless of whether the proposer is a live agent,
    a `StaticFeatureProposer`, or a `TestModel`-backed agent.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OpenAIError: Missing credentials` | Set `OPENAI_API_KEY` (or the provider's key). The agent builds lazily, so this only fires on the first GenAI call. |
| GenAI steps don't run | Confirm `genai.enabled=true` **and** the `genai` extra is installed (`firefly-ds doctor`). |
| Every proposed feature is rejected | Working as designed — the gate found no measurable lift. Lower `min_gain` or try a stronger model. |

## See also

- [GenAI Feature Engineering](genai-features.md)
- [Agentic Loop](agentic-loop.md)
- [Configuration](configuration.md)
- [Security Model](security.md)
- [Tutorial](tutorial.md)
