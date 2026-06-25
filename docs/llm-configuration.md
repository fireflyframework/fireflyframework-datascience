# Configuring the LLM

**How to point Firefly DataScience at a real LLM for GenAI feature engineering and the agentic loop.**

GenAI is **off by default** — the framework is classical-first, and everything except the GenAI steps
runs with no LLM at all. When you do enable GenAI, it is powered by
[`fireflyframework-agentic`](https://github.com/fireflyframework/fireflyframework-agentic), which wraps
[Pydantic AI](https://ai.pydantic.dev/) — so any provider Pydantic AI supports works here.

## 1. Enable GenAI

Two things switch it on: the `genai.enabled` flag, and an installed `genai` extra.

```bash
uv add 'fireflyframework-datascience[genai]'        # installs the agentic GenAI accelerators
export FIREFLY_DATASCIENCE_GENAI__ENABLED=true       # turn the GenAI auto-configurations on
```

Or in `firefly-datascience.yaml`:

```yaml
genai:
  enabled: true
  default_model: openai:gpt-4o
  cost_benefit_gate: true
```

With `genai.enabled=true`, the `FeaturesAutoConfiguration` (GenAI feature engineering) and
`EngineeringAutoConfiguration` (the agentic loop) register their agent-backed beans. The agent — and its
API client — is built **lazily on first use**, so the application still boots without a key.

## 2. Choose a provider and model

Set `genai.default_model` to a Pydantic AI model string, `"<provider>:<model>"`:

| Provider | Model string (example) | API key env var |
|---|---|---|
| OpenAI | `openai:gpt-4o`, `openai:gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic:claude-sonnet-4-5`, `anthropic:claude-opus-4-1` | `ANTHROPIC_API_KEY` |
| Google | `google-gla:gemini-2.0-flash` | `GEMINI_API_KEY` |
| Groq | `groq:llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| Mistral | `mistral:mistral-large-latest` | `MISTRAL_API_KEY` |
| Ollama (local) | `openai:llama3.2` via a local base URL | — (runs locally) |

```bash
export FIREFLY_DATASCIENCE_GENAI__DEFAULT_MODEL=anthropic:claude-sonnet-4-5
export ANTHROPIC_API_KEY=sk-ant-...
```

## 3. Where to put API keys

Keys are read from the environment (Pydantic AI's convention). Options, in order of convenience:

```bash
# 1. Shell environment
export OPENAI_API_KEY=sk-...

# 2. A local .env file (loaded automatically; real env vars always win)
echo 'OPENAI_API_KEY=sk-...' >> .env
```

> **Security.** Never commit API keys. Keep them in `.env` (git-ignored) or a secrets manager. The
> framework never logs keys, and `OutputGuard` (from agentic) redacts secrets from model output.

## 4. Use it

Once enabled, swap the deterministic stand-in proposers (used in tests/tutorials) for the agent-backed
ones — they pick up `genai.default_model` automatically:

```python
from fireflyframework_datascience.features.genai import AgentFeatureProposer, GenAIFeatureEngineer
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

train, _ = SklearnDatasetLoader().load("breast_cancer").train_test_split()

# The LLM proposes feature code; classical CV measures the lift; the gate decides.
engineer = GenAIFeatureEngineer(AgentFeatureProposer(model="openai:gpt-4o"))
result = engineer.engineer(train)
print(result.summary())            # e.g. "3 accepted, 5 rejected; roc_auc 0.97 -> 0.98 (+0.01)"
```

```python
from fireflyframework_datascience.engineering.loop import AgenticAutoML, AgentSolutionProposer

run = AgenticAutoML(AgentSolutionProposer(model="anthropic:claude-sonnet-4-5")).solve(train)
print(run.summary())               # the LLM reflects on history; the engine trains/verifies each candidate
```

Or wire everything from the application context (the model comes from config):

```python
from fireflyframework_datascience import FireflyDataScienceApplication

app = FireflyDataScienceApplication.run()              # genai.enabled -> agent beans registered
engineer = app.get(...)            # resolve the FeatureEngineerPort bean, already wired with your model
```

## 5. Cost, budget, and governance

GenAI is a **measurably-gated accelerator**, never a black box:

```yaml
genai:
  enabled: true
  cost_benefit_gate: true     # auto-disable a GenAI step that does not beat the seeded baseline
  budget_usd: 5.0             # optional hard spend ceiling for a run
```

- The **`CostBenefitGate`** accepts a proposed feature/candidate only if it improves the
  cross-validation score — the LLM never decides, the measured score does.
- Token usage and cost are tracked by agentic's `UsageTracker`; a `BudgetGate` enforces `budget_usd`.

## 6. Secure code execution

LLM-proposed feature code runs through static safety analysis and a restricted namespace. Choose the
sandbox tier under `execution`:

```yaml
execution:
  sandbox: monty          # monty (default, deny-by-default) | docker | e2b | local
  require_approval: true  # human-in-the-loop before any non-sandboxed execution
  timeout_seconds: 60
```

See [Security Model](security.md) for the full trust model.

## 7. Offline & testing (no key required)

For development, tests, and the [tutorial](tutorial.md), use the deterministic stand-ins — they exercise
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

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OpenAIError: Missing credentials` | Set `OPENAI_API_KEY` (or the provider's key). The agent builds lazily, so this only fires on first GenAI call. |
| GenAI steps don't run | Confirm `genai.enabled=true` **and** the `genai` extra is installed (`firefly-ds doctor`). |
| Every proposed feature is rejected | Working as designed — the gate found no measurable lift. Lower `min_gain` or try a stronger model. |

## See also

- [GenAI Feature Engineering](genai-features.md) · [Agentic Loop](agentic-loop.md) ·
  [Configuration](configuration.md) · [Security Model](security.md) · [Tutorial](tutorial.md)
