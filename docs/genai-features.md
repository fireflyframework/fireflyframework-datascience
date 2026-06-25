# GenAI Feature Engineering

**The LLM proposes feature code; classical cross-validation decides what survives.**

Firefly DataScience treats generative AI as a *proposer*, never a judge. A language model
suggests pandas snippets that add new columns; a deterministic engine measures the
cross-validation lift of each one, and a `CostBenefitGate` keeps a feature only if it
beats the current baseline by a measurable margin. The LLM never touches the score — it
just generates candidates, and the data does the rest.

![GenAI proposes, classical CV decides](img/genai-classical-fusion.svg)

## The loop

`GenAIFeatureEngineer` runs **propose → execute → measure → gate**:

1. **Propose** — a `FeatureProposer` returns a list of `FeatureProposal`s (name, code, rationale).
2. **Execute** — `FeatureCodeExecutor` statically vets and safely runs each snippet against a copy of the frame.
3. **Measure** — a classical estimator scores the candidate frame via cross-validation.
4. **Gate** — `CostBenefitGate` accepts the feature only if the score improves by at least `min_gain`.

Everything is injectable, so the loop runs fully offline with a stub proposer — no LLM required for tests.

## Quick start (no LLM)

Use `StaticFeatureProposer` to drive the loop with a fixed, known set of features.

```python
from fireflyframework_datascience.features import FeatureProposal, StaticFeatureProposer
from fireflyframework_datascience.features.genai import GenAIFeatureEngineer

proposer = StaticFeatureProposer([
    FeatureProposal(
        name="income_per_dependent",
        code="df['income_per_dependent'] = df['income'] / (df['dependents'] + 1)",
        rationale="Normalises income by household size.",
    ),
    FeatureProposal(
        name="utilization",
        code="df['utilization'] = df['balance'] / (df['credit_limit'] + 1)",
        rationale="Classic credit-risk ratio.",
    ),
])

engineer = GenAIFeatureEngineer(proposer, cv=5, max_features=5)
result = engineer.engineer(dataset)

print(result.summary())
# GenAI feature engineering: 1 accepted, 1 rejected; roc_auc 0.8123 -> 0.8310 (lift +0.0187)
```

## Reading the result

`engineer()` returns an `EngineeringResult` with the engineered dataset plus a full audit trail.

```python
result.dataset        # Dataset with accepted features merged in (dataset.with_features(...))
result.baseline_score # CV score before any GenAI feature
result.final_score    # CV score after accepted features
result.lift           # final_score - baseline_score
result.metric         # e.g. "roc_auc"

for acc in result.accepted:   # AcceptedFeature
    print(acc.proposal.name, acc.score, acc.gain)

for rej in result.rejected:   # RejectedFeature
    print(rej.proposal.name, rej.reason, rej.score)
```

A proposal is rejected when its code is unsafe, fails at runtime, adds no new numeric
column, or produces **no measured lift** over the current best score.

## The gate is the governance primitive

`CostBenefitGate` is what keeps GenAI honest. It compares the candidate score against the
current best and only accepts a strict improvement beyond `min_gain`.

```python
from fireflyframework_datascience.features import CostBenefitGate
from fireflyframework_datascience.features.genai import GenAIFeatureEngineer

# Require at least +0.005 of lift before a feature is worth its complexity.
gate = CostBenefitGate(min_gain=0.005)
engineer = GenAIFeatureEngineer(proposer, gate=gate)
```

`gate.accepts(current_score, candidate_score)` returns `True` only when
`candidate_score - current_score > min_gain`. With the default `min_gain=0.0`, any strict
improvement is kept; raise it to demand features that earn their keep.

## Proposing with an LLM agent

`AgentFeatureProposer` wraps a `FireflyAgent` from `fireflyframework-agentic`. The agent is
built lazily on first use, so no LLM client is created at startup. It sends the schema, a
few sample rows, and the task to the model, then maps the structured output to
`FeatureProposal`s.

```python
from fireflyframework_datascience.features.genai import (
    AgentFeatureProposer,
    GenAIFeatureEngineer,
)

# Pass a model string (default "openai:gpt-4o") or your own pre-built FireflyAgent.
proposer = AgentFeatureProposer(model="openai:gpt-4o", sample_rows=5)

engineer = GenAIFeatureEngineer(proposer, cv=5, max_features=8)
result = engineer.engineer(dataset)
print(result.summary())
```

The agent is instructed to return short pandas snippets that add **exactly one new numeric
column** to a DataFrame named `df`, using only `df`, `pd`, and `np` — no imports, no I/O.

For tests, inject a pre-built agent (or a fake) instead of a model:

```python
proposer = AgentFeatureProposer(agent=my_fake_agent)
```

## Secure execution

LLM-generated code is an attack surface, so `FeatureCodeExecutor` applies defence in depth
before anything runs. It reuses the static safety analysis from
`fireflyframework_agentic.execution` (denying imports, dunder access, and dangerous
builtins like `eval`/`exec`/`open`), then executes the vetted snippet in a restricted
namespace exposing only `df` (a copy of the frame), `pd`, and `np`, with a minimal
`__builtins__` allowlist.

```python
from fireflyframework_datascience.features.executor import (
    FeatureCodeExecutor,
    FeatureExecutionError,
)

executor = FeatureCodeExecutor()
try:
    frame = executor.execute("df['ratio'] = df['a'] / (df['b'] + 1)", X)
except FeatureExecutionError as exc:
    print("rejected:", exc)
```

`execute(code, X)` raises `FeatureExecutionError` if the code is unsafe, errors at runtime,
leaves something other than a DataFrame in `df`, adds no new column, or adds a non-numeric
column. Newly added columns also have `±inf` replaced with `NaN` so downstream estimators
do not break. You can pass a custom executor into the engineer:

```python
GenAIFeatureEngineer(proposer, executor=FeatureCodeExecutor())
```

## Customising the measurement

By default the engineer measures lift with a `HistGradientBoosting*` estimator (classifier
or regressor, chosen by task) wrapped in an imputation/encoding pipeline, scored with the
evaluator's default metric for the task. Override the scoring estimator or evaluator:

```python
from sklearn.ensemble import RandomForestClassifier

engineer = GenAIFeatureEngineer(
    proposer,
    scorer_estimator=lambda task: RandomForestClassifier(n_estimators=200),
    cv=10,
    random_state=7,
)
```

## See also

- [Datasets](datasets.md)
- [Evaluation & Metrics](automl.md)
- [Core Types](configuration.md)
- [AutoML Pipeline](automl.md)
