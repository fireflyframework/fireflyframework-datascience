# GenAI Feature Engineering

**The LLM proposes feature code; classical cross-validation decides what survives.**

Firefly DataScience treats generative AI as a *proposer*, never a judge. A language model
suggests pandas snippets that add new columns; a deterministic engine measures the
cross-validation lift of each one, and a `CostBenefitGate` keeps a feature only if it
beats the current baseline by a measurable margin. The LLM never touches the score — it
just generates candidates, and the data does the rest.

<p align="center"><img src="img/genai-classical-fusion.svg" alt="GenAI proposes feature code; classical cross-validation measures the lift and the gate decides what survives" width="100%"></p>

!!! firefly "Wired into AutoML"

    When a `FeatureEngineerPort` is present in the container (i.e. GenAI is enabled),
    [`AutoML.from_context(app)`](automl.md) runs this propose → execute → measure → gate loop
    **before** model selection and trains on the engineered features — the gate's accepted/rejected
    audit is threaded into `result.extras["feature_engineering"]`. Classical-first stays the default:
    with GenAI off, `AutoML` is unchanged.

## The loop

`GenAIFeatureEngineer` runs **propose → execute → measure → gate**:

1. **Propose** — a `FeatureProposer` returns a list of `FeatureProposal`s (`name`, `code`, `rationale`).
2. **Execute** — `FeatureCodeExecutor` statically vets and safely runs each snippet against a copy of the frame.
3. **Measure** — a classical estimator scores the candidate frame via cross-validation.
4. **Gate** — `CostBenefitGate` accepts the feature only if the score improves by more than `min_gain`.

Each accepted feature is folded into the working frame, so the next proposal is measured
against the *improved* baseline — features must earn their keep on top of everything kept
so far. Everything is injectable, so the loop runs fully offline with a stub proposer — no
LLM required for tests.

!!! firefly "The cost-benefit gate — GenAI proposes, the measured score decides"

    `CostBenefitGate` is the governance primitive that keeps GenAI honest. It compares the
    candidate score against the current best and accepts only a strict improvement beyond
    `min_gain`:

    ```python
    gate.accepts(current_score, candidate_score)
    # True  ⇔  (candidate_score - current_score) > min_gain
    ```

    With the default `min_gain=0.0`, any strict improvement is kept; raise it to demand
    features that clear a meaningful bar before they earn their complexity. A proposal that
    does not measurably beat the seeded classical baseline is rejected — the LLM never
    overrides the data.

## Quick start

Pick a proposer: a deterministic `StaticFeatureProposer` for known features and LLM-free
runs, or an `AgentFeatureProposer` that asks a model for candidates.

=== "Static (no LLM)"

    `StaticFeatureProposer` drives the loop with a fixed, known set of features — ideal for
    tests, reproducible pipelines, and codifying domain knowledge.

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
    ```

    !!! success "Expected"

        ```text
        GenAI feature engineering: 1 accepted, 1 rejected; roc_auc 0.8123 -> 0.8310 (lift +0.0187)
        ```

=== "Agent (LLM)"

    `AgentFeatureProposer` wraps a `FireflyAgent` from `fireflyframework-agentic`. It sends the
    schema, a few sample rows, and the task to the model, then maps the structured output to
    `FeatureProposal`s. The agent is built lazily on first use, so no LLM client is created at
    startup.

    ```python
    from fireflyframework_datascience.features.genai import (
        AgentFeatureProposer,
        GenAIFeatureEngineer,
    )

    proposer = AgentFeatureProposer(model="openai:gpt-4o", sample_rows=5)  # (1)!

    engineer = GenAIFeatureEngineer(proposer, cv=5, max_features=8)
    result = engineer.engineer(dataset)
    print(result.summary())
    ```

    1. Pass a model string (defaults to `"openai:gpt-4o"`) or your own pre-built `FireflyAgent`
       via `agent=...`. `sample_rows` controls how many rows of the frame are sent to the model.

    The agent is instructed to return short pandas snippets that add **exactly one new numeric
    column** to a DataFrame named `df`, using only `df`, `pd`, and `np` — no imports, no I/O.
    See [LLM configuration](llm-configuration.md) for choosing and configuring the model.

## Proposers and the proposer port

Both proposers satisfy the `FeatureProposer` protocol — `propose(dataset, *, max_features=5)
-> list[FeatureProposal]` — so the engineer depends only on the port, never a concrete LLM.

| Proposer | Signature | LLM? | Use it for |
|---|---|---|---|
| `StaticFeatureProposer` | `StaticFeatureProposer(proposals: list[FeatureProposal])` | No | Tests, reproducible runs, domain-known features |
| `AgentFeatureProposer` | `AgentFeatureProposer(*, model=None, agent=None, sample_rows=5)` | Yes (lazy) | Discovering candidates from the schema |

For tests, inject a pre-built agent (or a fake) instead of a model — no network call, no
startup client:

```python
proposer = AgentFeatureProposer(agent=my_fake_agent)
```

The structured output the agent returns is a `FeatureList` of `Feature` objects
(`name`, `code`, `rationale`), defined in `features/_schema.py`; the proposer maps each one
into a `FeatureProposal` and truncates to `max_features`.

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

`AcceptedFeature` carries the `proposal`, its `score`, and the `gain` over the previous best;
`RejectedFeature` carries the `proposal`, a `reason`, and the candidate `score` (`NaN` when
the code never ran). A proposal is rejected when its code is unsafe, fails at runtime, adds
no new numeric column, or produces **no measured lift** over the current best score — in which
case the reason reads `no lift (<candidate> <= <current>)`.

## Secure execution

LLM-generated code is an attack surface, so `FeatureCodeExecutor` applies defence in depth
before anything runs. It reuses the static safety analysis from
`fireflyframework_agentic.execution` (`analyze_code` against a `SafetyPolicy`), then executes
the vetted snippet in a restricted namespace.

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

The defence is layered:

1. **Static analysis** rejects denied modules (`os`, `sys`, `subprocess`, `shutil`, `socket`,
   `pathlib`, `importlib`, `builtins`), dunder access, and dangerous builtins
   (`eval`, `exec`, `compile`, `open`, `__import__`, `input`, `globals`, `locals`, `vars`,
   `getattr`, `setattr`) before anything runs.
2. **Restricted execution** runs the snippet against a *copy* of the frame in a namespace that
   exposes only `df`, `pd`, and `np`, with a minimal `__builtins__` allowlist (arithmetic and
   aggregation helpers like `abs`, `min`, `max`, `sum`, `round`, `len`, `range` — and nothing
   that performs I/O).
3. **Output validation** rejects anything that is not a DataFrame, adds no new column, or adds
   a non-numeric column; surviving new columns have `±inf` replaced with `NaN` so downstream
   estimators do not break.

`execute(code, X)` raises `FeatureExecutionError` if any layer fails. This is the CAAFE
pattern: pandas/numpy transforms only, never arbitrary capability.

!!! warning "Untrusted data needs more than the in-process allowlist"

    The in-process sandbox blocks the obvious escapes, but for untrusted inputs you can still
    require human-in-the-loop approval and/or route execution to a container sandbox via
    `config.execution.sandbox`. See [Security](security.md).

You can pass a custom executor into the engineer:

```python
GenAIFeatureEngineer(proposer, executor=FeatureCodeExecutor())
```

## Customising the measurement

By default the engineer measures lift with a `HistGradientBoosting*` estimator — a
`HistGradientBoostingClassifier` for classification tasks, otherwise a
`HistGradientBoostingRegressor` — wrapped in an imputation/encoding pipeline (median impute
for numerics; most-frequent impute plus one-hot encoding for categoricals). It is scored with
the evaluator's default metric for the task. Override the scoring estimator, the evaluator, or
the CV folds:

```python
from sklearn.ensemble import RandomForestClassifier

engineer = GenAIFeatureEngineer(
    proposer,
    scorer_estimator=lambda task: RandomForestClassifier(n_estimators=200),  # (1)!
    cv=10,
    random_state=7,
)
```

1. `scorer_estimator` is a `Callable[[TaskType], estimator]`: it receives the task and returns
   the estimator used to measure lift. The same estimator scores both the baseline and every
   candidate, so the comparison stays fair.

To tighten the acceptance bar, supply a gate with a non-zero `min_gain`:

```python
from fireflyframework_datascience.features import CostBenefitGate
from fireflyframework_datascience.features.genai import GenAIFeatureEngineer

# Require more than +0.005 of lift before a feature is worth its complexity.
gate = CostBenefitGate(min_gain=0.005)
engineer = GenAIFeatureEngineer(proposer, gate=gate)
```

## See also

- [Datasets](datasets.md) — the `Dataset` the engineer consumes and `with_features` returns.
- [AutoML pipeline](automl.md) — where GenAI feature engineering fits in the end-to-end run.
- [Agentic loop](agentic-loop.md) — the broader propose-gate pattern across the framework.
- [LLM configuration](llm-configuration.md) — choosing and wiring the model behind the agent.
- [Security](security.md) — sandboxing and approval for model-generated code.
