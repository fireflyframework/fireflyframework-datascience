# Security Model

**Firefly DataScience treats LLM-generated code as hostile input: it is statically vetted, run with a stripped namespace, and — for untrusted data — pushed behind a sandbox or a human.**

The GenAI accelerators (CAAFE-style automated feature engineering, agentic analysis) ask a model to *write Python that runs against your data*. That is an attack surface. The framework's job is to make the default path safe even when the model is wrong, compromised, or steered by adversarial data. This page describes the trust model, the controls that enforce it, and — importantly — where those controls stop.

!!! warning "GenAI is off until you enable it"
    Firefly is classical-first. `genai.enabled` defaults to **`False`** ([`GenAIConfig`](configuration.md)), so none of the code-generating accelerators run unless you opt in. Until then there is no LLM, no generated code, and no executor invocation — the secure default is *nothing executes*. Everything below describes the controls that engage **once you turn GenAI on**.

<p align="center">
  <img src="img/security.svg" alt="Secure-by-default execution tiers" width="85%">
</p>

## Threat model

The model is **not** trusted. We assume any of:

- A model emits code that exfiltrates data (`open`, `socket`, `requests`) or escalates (`os.system`, `subprocess`).
- The data itself carries a **prompt injection** — a column value or header crafted to make the model write malicious code.
- The model hallucinates code that corrupts the feature frame or hangs.

We *do* trust the host process, the installed libraries (`pandas`, `numpy`), and the configuration. The goal is: a wrong or hostile model snippet cannot do more than fail loudly.

!!! firefly "The LLM proposes; the executor decides"
    The model only ever produces *candidate text*. It cannot import, open files, reach the network, or mutate your data. The classical executor — `FeatureCodeExecutor` — is the part that actually runs anything, and it does so under a static analysis pass, a minimal builtins allowlist, and a strict result contract. The proposal is cheap and untrusted; the decision to run is governed.

## Layer 1 — static safety analysis

Before a single byte of model output executes, `FeatureCodeExecutor` runs it through the static analyzer reused from `fireflyframework_agentic.execution`. The policy denies dangerous modules, dangerous builtins, and **all dunder access** (which is how sandbox escapes are typically built):

```python
from fireflyframework_agentic.execution import SafetyPolicy

policy = SafetyPolicy(
    denied_modules=frozenset(
        {"os", "sys", "subprocess", "shutil", "socket", "pathlib", "importlib", "builtins"}
    ),
    denied_builtins=frozenset(
        {"eval", "exec", "compile", "open", "__import__",
         "input", "globals", "locals", "vars", "getattr", "setattr"}
    ),
    deny_dunder_access=True,  # blocks ().__class__.__bases__... escapes
)
```

`FeatureCodeExecutor` constructs exactly this policy in its `__init__`, so you do not configure it by hand — but you should understand it. Analysis is AST-based and happens *before* execution:

```python
from fireflyframework_datascience.features.executor import (
    FeatureCodeExecutor,
    FeatureExecutionError,
)

executor = FeatureCodeExecutor()

# Rejected statically — never runs:
try:
    executor.execute("import os; os.system('rm -rf /')", X)  # (1)!
except FeatureExecutionError as exc:
    print(exc)  # Unsafe feature code rejected: ...
```

1. `os` is in `denied_modules`, so `analyze_code` reports the snippet as unsafe and `execute` raises before the `exec` call is ever reached.

Internally the executor calls `analyze_code(code, policy)` and refuses to proceed unless `report.safe` is true, surfacing each `report.violations[*].message` in the raised `FeatureExecutionError`.

!!! success "Expected"
    ```text
    Unsafe feature code rejected: ...
    ```
    A rejected snippet raises a typed `FeatureExecutionError` (a subclass of `FireflyDataScienceError`) — it does not run, does not return a partial frame, and does not leak a stack trace from inside the model's code.

## Layer 2 — restricted execution

Code that passes static analysis is still not trusted. It runs via `exec` with a **minimal `__builtins__` allowlist** and a namespace that exposes only the dataframe and the two numeric libraries:

```python
# Inside FeatureCodeExecutor.execute, conceptually:
namespace = {"df": X.copy(), "pd": pd, "np": np}  # (1)!
exec(compile(code, "<feature>", "exec"), {"__builtins__": _SAFE_BUILTINS}, namespace)  # (2)!
```

1. The frame is a **copy** (`X.copy()`) — model code mutates `df` in place, never the caller's original data.
2. The global `__builtins__` is replaced by `_SAFE_BUILTINS`, so the usual escape hatches (`__import__`, `open`, `eval`) simply do not exist in scope.

`_SAFE_BUILTINS` is a hand-picked set — `abs`, `min`, `max`, `sum`, `round`, `len`, `range`, `enumerate`, `zip`, `map`, `filter`, `sorted`, the numeric/collection constructors (`float`, `int`, `bool`, `str`, `list`, `dict`, `tuple`, `set`), and `pow`. There is no `open`, no `__import__`, no I/O. Key properties:

- The frame is a **copy** (`X.copy()`) — model code cannot mutate the caller's data in place.
- The contract is **pandas/numpy transforms only**, never arbitrary capability. This is the CAAFE pattern.
- The result must be a `DataFrame`, must add at least one **new column**, and every new column must be **numeric and finite** (`inf`/`-inf` are replaced with `NaN`). Anything else raises `FeatureExecutionError`.

```python
# Valid CAAFE-style snippet — adds a numeric feature, mutates `df`:
code = "df['amount_per_day'] = df['amount'] / df['tenure_days'].clip(lower=1)"
X_enriched = executor.execute(code, X)
```

The post-conditions are enforced in order: a non-`DataFrame` result raises `Feature code must leave a pandas DataFrame in `df``; a frame with no new columns raises `Feature code added no new column`; a non-numeric new column raises `New feature <name> is not numeric`. Every failure mode is a typed `FeatureExecutionError`, so downstream estimators never receive a malformed frame.

## Layer 3 — the tiered sandbox

Layers 1 and 2 run **in-process**. They block the obvious capabilities, but a determined escape against a CPython process is never something to bet sensitive data on. For untrusted data, escalate isolation with `execution.sandbox` in `ExecutionConfig`:

```python
from fireflyframework_datascience.core.config import FireflyDataScienceConfig

config = FireflyDataScienceConfig.load(profiles=["prod"])
config.execution.sandbox          # "monty" | "docker" | "e2b" | "local"
config.execution.timeout_seconds  # 60 by default
config.execution.require_approval # True by default — HITL gate
```

The tiers, from least to most isolated:

| `sandbox` | Isolation | Use for |
|-----------|-----------|---------|
| `local`   | None — host process, restricted exec only | trusted data, dev only |
| `monty`   | In-process restricted interpreter (default) | typical CAAFE on owned data |
| `docker`  | OS-level container, no host network/FS | untrusted data |
| `e2b`     | Remote ephemeral microVM | untrusted data at higher assurance |

The literal type for `sandbox` is exactly `Literal["monty", "docker", "e2b", "local"]`, defaulting to `"monty"` — any other value fails validation at load time. Set it via env or YAML — never hardcode `local` for production:

=== "Environment variables"

    ```bash
    export FIREFLY_DATASCIENCE_EXECUTION__SANDBOX=docker  # (1)!
    export FIREFLY_DATASCIENCE_EXECUTION__TIMEOUT_SECONDS=30
    ```

    1. The `FIREFLY_DATASCIENCE_` prefix and the `__` nested delimiter come straight from `SettingsConfigDict` on `FireflyDataScienceConfig`. `EXECUTION__SANDBOX` maps onto `config.execution.sandbox`.

=== "Profile YAML"

    ```yaml
    # firefly-datascience-prod.yaml
    execution:
      sandbox: e2b
      timeout_seconds: 30
      require_approval: true
    ```

    Profile overlays outrank the base `firefly-datascience.yaml`, so a `prod` profile can tighten isolation without touching the base file. See [Configuration](configuration.md) for the full precedence order.

Beyond the strongest sandbox sits **HITL** (human-in-the-loop): when `execution.require_approval` is `True` (the default), generated code is surfaced for human approval before it runs. This is the final tier — a person, not a policy, signs off.

!!! note "Defaults are the safe end of every axis"
    Out of the box, `sandbox = "monty"` (in-process restricted interpreter), `timeout_seconds = 60`, and `require_approval = True`. You loosen these deliberately — and only `local` removes isolation entirely.

## Prompt-injection-via-data defense

The subtle attack is not the model going rogue on its own; it is a **column value or header that steers the model** into writing malicious code. Firefly's answer is *defense in depth that does not trust the model's intent*:

1. **Static analysis is content-blind.** It rejects `os`, `subprocess`, `socket`, dunder access, and `eval`/`exec`/`open` regardless of *why* the model wrote them — so a successful injection still produces code that gets rejected.
2. **The restricted namespace** means even "clever" injected code has no I/O, no imports, no host reach.
3. **The numeric-new-column contract** means injected code that tries to do anything other than add a numeric feature fails the post-conditions.
4. **Sandboxing + HITL** mean that for genuinely untrusted data you route to `docker`/`e2b` and require approval — so injection cannot silently reach a capability.

!!! warning "The framework does not read your data's meaning"
    Firefly cannot inspect or sanitize the *semantics* of your data. Prompt-injection defense rests on capability restriction and sandboxing, not on detecting malicious text. Treat data of unknown provenance as untrusted input: raise `execution.sandbox` and keep `require_approval` on.

## Governance — the CostBenefitGate

GenAI is **off by default** (`genai.enabled = False`) — Firefly is classical-first. When you do enable it, the `CostBenefitGate` is the governance control: it decides whether an LLM call is *worth it* before spending tokens, bounded by a budget.

```python
config.genai.enabled            # False by default
config.genai.cost_benefit_gate  # True — gate LLM spend on expected benefit
config.genai.budget_usd         # optional hard ceiling (float | None), e.g. 5.00
```

```yaml
# firefly-datascience.yaml
genai:
  enabled: true
  default_model: "openai:gpt-4o"
  cost_benefit_gate: true
  budget_usd: 5.00
```

!!! firefly "Two orthogonal gates: how much, and what"
    The `CostBenefitGate` is a *governance* control, not a security control: it limits spend and runaway agentic loops, not capability. Keep both axes in mind — `cost_benefit_gate` governs **how much** the model runs; the executor and sandbox govern **what its output may do**. Neither substitutes for the other.

## Limits of the trust model

Be precise about what these controls do and do not give you:

- In-process tiers (`local`, `monty`) **reduce** but do not eliminate the risk of a CPython sandbox escape. For untrusted data use `docker`/`e2b`.
- Static analysis is an **allowlist of capabilities**, not a proof of harmlessness. Code can still be wrong, slow (mitigated by `timeout_seconds`), or compute misleading features.
- The framework does not sanitize your **data's content** — prompt-injection defense rests on capability restriction and sandboxing, not on detecting malicious text.
- `require_approval` is only as strong as the human approving. Do not rubber-stamp generated code.
- Secrets in the host environment are visible to `local`/`monty` execution. Do not run untrusted-data jobs in a process holding production credentials.

!!! tip "Secure default, stated once"
    `genai.enabled = False`; when enabled, `sandbox = "monty"`, `require_approval = True`, `cost_benefit_gate = True`. Relax deliberately, per profile, never globally.

## See also

- [Configuration](configuration.md) — `ExecutionConfig`, `GenAIConfig`, and the resolution precedence
- [LLM configuration](llm-configuration.md) — wiring a model once GenAI is enabled
- [GenAI features](genai-features.md) — the CAAFE accelerator the executor protects
- [Agentic loop](agentic-loop.md) — where generated code and the `CostBenefitGate` meet
- [Getting started](quickstart.md) — the classical-first default path
