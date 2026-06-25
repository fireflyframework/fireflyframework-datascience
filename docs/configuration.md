# Configuration

**One typed settings model resolves your whole app — constructor args, environment, `.env`, and layered YAML, in that order.**

`FireflyDataScienceConfig` is a [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) model. Load it once at startup with `FireflyDataScienceConfig.load(...)` and everything — ML framework, tracking, GenAI, code execution, banner — resolves from a single, predictable precedence chain.

```python
from fireflyframework_datascience.core.config import FireflyDataScienceConfig

config = FireflyDataScienceConfig.load(config_dir="config", profiles=["prod"])
print(config.default_ml_framework)  # "sklearn"
print(config.genai.enabled)         # False
```

The model is classical-first by design: GenAI is **off** until you turn it on, and even then every GenAI call sits behind a cost/benefit gate.

!!! firefly "The LLM proposes; the classical engine decides"

    Configuration encodes the framework's governance posture. `genai.enabled` defaults to `False`
    and `genai.cost_benefit_gate` defaults to `True`, so the deterministic classical core runs unless
    you explicitly opt into the GenAI accelerator — and even then GenAI stays gated behind a measured
    improvement. You change one model; the whole app inherits the policy.

## Precedence

Values resolve from highest priority to lowest. Anything set at a higher level wins; a missing source is simply skipped.

| Priority | Source | How to set it |
| --- | --- | --- |
| 1 (highest) | **Constructor kwargs** | values passed directly to `FireflyDataScienceConfig(...)` |
| 2 | **Environment variables** | prefixed `FIREFLY_DATASCIENCE_`, nested via `__` |
| 3 | **`.env` file** | same naming as environment variables |
| 4 | **Profile YAML overlays** | `firefly-datascience-<profile>.yaml` (later profiles outrank earlier ones) |
| 5 | **Base YAML** | `firefly-datascience.yaml` |
| 6 (lowest) | **Field defaults** | the defaults shown below |

This ordering comes straight from `settings_customise_sources`, which returns `(init_settings, env_settings, dotenv_settings, *reversed(yaml_sources), file_secret_settings)` — earlier sources win, and reversing the YAML list lets profile overlays outrank the base file.

=== "Environment beats YAML"

    ```bash
    # Environment beats both YAML files and the field default:
    export FIREFLY_DATASCIENCE_DEFAULT_ML_FRAMEWORK=pytorch
    export FIREFLY_DATASCIENCE_GENAI__ENABLED=true        # nested via __
    export FIREFLY_DATASCIENCE_BANNER__MODE=MINIMAL
    ```

=== "Constructor beats everything"

    ```python
    # Constructor kwargs beat env, .env, YAML, and defaults (useful in tests):
    config = FireflyDataScienceConfig(default_ml_framework="xgboost")
    assert config.default_ml_framework == "xgboost"
    ```

### Environment variable naming

Every field is reachable from the environment using the `FIREFLY_DATASCIENCE_` prefix; nested models use the `__` delimiter once per level of nesting.

| Field path | Environment variable |
| --- | --- |
| `default_ml_framework` | `FIREFLY_DATASCIENCE_DEFAULT_ML_FRAMEWORK` |
| `tracking_enabled` | `FIREFLY_DATASCIENCE_TRACKING_ENABLED` |
| `banner.mode` | `FIREFLY_DATASCIENCE_BANNER__MODE` |
| `genai.enabled` | `FIREFLY_DATASCIENCE_GENAI__ENABLED` |
| `genai.budget_usd` | `FIREFLY_DATASCIENCE_GENAI__BUDGET_USD` |
| `execution.sandbox` | `FIREFLY_DATASCIENCE_EXECUTION__SANDBOX` |
| `execution.timeout_seconds` | `FIREFLY_DATASCIENCE_EXECUTION__TIMEOUT_SECONDS` |

!!! note "Two loader-only environment variables"

    `FIREFLY_DATASCIENCE_CONFIG_DIR` and `FIREFLY_DATASCIENCE_PROFILES` are read by `load` itself — not
    declared model fields — to discover the YAML directory and the active profiles. See
    [`load(config_dir, profiles)`](#loadconfig_dir-profiles).

## `load(config_dir, profiles)`

```python
@classmethod
def load(
    cls,
    *,
    config_dir: str | Path | None = None,
    profiles: list[str] | None = None,
) -> FireflyDataScienceConfig: ...
```

`load` is the supported entry point. It wires up the YAML sources, then constructs the model so env and `.env` overlays still apply.

- **`config_dir`** — directory holding the YAML files. Defaults to the `FIREFLY_DATASCIENCE_CONFIG_DIR` env var, then `.` (current directory).
- **`profiles`** — list of active profiles. When `None`, it reads the comma-separated `FIREFLY_DATASCIENCE_PROFILES` env var (e.g. `"dev,gpu"`).

```python
# Explicit arguments:
config = FireflyDataScienceConfig.load(config_dir="config", profiles=["dev", "gpu"])  # (1)!

# Driven entirely by the environment:
#   FIREFLY_DATASCIENCE_CONFIG_DIR=config
#   FIREFLY_DATASCIENCE_PROFILES=dev,gpu
config = FireflyDataScienceConfig.load()  # (2)!

print(config.profiles)  # ["dev", "gpu"]  # (3)!
```

1. Explicit `config_dir` and `profiles` take priority over the matching environment variables.
2. With no arguments, `load` reads `FIREFLY_DATASCIENCE_CONFIG_DIR` and the comma-separated `FIREFLY_DATASCIENCE_PROFILES`.
3. When profiles came from the loader (not from YAML), `load` back-fills the `profiles` field so the active profiles are visible on the returned config.

YAML files are discovered relative to `config_dir`:

```
config/
  firefly-datascience.yaml          # base — lowest YAML priority
  firefly-datascience-dev.yaml      # overlay for profile "dev"
  firefly-datascience-gpu.yaml      # overlay for profile "gpu" (outranks "dev")
```

A file that does not exist is skipped — only base and the overlays for active profiles are read.

## Configuration fields

### Top level

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `app_name` | `str` | `"firefly-datascience-app"` | Application name, shown in the banner. |
| `profiles` | `list[str]` | `[]` | Active profiles (populated by `load` if not set in YAML). |
| `default_ml_framework` | `str` | `"sklearn"` | Default ML framework for estimators. |
| `default_dataset_backend` | `str` | `"pandas"` | Default dataframe/dataset backend. |
| `tracking_enabled` | `bool` | `False` | Enable experiment tracking. |
| `model_registry_url` | `str \| None` | `None` | Model registry endpoint. |
| `feature_store_endpoint` | `str \| None` | `None` | Feature store endpoint. |

### `banner.*`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `banner.mode` | `BannerMode` | `BannerMode.TEXT` | Startup banner: `TEXT`, `MINIMAL`, or `OFF`. |

### `genai.*`

GenAI is **off by default** — the framework is classical-first.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `genai.enabled` | `bool` | `False` | Turn the GenAI accelerator on. |
| `genai.default_model` | `str` | `"openai:gpt-4o"` | Default LLM model id. |
| `genai.cost_benefit_gate` | `bool` | `True` | Gate GenAI calls behind a cost/benefit check. |
| `genai.budget_usd` | `float \| None` | `None` | Optional spend cap, in USD. |

### `execution.*`

Secure code-execution settings for LLM-generated code.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `execution.sandbox` | `"monty" \| "docker" \| "e2b" \| "local"` | `"monty"` | Sandbox runtime for generated code. |
| `execution.timeout_seconds` | `int` | `60` | Per-execution timeout. |
| `execution.require_approval` | `bool` | `True` | Require human approval before running generated code. |

## Profiles in practice

A profile is just a named YAML overlay. Keep a base file with shared settings, then add one overlay per environment or hardware target and activate them by name. Each tab below is a complete, self-contained overlay.

=== "Base"

    `config/firefly-datascience.yaml`

    ```yaml
    app_name: lumen-ds
    default_ml_framework: sklearn
    tracking_enabled: false
    banner:
      mode: TEXT
    genai:
      enabled: false
      default_model: openai:gpt-4o
      cost_benefit_gate: true
    execution:
      sandbox: monty
      timeout_seconds: 60
      require_approval: true
    ```

=== "gpu"

    `config/firefly-datascience-gpu.yaml`

    ```yaml
    default_ml_framework: pytorch
    execution:
      sandbox: docker
      timeout_seconds: 300
    ```

=== "prod"

    `config/firefly-datascience-prod.yaml`

    ```yaml
    tracking_enabled: true
    genai:
      enabled: true
      budget_usd: 25.0
    execution:
      require_approval: true
    ```

Activating the `gpu` profile overlays the base file; untouched keys fall back to base, then to field defaults:

```python
config = FireflyDataScienceConfig.load(config_dir="config", profiles=["gpu"])
assert config.default_ml_framework == "pytorch"   # overlay wins over base
assert config.tracking_enabled is False           # untouched key falls back to base
```

!!! tip "Stacking profiles"

    Pass more than one profile to compose overlays — `profiles=["dev", "gpu"]`. They apply in order,
    and a later profile outranks an earlier one for any key both set.

## The banner

`banner.mode` controls the startup banner. Build a printer from a loaded config:

```python
from fireflyframework_datascience.core.banner import BannerPrinter

printer = BannerPrinter.from_config(config, app_name="lumen-ds", app_version="1.0.0")
print(printer.render())
```

- `TEXT` — full ASCII art plus a status line (the default).
- `MINIMAL` — a single `:: Firefly DataScience :: (vX.Y.Z)` line.
- `OFF` — renders the empty string.

`from_config` carries the active profiles and `genai.enabled` into the printer, so the `TEXT` status line reflects the resolved config:

!!! success "Expected — `TEXT` status line"

    ```
    :: Firefly DataScience :: (v1.2.3)  app=lumen-ds v1.0.0  profiles=['gpu']  genai=off
    ```

The framework version is filled in automatically; `app=`, `profiles=`, and `genai=` come from the config and the arguments you pass to `from_config`.

Override the mode without touching YAML:

```bash
export FIREFLY_DATASCIENCE_BANNER__MODE=OFF
```

!!! warning "Enum values are case-sensitive"

    `BannerMode` is a string enum with members `TEXT`, `MINIMAL`, and `OFF`. Set the env var to one of
    those exact upper-case strings — `off` or `text` will not parse.

## See also

- [Getting Started](quickstart.md)
- [Configure the LLM](llm-configuration.md)
- [GenAI Accelerator](genai-features.md)
- [Code Execution & Security](security.md)
- [Architecture](architecture.md)
