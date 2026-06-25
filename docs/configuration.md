# Configuration

**One typed settings model resolves your whole app — constructor args, environment, `.env`, and layered YAML, in that order.**

`FireflyDataScienceConfig` is a [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) model. Load it once at startup with `FireflyDataScienceConfig.load(...)` and everything — ML framework, tracking, GenAI, code execution, banner — resolves from a single, predictable precedence chain.

```python
from fireflyframework_datascience.core.config import FireflyDataScienceConfig

config = FireflyDataScienceConfig.load(config_dir="config", profiles=["prod"])
print(config.default_ml_framework)  # "sklearn"
print(config.genai.enabled)         # False
```

## Precedence

Values resolve from highest priority to lowest:

1. **Constructor kwargs** — values passed directly to `FireflyDataScienceConfig(...)`.
2. **Environment variables** — prefixed `FIREFLY_DATASCIENCE_`, nested via `__`.
3. **`.env` file** — same naming as environment variables.
4. **Profile YAML overlays** — `firefly-datascience-<profile>.yaml` (later profiles outrank earlier ones).
5. **Base YAML** — `firefly-datascience.yaml`.
6. **Field defaults** — the defaults shown below.

Anything set at a higher level wins. A missing source is simply skipped.

```bash
# Environment beats both YAML files and the field default:
export FIREFLY_DATASCIENCE_DEFAULT_ML_FRAMEWORK=pytorch
export FIREFLY_DATASCIENCE_GENAI__ENABLED=true        # nested via __
export FIREFLY_DATASCIENCE_BANNER__MODE=MINIMAL
```

```python
# Constructor kwargs beat everything (useful in tests):
config = FireflyDataScienceConfig(default_ml_framework="xgboost")
assert config.default_ml_framework == "xgboost"
```

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
config = FireflyDataScienceConfig.load(config_dir="config", profiles=["dev", "gpu"])

# Driven entirely by the environment:
#   FIREFLY_DATASCIENCE_CONFIG_DIR=config
#   FIREFLY_DATASCIENCE_PROFILES=dev,gpu
config = FireflyDataScienceConfig.load()

print(config.profiles)  # ["dev", "gpu"]
```

YAML files are discovered relative to `config_dir`:

```
config/
  firefly-datascience.yaml          # base — lowest YAML priority
  firefly-datascience-dev.yaml      # overlay for profile "dev"
  firefly-datascience-gpu.yaml      # overlay for profile "gpu" (outranks "dev")
```

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

## Example YAML

`config/firefly-datascience.yaml` (base):

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

`config/firefly-datascience-gpu.yaml` (profile overlay):

```yaml
default_ml_framework: pytorch
execution:
  sandbox: docker
  timeout_seconds: 300
```

```python
config = FireflyDataScienceConfig.load(config_dir="config", profiles=["gpu"])
assert config.default_ml_framework == "pytorch"   # overlay wins over base
assert config.tracking_enabled is False           # untouched key falls back to base
```

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

Override it without touching YAML:

```bash
export FIREFLY_DATASCIENCE_BANNER__MODE=OFF
```

## See also

- [Getting Started](quickstart.md)
- [GenAI Accelerator](genai-features.md)
- [Code Execution](security.md)
