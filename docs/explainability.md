# Explainability

**Every fitted model can explain which features drive its predictions — with deterministic,
well-understood methods, never an LLM.**

Explainability is a first-class port in Firefly DataScience. After AutoML selects and refits a winner,
you get a model that can describe *why* it predicts what it predicts: globally (which features matter
across the dataset) and — with the optional SHAP adapter — locally (which features moved a single
prediction). This is table-stakes for regulated domains (lending, healthcare, insurance) where a model
you cannot explain is a model you cannot ship.

!!! firefly "Explanations are classical, not generated"

    Importances come from **permutation importance** (the dependency-free default) or **SHAP** — both
    deterministic, peer-reviewed methods. The LLM is never in the explanation path: just as GenAI
    *proposes* and a classical engine *decides*, here the classical engine also *explains*. The numbers
    are reproducible from a seed.

## Global feature importance

Every `AutoMLResult` can explain its winning model. Call `explain()` with a dataset (typically your
held-out split):

```python
from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader

ds = SklearnDatasetLoader().load("breast_cancer")
train, test = ds.train_test_split(test_size=0.25, random_state=0)

result = AutoML(cv=3, random_state=0).fit(train)
explanation = result.explain(test)          # (1)!

print(explanation.method)                    # "permutation_importance"
for name, importance in explanation.top(8):
    print(f"{name:<26} {importance:+.4f}")
```

1. `explain()` uses the DI-wired `ExplainerPort` when the result came from `AutoML.from_context`,
   otherwise the dependency-free permutation-importance explainer. It returns a `GlobalExplanation`.

!!! success "Expected (a real run on `breast_cancer`)"

    ```text
    winner: linear | holdout roc_auc: 0.9952
    permutation_importance
    radius error               +0.0182
    fractal dimension error    +0.0140
    mean concave points        +0.0091
    mean concavity             +0.0077
    compactness error          +0.0056
    worst area                 +0.0056
    worst symmetry             +0.0056
    perimeter error            +0.0049
    ```

    Each value is the mean drop in the model's score when that feature is randomly permuted — higher
    means more important. A pure-noise column lands at ≈ 0. Exact numbers depend on the data, the
    winning model, and the seed.

`GlobalExplanation` exposes `feature_importances` (a `dict[str, float]`), `std`, `baseline_score`,
`.top(k)`, and `.to_frame()` for a tidy pandas table.

## Local (per-prediction) explanations with SHAP

For per-prediction attributions — "why did *this* applicant score the way they did?" — install the
optional `explain` extra and the SHAP explainer is used automatically:

```bash
uv add "fireflyframework-datascience[explain]"     # adds shap
```

```python
from fireflyframework_datascience.explainability.adapters import ShapExplainer

explainer = ShapExplainer()
local = explainer.explain_local(result.best_model, test.X.iloc[:1])   # one LocalExplanation per row
for feature, contribution in local[0].top(5):
    print(f"{feature:<26} {contribution:+.4f}")
```

Each `LocalExplanation` carries the `prediction`, per-feature `contributions`, and a `base_value`;
`.top(k)` ranks features by absolute contribution.

## How it fits the architecture

| Piece | What it is |
| --- | --- |
| `ExplainerPort` | The `Protocol`: `supports(model)` + `explain_global(model, dataset)`. |
| `PermutationImportanceExplainer` | Default adapter — model-agnostic, scikit-learn only (no extra dependency). |
| `ShapExplainer` | Optional adapter (`explain` extra) — global **and** local attributions. |
| `ExplainabilityAutoConfiguration` | Registers the explainer (SHAP when installed, else permutation). |
| `AutoMLResult.explain(dataset)` | The handle most users hold — delegates to the wired explainer. |

Because it is a port, you can inject your own explainer (register an `ExplainerPort` bean and it wins),
and the DI-wired `AutoML.from_context(app)` automatically threads it into every result.

## See also

- [Classical AutoML](automl.md) — the engine that produces the model you explain.
- [Architecture](architecture.md) — how ports, adapters, and auto-configuration fit together.
- [GenAI features](genai-features.md) — the gated accelerator (proposals are explained the same way).
- [Security](security.md) — why generated code is treated as untrusted, and what is enforced today.
