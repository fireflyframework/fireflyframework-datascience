# Samples

**Every sample is a single, runnable script that is covered by a test — so each one is guaranteed to work.**

The four scripts live in
[`samples/`](https://github.com/fireflyframework/fireflyframework-datascience/tree/main/samples).
Three run **offline with no LLM key** (the GenAI steps use deterministic stand-in proposers); one
calls a **real LLM**. Pick the card that matches what you want to see, copy its run command, and go.

!!! firefly "The pattern every sample demonstrates — the LLM proposes; the classical engine decides"

    GenAI proposes feature code and model candidates; a deterministic engine cross-validates and
    measures; and the cost/benefit gate keeps only what beats a seeded baseline. The offline samples
    swap the LLM for a fixed proposer so the exact same gate runs without a key — see
    [GenAI Feature Engineering](genai-features.md) and the [Agentic Loop](agentic-loop.md).

## The four samples

<div class="grid cards" markdown>

-   :material-map-outline:{ .lg .middle } __`tutorial.py` — the full guided tour__

    ---

    The whole framework end-to-end on a synthetic credit-risk dataset: boot → validate → classical
    AutoML → GenAI feature engineering → agentic loop → serve. Runs **offline**; it uses deterministic
    stand-in proposers and prints exactly how to switch on a real LLM. Needs the `tabular` extra.

    ```bash
    uv run python samples/tutorial.py
    ```

    [:octicons-arrow-right-24: Walkthrough](tutorial.md)

-   :material-bank-outline:{ .lg .middle } __`lumen_credit_risk.py` — a focused use case__

    ---

    One realistic (synthetic) lending dataset where default risk is driven by debt-to-income.
    GenAI feature engineering discovers `debt_to_income`, the gate keeps it because it measurably
    lifts the score, AutoML selects the winner, and it is served to score a new applicant. Runs
    **offline** (`StaticFeatureProposer`). Needs the `tabular` extra.

    ```bash
    uv run python samples/lumen_credit_risk.py
    ```

    [:octicons-arrow-right-24: Use case: Lumen](use-case-lumen.md)

-   :material-database-outline:{ .lg .middle } __`industry_showcase.py` — real public data__

    ---

    The full pipeline (load → validate → AutoML → holdout evaluation) on genuine, mixed-type data
    with categorical features, loaded straight from OpenML — no Kaggle account needed. Two cases:
    `credit-g` (German credit risk) and `bank-marketing` (campaign conversion). Needs the `tabular`
    and `data` extras and network access.

    ```bash
    uv run python samples/industry_showcase.py
    ```

    [:octicons-arrow-right-24: Datasets](datasets.md)

-   :material-creation-outline:{ .lg .middle } __`genai_llm_showcase.py` — a real LLM__

    ---

    The only sample that calls a real model (Claude / GPT / …) for **both** GenAI feature engineering
    and the agentic ML-engineering loop. Model and credentials come from the environment — nothing is
    hard-coded. Needs the `tabular` and `genai` extras and an LLM key.

    ```bash
    export ANTHROPIC_API_KEY=sk-ant-...  # (1)!
    uv run python samples/genai_llm_showcase.py
    ```

    [:octicons-arrow-right-24: Configuring the LLM](llm-configuration.md)

</div>

1.  Or `OPENAI_API_KEY` / `GEMINI_API_KEY`. The model string defaults to
    `anthropic:claude-haiku-4-5`; override it with
    `export FIREFLY_DATASCIENCE_GENAI__DEFAULT_MODEL=...`. The script exits cleanly with a message if
    no key is set.

## Run them

=== "Offline (no key)"

    ```bash
    uv run python samples/tutorial.py            # the full guided tour
    uv run python samples/lumen_credit_risk.py   # focused credit-risk use case
    uv run python samples/industry_showcase.py   # real OpenML data (needs network)
    ```

=== "Real LLM"

    ```bash
    export ANTHROPIC_API_KEY=sk-ant-...                                       # or OPENAI_API_KEY=...
    export FIREFLY_DATASCIENCE_GENAI__DEFAULT_MODEL=anthropic:claude-haiku-4-5  # optional; this is the default
    uv run python samples/genai_llm_showcase.py
    ```

!!! tip "The offline samples are the place to start"

    `tutorial.py` and `lumen_credit_risk.py` run the *same* cost/benefit gate as the real-LLM
    showcase — the only difference is who proposes the features and models. Both finish in a few
    seconds offline (≈5–10 s on a laptop), so you can see the full governance loop without spending a token.

## What the real-LLM showcase produces

A representative run with `anthropic:claude-haiku-4-5`:

```
[1] GenAI feature engineering — the LLM proposes, the gate decides:
    ✓ accepted loan_to_income_ratio     gain=+0.0013   df['loan_to_income_ratio'] = df['loan_amount'] / (df['income'] + 1)
    ✓ accepted default_risk_score       gain=+0.0006   df['default_risk_score'] = df['num_prior_defaults'] * 100 + ...
    ✗ rejected employment_to_loan_ratio (no measured lift)
    → 2 accepted, 4 rejected; roc_auc 0.7875 -> 0.7895

[2] Agentic ML-engineering loop — the LLM reflects, the engine verifies:
    · linear  {}  score=0.9939   ...   · linear {'C': 0.15, 'penalty': 'l1', ...} score=0.9955
    → 9 attempts (9 verified); best=linear roc_auc=0.9955
```

The model proposes; the deterministic engine measures; the gate keeps only what is proven. Nothing
unverified is adopted — exactly the governance described in
[GenAI Feature Engineering](genai-features.md) and the [Agentic Loop](agentic-loop.md).

!!! note "LLM output is non-deterministic"

    The exact feature names, gains and attempt counts vary run to run — the LLM is free to propose
    anything. What is invariant is the gate: a proposal is accepted only if it lifts a seeded,
    cross-validated baseline, so the *decision* is always reproducible even when the *proposal* is not.

## Benchmark scenarios

For evaluation rather than demonstration, the
[`benchmarks/`](https://github.com/fireflyframework/fireflyframework-datascience/tree/main/benchmarks)
directory holds the harnesses; all results live in
[`benchmarks/RESULTS.md`](https://github.com/fireflyframework/fireflyframework-datascience/blob/main/benchmarks/RESULTS.md).

| Script | What it measures |
|---|---|
| `automl_benchmark.py` | Tier-2 offline smoke suite (scikit-learn datasets) |
| `amlb_benchmark.py` | Tier-1 OpenML-CC18 (AMLB-style), real categorical data |
| `scientific_eval.py` | **Nested 5-fold CV** vs fixed single models + Wilcoxon significance (unbiased) |
| `genai_value.py` | **Controlled ablation** of GenAI feature engineering with a real LLM (+ cost) |
| `beat_baseline.py` | A quick cross-validated head-to-head vs a default baseline |

## See also

- [Tutorial](tutorial.md) — the step-by-step walkthrough behind `tutorial.py`.
- [Use case: Lumen](use-case-lumen.md) — the credit-risk story behind `lumen_credit_risk.py`.
- [Configuring the LLM](llm-configuration.md) — providers, model strings and keys for the real-LLM sample.
- [Datasets](datasets.md) — the OpenML loader used by `industry_showcase.py`.
- [Benchmarks](benchmarks.md) — the evaluation harnesses and their results.
