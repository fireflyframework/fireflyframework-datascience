# Samples

**Every sample is runnable and covered by a test.** They live in
[`samples/`](https://github.com/fireflyframework/fireflyframework-datascience/tree/main/samples). The
first three run **offline with no LLM key**; the last two use real data / a real model.

| Sample | What it shows | Needs |
|---|---|---|
| [`tutorial.py`](https://github.com/fireflyframework/fireflyframework-datascience/blob/main/samples/tutorial.py) | The full guided tour — boot → validate → AutoML → GenAI features → agentic loop → serve | `tabular` |
| [`lumen_credit_risk.py`](https://github.com/fireflyframework/fireflyframework-datascience/blob/main/samples/lumen_credit_risk.py) | A focused credit-risk use case: GenAI discovers `debt_to_income`, AutoML serves the winner | `tabular` |
| [`genai_llm_showcase.py`](https://github.com/fireflyframework/fireflyframework-datascience/blob/main/samples/genai_llm_showcase.py) | **Real LLM** — Claude proposes features and reflects in the agentic loop; the gate decides | `tabular`, `genai`, an LLM key |
| [`industry_showcase.py`](https://github.com/fireflyframework/fireflyframework-datascience/blob/main/samples/industry_showcase.py) | The pipeline on **real finance & retail** data (OpenML credit-g, bank-marketing) | `tabular`, `data` |

## Run them

```bash
uv run python samples/tutorial.py            # offline, ~5 s
uv run python samples/lumen_credit_risk.py   # offline, ~10 s
uv run python samples/industry_showcase.py   # real OpenML data (network)

# real LLM — set a key first (see Configuring the LLM)
export ANTHROPIC_API_KEY=sk-ant-...
export FIREFLY_DATASCIENCE_GENAI__DEFAULT_MODEL=anthropic:claude-haiku-4-5
uv run python samples/genai_llm_showcase.py
```

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
unverified is adopted — exactly the governance described in [GenAI Feature Engineering](genai-features.md)
and the [Agentic Loop](agentic-loop.md).

## See also

- [Tutorial](tutorial.md) · [Configuring the LLM](llm-configuration.md) · [Benchmarks](benchmarks.md) ·
  [Use Case: Lumen](use-case-lumen.md)
