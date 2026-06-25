# Agentic ML-Engineering Loop

**An LLM proposes, the classical engine decides — a deterministic verifier, not "it ran", is the judge.**

The agentic loop realizes the SOTA AutoML pattern grounded on a deterministic executor: an LLM
*proposes* a solution (a trainer plus hyperparameters), the classical engine *trains and
cross-validates* it, and a **Verifier** — a stage distinct from execution-success — decides whether
the result is genuinely good. Search is greedy with reflection over the attempt history, bounded by
an iteration and patience budget.

The whole cycle is: **propose → train/CV → verify → reflect → select**.

## The pieces

| Type | Role |
| --- | --- |
| `SolutionCandidate` | A proposal: `trainer`, `params`, `rationale`. |
| `Verdict` | The verifier's judgement: `valid`, `reason`, `score`. |
| `AttemptRecord` | One iteration: `candidate`, `score`, `verdict`. |
| `EngineeringRun` | The full trace of a run, plus the fitted best model. |
| `AgenticAutoML` | The loop engine (`AgenticLoopPort`). |
| `DeterministicVerifier` | Correctness check: finite + beats the trivial baseline. |
| `AgentSolutionProposer` | LLM-backed proposer (reflects via a `FireflyAgent`). |
| `SequenceProposer` | Deterministic proposer for tests / fixed strategies. |

## Quick start

`AgenticAutoML` takes any proposer and runs the loop over a `Dataset`:

```python
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.engineering import SolutionCandidate, SequenceProposer
from fireflyframework_datascience.engineering.loop import AgenticAutoML

dataset = Dataset.from_frame(df, target="churned")

proposer = SequenceProposer([
    SolutionCandidate("linear"),
    SolutionCandidate("random_forest", {"n_estimators": 300}),
    SolutionCandidate("hist_gradient_boosting", {"learning_rate": 0.05}),
])

loop = AgenticAutoML(proposer, max_iterations=8, patience=3, cv=5)
run = loop.solve(dataset)

print(run.summary())
# Agentic AutoML: 3 attempts (2 verified); best=hist_gradient_boosting roc_auc=0.9123 (baseline 0.5000)
```

The engine seeds the candidates from `propose_initial`, then repeatedly calls `propose_next` until a
candidate is `None`, the iteration budget is spent, or patience runs out.

## propose → train/CV → verify → reflect → select

The engine never trusts a candidate just because it executed. Each attempt is:

1. **propose** — the proposer yields a `SolutionCandidate`.
2. **train / CV** — the candidate is wrapped in a preprocessing pipeline and scored with
   `sklearn`'s `cross_val_score` (default `cv=5`). A failing candidate scores `-inf` and never
   aborts the loop.
3. **verify** — the `Verifier` turns a raw score into a `Verdict`. Only `valid` verdicts can
   become the best.
4. **reflect** — `propose_next` is handed the full `history` to inform the next proposal.
5. **select** — the highest-scoring *verified* candidate wins and is refit on all data.

## DeterministicVerifier — correctness, not execution

A run that produces a number is not the same as a run that produced a *good* number.
`DeterministicVerifier` requires a finite score that beats the trivial baseline (a `DummyClassifier`
with `strategy="prior"` / `DummyRegressor` with `strategy="mean"`) by a `margin`:

```python
from fireflyframework_datascience.engineering.loop import DeterministicVerifier

verifier = DeterministicVerifier(margin=0.01)   # must beat baseline by at least 0.01
loop = AgenticAutoML(proposer, verifier=verifier)
```

Verdicts read like a review:

```python
Verdict(valid=False, reason="training failed or produced a non-finite score", score=-inf)
Verdict(valid=False, reason="does not beat the trivial baseline (0.5010 <= 0.5000)", score=0.5010)
Verdict(valid=True,  reason="beats trivial baseline by +0.4123", score=0.9123)
```

You can supply any object implementing the `Verifier` protocol
(`verify(dataset, candidate, score, baseline) -> Verdict`).

## Proposers

### AgentSolutionProposer — the LLM in the loop

`AgentSolutionProposer` seeds every trainer at its defaults, then reflects on the ranked history via a
`FireflyAgent`. The LLM client is built lazily on first reflection — no client is created at startup:

```python
from fireflyframework_datascience.engineering.loop import AgenticAutoML, AgentSolutionProposer

proposer = AgentSolutionProposer(model="openai:gpt-4o")
run = AgenticAutoML(proposer, max_iterations=10).solve(dataset)
```

On each `propose_next`, the agent receives the task, the allowed trainers, and the best-first
attempt history, and returns a structured `(trainer, params_json, rationale)`. If the model names a
trainer outside the allowed list, the proposer falls back to the best trainer seen so far; malformed
`params_json` degrades to `{}`. You can also inject a pre-built agent with `AgentSolutionProposer(agent=my_agent)`.

### SequenceProposer — deterministic strategies

For tests or fixed search plans, `SequenceProposer` replays a fixed candidate list — the first is the
seed, the rest are dispensed one per `propose_next`:

```python
from fireflyframework_datascience.engineering import SequenceProposer, SolutionCandidate

proposer = SequenceProposer([
    SolutionCandidate("linear", rationale="cheap baseline"),
    SolutionCandidate("random_forest", {"max_depth": 8}),
])
```

To write your own, implement the `CandidateProposer` protocol: `propose_initial(dataset, trainers)`
and `propose_next(dataset, history, trainers)`.

## The EngineeringRun trace

`solve` returns an `EngineeringRun` — a full, auditable trace plus the refit best model:

```python
run = loop.solve(dataset)

run.best_candidate      # SolutionCandidate | None
run.best_score          # float (nan if nothing verified)
run.model               # the refit Model (None if nothing verified)
run.metric              # e.g. "roc_auc"
run.baseline_score      # the trivial baseline it had to beat
run.n_iterations        # total attempts
run.valid_attempts      # only the verified ones

for a in run.attempts:
    print(a.candidate.trainer, a.score, a.verdict.valid, a.verdict.reason)
```

## Budgets: iterations and patience

The loop is greedy with two knobs:

- `max_iterations` (default `8`) — the hard cap on reflection rounds after seeding.
- `patience` (default `3`) — consecutive non-improving attempts allowed before early stopping.

Each improving verified attempt resets patience to the full budget; each non-improving one decrements
it, and the loop stops when it hits zero. Tune the trade-off between thoroughness and cost:

```python
loop = AgenticAutoML(
    proposer,
    cv=5,
    max_iterations=12,   # explore more
    patience=4,          # tolerate more dead ends
    random_state=42,
)
```

## See also

- [Datasets](datasets.md) — the `Dataset` the loop searches over.
- [Models & trainers](models.md) — the trainer registry candidates draw from.
- [Evaluation](evaluation.md) — metrics, scoring, and the trivial baseline.
- [Preprocessing](preprocessing.md) — the pipeline wrapped around every candidate.
