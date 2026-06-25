# Agentic ML-engineering loop

**An LLM proposes, the classical engine decides — a deterministic verifier, not "it ran", is the judge.**

The agentic loop realizes the SOTA AutoML pattern grounded on a deterministic executor: an LLM
*proposes* a solution (a trainer plus hyperparameters), the classical engine *trains and
cross-validates* it, and a **Verifier** — a stage distinct from execution-success — decides whether
the result is genuinely good. Search is greedy with reflection over the attempt history, bounded by
an iteration and patience budget.

The whole cycle is **propose → train/CV → verify → reflect → select**.

!!! firefly "The recurring pattern — the LLM proposes; the classical engine decides"

    The LLM never gets the last word. It *suggests* the next `(trainer, params)` candidate; the
    classical engine cross-validates it and a deterministic `Verifier` rules on whether it actually
    beats a trivial baseline. A candidate that runs but fails to clear the baseline is rejected — so
    the loop can only ever return a model that measurably earned its place.

<p align="center">
  <img src="img/agentic-loop.svg" alt="The agentic ML-engineering loop" width="85%">
</p>

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

`SolutionCandidate`, `Verdict`, and `AttemptRecord` are frozen dataclasses; `EngineeringRun` carries
the trace and the refit model. The two proposers both satisfy the `CandidateProposer` protocol, and
`DeterministicVerifier` satisfies the `Verifier` protocol — so any of them can be swapped for a custom
implementation.

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
```

!!! success "Expected"

    ```text
    Agentic AutoML: 3 attempts (2 verified); best=hist_gradient_boosting roc_auc=0.9123 (baseline 0.5000)
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

```python
for candidate in self._proposer.propose_initial(dataset, names):  # (1)!
    record = self._attempt(dataset, candidate, task, scoring, baseline)  # (2)!
    attempts.append(record)
    if record.verdict.valid and record.score > best_score:  # (3)!
        best, best_score = candidate, record.score

patience = self._patience
for _ in range(self._max_iterations):
    candidate = self._proposer.propose_next(dataset, attempts, names)  # (4)!
    if candidate is None:
        break
    record = self._attempt(dataset, candidate, task, scoring, baseline)
    attempts.append(record)
    if record.verdict.valid and record.score > best_score:
        best, best_score, patience = candidate, record.score, self._patience  # (5)!
    else:
        patience -= 1
        if patience <= 0:
            break
```

1. **Seed** — `propose_initial` returns the starting population, evaluated before any reflection.
2. **Train / CV + verify** — `_attempt` cross-validates the candidate and asks the `Verifier` for a
   `Verdict` in one step.
3. **Select** — only a `valid` verdict that *improves* on the running best can take the lead.
4. **Reflect** — `propose_next` receives the full `attempts` history; a returned `None` ends the loop.
5. **Patience reset** — an improving verified attempt restores the full patience budget; a
   non-improving one decrements it, stopping the loop at zero.

## DeterministicVerifier — correctness, not execution

A run that produces a number is not the same as a run that produced a *good* number.

!!! note "Correctness ≠ ran"

    A candidate that trained and returned a score has only proven it *executed*. Verification is a
    separate stage: `DeterministicVerifier` demands a *finite* score that *beats the trivial
    baseline* by a `margin`. Anything else is rejected — execution-success is never mistaken for
    correctness.

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

Both built-in proposers satisfy the `CandidateProposer` protocol — pick the LLM-backed one for real
search, or the deterministic one for tests and fixed strategies.

=== "AgentSolutionProposer (LLM)"

    `AgentSolutionProposer` seeds every trainer at its defaults, then reflects on the ranked history
    via a `FireflyAgent`. The LLM client is built lazily on first reflection — no client is created at
    startup:

    ```python
    from fireflyframework_datascience.engineering.loop import AgenticAutoML, AgentSolutionProposer

    proposer = AgentSolutionProposer(model="openai:gpt-4o")
    run = AgenticAutoML(proposer, max_iterations=10).solve(dataset)
    ```

    On each `propose_next`, the agent receives the task, the allowed trainers, and the best-first
    attempt history (top 8), and returns a structured `(trainer, params_json, rationale)`. If the
    model names a trainer outside the allowed list, the proposer falls back to the best trainer seen so
    far; malformed `params_json` degrades to `{}`. You can also inject a pre-built agent with
    `AgentSolutionProposer(agent=my_agent)`.

=== "SequenceProposer (deterministic)"

    For tests or fixed search plans, `SequenceProposer` replays a fixed candidate list — the first is
    the seed, the rest are dispensed one per `propose_next`:

    ```python
    from fireflyframework_datascience.engineering import SequenceProposer, SolutionCandidate

    proposer = SequenceProposer([
        SolutionCandidate("linear", rationale="cheap baseline"),
        SolutionCandidate("random_forest", {"max_depth": 8}),
    ])
    ```

To write your own, implement the `CandidateProposer` protocol: `propose_initial(dataset, trainers)`
and `propose_next(dataset, history, trainers)`.

!!! tip "Which trainers are allowed"

    The `trainers` list a proposer sees comes from the loop's registry. By default that is `linear`,
    `random_forest`, and `hist_gradient_boosting`, plus `xgboost`, `lightgbm`, and `catboost` when
    those optional libraries are installed. Pass `trainers=...` to `AgenticAutoML` to constrain or
    extend the search space.

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

`n_iterations` and `valid_attempts` are derived from `attempts`, and `summary()` renders the one-line
recap shown under [Quick start](#quick-start).

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

!!! warning "Patience only counts after seeding"

    The initial population from `propose_initial` is always fully evaluated; patience and
    `max_iterations` bound only the reflection rounds that follow. A run with an empty or trivial seed
    still respects the iteration budget.

## See also

- [Datasets](datasets.md) — the `Dataset` the loop searches over.
- [AutoML](automl.md) — the trainer registry, metrics, and the preprocessing pipeline wrapped around every candidate.
- [GenAI features](genai-features.md) — other places the LLM proposes and the classical engine decides.
- [LLM configuration](llm-configuration.md) — wiring the model behind `AgentSolutionProposer`.
- [Architecture](architecture.md) — the ports and adapters the loop plugs into.
