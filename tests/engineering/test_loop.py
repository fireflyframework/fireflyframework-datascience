# Copyright 2026 Firefly Software Foundation.
"""Tests for the agentic ML-engineering loop (LLM-free via SequenceProposer)."""

from __future__ import annotations

from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader
from fireflyframework_datascience.engineering import SequenceProposer, SolutionCandidate
from fireflyframework_datascience.engineering.loop import AgenticAutoML, DeterministicVerifier


def test_verifier_requires_beating_baseline() -> None:
    verifier = DeterministicVerifier()
    ds = SklearnDatasetLoader().load("iris")
    cand = SolutionCandidate("random_forest")
    assert verifier.verify(ds, cand, score=0.95, baseline=0.5).valid is True
    assert verifier.verify(ds, cand, score=0.5, baseline=0.5).valid is False  # ran but useless
    assert verifier.verify(ds, cand, score=float("-inf"), baseline=0.5).valid is False


def test_loop_selects_best_verified_candidate() -> None:
    ds, _ = SklearnDatasetLoader().load("breast_cancer").train_test_split(random_state=0)
    proposer = SequenceProposer(
        [
            SolutionCandidate("linear"),
            SolutionCandidate("random_forest"),
            SolutionCandidate("does_not_exist"),  # invalid trainer -> -inf -> verdict invalid
        ]
    )
    run = AgenticAutoML(proposer, cv=3, max_iterations=5).solve(ds)

    assert run.n_iterations == 3
    assert run.best_candidate is not None
    assert run.best_candidate.trainer in {"linear", "random_forest"}
    assert run.best_score > run.baseline_score
    # the bogus trainer must be recorded but judged invalid
    invalid = [a for a in run.attempts if not a.verdict.valid]
    assert any(a.candidate.trainer == "does_not_exist" for a in invalid)
    assert len(run.valid_attempts) >= 1


def test_loop_fits_a_usable_model() -> None:
    train, test = SklearnDatasetLoader().load("breast_cancer").train_test_split(random_state=0)
    proposer = SequenceProposer([SolutionCandidate("random_forest"), SolutionCandidate("hist_gradient_boosting")])
    run = AgenticAutoML(proposer, cv=3).solve(train)

    assert run.model is not None
    preds = run.model.predict(test.X)
    assert len(preds) == test.n_rows


def test_agent_proposer_with_test_model() -> None:
    from pydantic_ai.models.test import TestModel

    from fireflyframework_datascience.engineering.loop import AgentSolutionProposer

    model = TestModel(
        custom_output_args={"trainer": "random_forest", "params_json": '{"n_estimators": 50}', "rationale": "x"}
    )
    proposer = AgentSolutionProposer(model=model)
    ds = SklearnDatasetLoader().load("iris")

    initial = proposer.propose_initial(ds, ["linear", "random_forest"])
    assert {c.trainer for c in initial} == {"linear", "random_forest"}

    nxt = proposer.propose_next(ds, [], ["linear", "random_forest"])
    assert nxt is not None
    assert nxt.trainer == "random_forest"
    assert nxt.params == {"n_estimators": 50}


def test_loop_task_is_respected() -> None:
    ds = SklearnDatasetLoader().load("iris")
    run = AgenticAutoML(SequenceProposer([SolutionCandidate("random_forest")]), cv=3).solve(ds)
    assert run.metric == "accuracy"  # multiclass default
    assert run.best_candidate is not None
