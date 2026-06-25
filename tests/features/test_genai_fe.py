# Copyright 2026 Firefly Software Foundation.
"""Tests for the GenAI feature-engineering loop (LLM-free via injectable proposer)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.features import (
    CostBenefitGate,
    FeatureProposal,
    StaticFeatureProposer,
)
from fireflyframework_datascience.features.genai import GenAIFeatureEngineer


def _xor_dataset() -> Dataset:
    """y depends on the sign of a*b — an interaction a linear model cannot capture alone."""
    rng = np.random.RandomState(0)
    n = 400
    a = rng.uniform(-1, 1, n)
    b = rng.uniform(-1, 1, n)
    y = ((a * b) > 0).astype(int)
    X = pd.DataFrame({"a": a, "b": b})
    return Dataset("xor", X, pd.Series(y), task=TaskType.BINARY, target_name="y", feature_names=["a", "b"])


def _linear_scorer(task: TaskType):  # type: ignore[no-untyped-def]
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000)


def test_cost_benefit_gate() -> None:
    gate = CostBenefitGate(min_gain=0.01)
    assert gate.accepts(0.80, 0.85) is True
    assert gate.accepts(0.80, 0.805) is False  # below min_gain
    assert gate.accepts(0.80, 0.79) is False


def test_useful_feature_accepted() -> None:
    ds = _xor_dataset()
    proposer = StaticFeatureProposer([FeatureProposal("ab", "df['ab'] = df['a'] * df['b']", "interaction")])
    engineer = GenAIFeatureEngineer(proposer, scorer_estimator=_linear_scorer, cv=3)

    result = engineer.engineer(ds)
    assert len(result.accepted) == 1
    assert result.accepted[0].proposal.name == "ab"
    assert result.lift > 0.1  # the interaction feature lifts a linear model substantially
    assert "ab" in result.dataset.X.columns


def test_useless_feature_rejected() -> None:
    ds = _xor_dataset()
    proposer = StaticFeatureProposer([FeatureProposal("noise", "df['noise'] = 0.0", "no signal")])
    engineer = GenAIFeatureEngineer(proposer, scorer_estimator=_linear_scorer, cv=3)

    result = engineer.engineer(ds)
    assert len(result.accepted) == 0
    assert len(result.rejected) == 1
    assert "noise" not in result.dataset.X.columns


def test_unsafe_proposal_rejected_not_raised() -> None:
    ds = _xor_dataset()
    proposer = StaticFeatureProposer([FeatureProposal("evil", "import os\ndf['x'] = 1", "malicious")])
    engineer = GenAIFeatureEngineer(proposer, scorer_estimator=_linear_scorer, cv=3)

    result = engineer.engineer(ds)
    assert len(result.accepted) == 0
    assert len(result.rejected) == 1
    assert "nsafe" in result.rejected[0].reason


def test_agent_proposer_with_test_model() -> None:
    """The real agentic FireflyAgent integration, exercised LLM-free via pydantic-ai TestModel."""
    from pydantic_ai.models.test import TestModel

    from fireflyframework_datascience.features.genai import AgentFeatureProposer

    custom = TestModel(
        custom_output_args={"features": [{"name": "ab", "code": "df['ab'] = df['a'] * df['b']", "rationale": "x"}]}
    )
    proposer = AgentFeatureProposer(model=custom)
    proposals = proposer.propose(_xor_dataset(), max_features=3)
    assert len(proposals) == 1
    assert proposals[0].name == "ab"
    assert "df['ab']" in proposals[0].code
