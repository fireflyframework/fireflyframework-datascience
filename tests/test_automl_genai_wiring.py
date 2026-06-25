# Copyright 2026 Firefly Software Foundation.
"""When a FeatureEngineerPort is wired, AutoML.fit must engineer features first and train on them.

This closes the integrity gap where enabling GenAI did not actually change AutoML: the engineer was a
registered bean the facade never consumed. Real data (breast_cancer), LLM-free (StaticFeatureProposer).
"""

from __future__ import annotations


def test_automl_applies_a_wired_feature_engineer() -> None:
    from sklearn.datasets import load_breast_cancer

    from fireflyframework_datascience.automl import AutoML
    from fireflyframework_datascience.core.types import TaskType
    from fireflyframework_datascience.datasets import Dataset
    from fireflyframework_datascience.features import CostBenefitGate, FeatureProposal, StaticFeatureProposer
    from fireflyframework_datascience.features.genai import GenAIFeatureEngineer

    raw = load_breast_cancer(as_frame=True)
    ds = Dataset(
        name="breast_cancer",
        X=raw.data,
        y=raw.target,
        task=TaskType.BINARY,
        target_name="target",
        feature_names=list(raw.data.columns),
    )
    proposer = StaticFeatureProposer(
        [FeatureProposal("rad_area", "df['rad_area'] = df['mean radius'] * df['mean area']", "interaction")]
    )
    # A permissive gate so the test asserts the *wiring* (engineer runs, its dataset is used), not the
    # lift magnitude — the accept/reject logic is covered by the feature-engineering tests.
    engineer = GenAIFeatureEngineer(proposer, gate=CostBenefitGate(min_gain=-1.0), cv=3)

    result = AutoML(feature_engineer=engineer, cv=3, n_trials=1, random_state=0).fit(ds)

    # the engineered column was used to train the winning model
    assert "rad_area" in result.best_model.feature_names
    # and the audit trail is threaded into the result
    engineering = result.extras["feature_engineering"]
    assert any(a.proposal.name == "rad_area" for a in engineering.accepted)


def test_automl_without_feature_engineer_has_no_engineering_extra() -> None:
    from sklearn.datasets import load_breast_cancer

    from fireflyframework_datascience.automl import AutoML
    from fireflyframework_datascience.core.types import TaskType
    from fireflyframework_datascience.datasets import Dataset

    raw = load_breast_cancer(as_frame=True)
    ds = Dataset(
        name="breast_cancer",
        X=raw.data,
        y=raw.target,
        task=TaskType.BINARY,
        target_name="target",
        feature_names=list(raw.data.columns),
    )
    result = AutoML(cv=3, n_trials=1, random_state=0).fit(ds)
    assert "feature_engineering" not in result.extras
    assert "rad_area" not in result.best_model.feature_names
