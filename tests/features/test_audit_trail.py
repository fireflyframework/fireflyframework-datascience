# Copyright 2026 Firefly Software Foundation.
"""The GenAI gate must persist an auditable trail of every decision.

The docs claim "every decision is logged and auditable" — until now that was only in-memory
(`EngineeringResult`). This verifies decisions are written durably (JSONL), one record per proposal.
Real data, LLM-free (StaticFeatureProposer).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def test_genai_engineer_writes_persistent_audit_trail(tmp_path) -> None:
    from fireflyframework_datascience.core.types import TaskType
    from fireflyframework_datascience.datasets import Dataset
    from fireflyframework_datascience.features import FeatureProposal, StaticFeatureProposer
    from fireflyframework_datascience.features.audit import JsonlAuditLog
    from fireflyframework_datascience.features.genai import GenAIFeatureEngineer

    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": b})
    y = pd.Series((a + b > 0).astype(int))
    ds = Dataset(name="t", X=X, y=y, task=TaskType.BINARY, target_name="y", feature_names=["a", "b"])

    proposer = StaticFeatureProposer(
        [
            FeatureProposal("sum_ab", "df['sum_ab'] = df['a'] + df['b']", "useful"),
            FeatureProposal("noise", "df['noise'] = 0.0", "useless constant"),
        ]
    )
    audit_path = tmp_path / "audit.jsonl"
    engineer = GenAIFeatureEngineer(proposer, audit_log=JsonlAuditLog(audit_path), cv=3)
    engineer.engineer(ds)

    records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
    # every proposal produces exactly one durable decision record
    assert len(records) == 2
    by_feature = {r["feature"]: r for r in records}
    assert set(by_feature) == {"sum_ab", "noise"}
    for record in records:
        assert record["decision"] in {"accepted", "rejected"}
        assert "score" in record and "baseline" in record and "code" in record
        assert record["dataset"] == "t"


def test_audit_log_is_optional() -> None:
    # No audit log wired -> engineer still works (in-memory trail only), nothing written.
    from fireflyframework_datascience.core.types import TaskType
    from fireflyframework_datascience.datasets import Dataset
    from fireflyframework_datascience.features import FeatureProposal, StaticFeatureProposer
    from fireflyframework_datascience.features.genai import GenAIFeatureEngineer

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    y = pd.Series([0, 0, 1, 1])
    ds = Dataset(name="t", X=X, y=y, task=TaskType.BINARY, target_name="y", feature_names=["a", "b"])
    proposer = StaticFeatureProposer([FeatureProposal("c", "df['c'] = df['a'] + df['b']", "")])

    result = GenAIFeatureEngineer(proposer, cv=2).engineer(ds)
    assert len(result.accepted) + len(result.rejected) == 1
