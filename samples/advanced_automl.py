# Copyright 2026 Firefly Software Foundation.
"""Advanced AutoML — production-grade selection, trust, and governance, on real data.

One runnable script that exercises every modeling/trust feature added in the latest release, on the
real ``breast_cancer`` dataset (no synthetic data):

  1. **Governed feature engineering with a persisted audit trail.** The LLM (or, offline, a
     deterministic stand-in) proposes features; the cost/benefit gate keeps only measured wins; and
     *every* decision — accepted or rejected — is appended to a JSONL audit log.
  2. **Robust model selection.** A scikit-learn ``StratifiedKFold`` splitter drives cross-validation,
     and the winner is chosen on **PR-AUC** (``average_precision``) — the right target for imbalanced
     or cost-sensitive binary problems.
  3. **A calibrated stacking ensemble.** The top-k candidates are stacked, then calibrated so the
     predicted probabilities are trustworthy (reported via the Brier score).
  4. **Explainability.** The winner reports deterministic global feature importances on the holdout.

The GenAI step uses a real LLM when a key is present (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` /
``GEMINI_API_KEY``), and a deterministic ``StaticFeatureProposer`` otherwise — so the same sample runs
offline in CI and against a live model when credentials are available.

Run it:  ``python samples/advanced_automl.py``   (needs the ``tabular`` extra; ``explain`` for SHAP)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from fireflyframework_datascience.automl import AutoML
from fireflyframework_datascience.datasets.adapters import SklearnDatasetLoader
from fireflyframework_datascience.features import FeatureProposal, StaticFeatureProposer
from fireflyframework_datascience.features.audit import JsonlAuditLog
from fireflyframework_datascience.features.genai import GenAIFeatureEngineer

DEFAULT_MODEL = os.getenv("FIREFLY_DATASCIENCE_GENAI__DEFAULT_MODEL", "anthropic:claude-haiku-4-5")


def _has_llm_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"))


def _make_proposer() -> tuple[Any, str]:
    """A real LLM proposer when a key is present; otherwise a deterministic, offline stand-in."""
    if _has_llm_key():
        from fireflyframework_datascience.features.genai import AgentFeatureProposer

        return AgentFeatureProposer(model=DEFAULT_MODEL), "llm"
    # Offline: real feature code over real columns. The gate decides what (if anything) earns its keep.
    proposer = StaticFeatureProposer(
        [
            FeatureProposal(
                "area_to_perimeter",
                "df['area_to_perimeter'] = df['worst area'] / (df['worst perimeter'] + 1)",
                "compactness proxy",
            ),
            FeatureProposal(
                "concavity_interaction",
                "df['concavity_interaction'] = df['mean concavity'] * df['mean concave points']",
                "interaction term",
            ),
            FeatureProposal("noise", "df['noise'] = 0.0", "should be rejected — adds nothing"),
        ]
    )
    return proposer, "static"


def _apply(engineered: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the *accepted* feature code to a frame, keeping train and test consistent."""
    from fireflyframework_datascience.features.executor import FeatureCodeExecutor

    executor = FeatureCodeExecutor()
    working = X.copy()
    for accepted in engineered.accepted:
        working = executor.execute(accepted.proposal.code, working)
    return working


def run(audit_path: str | Path | None = None) -> dict[str, Any]:
    """Run the advanced AutoML pipeline end-to-end and return a report dict."""
    from sklearn.model_selection import StratifiedKFold

    ds = SklearnDatasetLoader().load("breast_cancer")  # real data
    train, test = ds.train_test_split(test_size=0.25, random_state=0)

    # 1. Governed GenAI feature engineering with a persisted, append-only audit trail.
    cleanup = False
    if audit_path is None:
        audit_path = Path(tempfile.mkdtemp(prefix="firefly-audit-")) / "genai-decisions.jsonl"
        cleanup = True
    proposer, proposer_kind = _make_proposer()
    audit = JsonlAuditLog(audit_path)
    engineer = GenAIFeatureEngineer(proposer, audit_log=audit, cv=4)
    engineered = engineer.engineer(train)

    # 2. Robust selection: a StratifiedKFold splitter + PR-AUC, with a calibrated stacking ensemble.
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    automl = AutoML(cv=splitter, n_trials=1, calibrate=True, ensemble=True, ensemble_size=3, random_state=0)
    result = automl.fit(engineered.dataset, metric="average_precision")

    test_engineered = test.with_features(_apply(engineered, test.X))
    evaluation = result.evaluate(test_engineered)

    # 3. Explainability — deterministic global feature importances on the holdout.
    explanation = result.explain(test_engineered)

    # 4. Read the persisted audit trail back (one JSON line per gate decision).
    audit_records = [json.loads(line) for line in Path(audit_path).read_text(encoding="utf-8").splitlines()]
    if cleanup:
        Path(audit_path).unlink(missing_ok=True)

    return {
        "proposer": proposer_kind,
        "accepted_features": [a.proposal.name for a in engineered.accepted],
        "rejected_features": [r.proposal.name for r in engineered.rejected],
        "fe_lift": engineered.lift,
        "winner": result.best_model.name,
        "selection_metric": result.metric,
        "cv_scoring": result.cv_scoring,
        "leaderboard": result.leaderboard_table(),
        "holdout": evaluation.metrics,
        "explanation_method": explanation.method,
        "top_features": explanation.top(8),
        "audit_trail": audit_records,
    }


def main() -> None:
    report = run()
    print("=== Advanced AutoML — calibrated stacking ensemble, PR-AUC selection, explainability ===")
    print(f"proposer          : {report['proposer']}")
    print(f"accepted features : {report['accepted_features']}")
    print(f"rejected features : {report['rejected_features']}")
    print(f"winning model     : {report['winner']}")
    print(f"selected on       : {report['selection_metric']} (cv scorer: {report['cv_scoring']})")
    print("leaderboard:")
    print(report["leaderboard"])
    print(f"holdout metrics   : {report['holdout']}")
    print(f"explanation       : {report['explanation_method']}")
    for name, importance in report["top_features"]:
        print(f"  {name:<26} {importance:+.4f}")
    print(f"audit trail       : {len(report['audit_trail'])} decisions persisted")
    for record in report["audit_trail"]:
        print(f"  {record['decision']:<9} {record['feature']:<24} ({record['detail']})")


if __name__ == "__main__":
    main()
