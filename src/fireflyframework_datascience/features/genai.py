# Copyright 2026 Firefly Software Foundation.
"""GenAI feature engineering: LLM proposes feature code, classical CV measures the lift, the gate decides.

``AgentFeatureProposer`` wraps a ``fireflyframework_agentic.FireflyAgent`` to propose features;
``GenAIFeatureEngineer`` executes each proposal safely, measures cross-validation lift, and keeps only
those the :class:`CostBenefitGate` accepts. The proposer is injectable, so the loop is fully testable
without an LLM.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fireflyframework_datascience.core.types import TaskType
from fireflyframework_datascience.datasets import Dataset
from fireflyframework_datascience.evaluation import MetricsEvaluatorPort
from fireflyframework_datascience.features import (
    AcceptedFeature,
    CostBenefitGate,
    EngineeringResult,
    FeatureProposal,
    FeatureProposer,
    RejectedFeature,
)
from fireflyframework_datascience.features.executor import FeatureCodeExecutor, FeatureExecutionError

_CLASSIFICATION = {TaskType.BINARY, TaskType.MULTICLASS, TaskType.CLASSIFICATION}

_INSTRUCTIONS = (
    "You are an expert tabular feature engineer. Given a dataset schema, propose new features as short "
    "pandas snippets that add columns to a DataFrame named `df`. Rules: use only `df`, `pd`, and `np`; "
    "no imports; no I/O; each snippet must add exactly one new numeric column via assignment, e.g. "
    "df['ratio'] = df['a'] / (df['b'] + 1). Return a name, the code, and a one-line rationale per feature."
)


class GenAIFeatureEngineer:
    """Runs the propose → execute → measure → gate loop. Implements ``FeatureEngineerPort``."""

    name = "genai"

    def __init__(
        self,
        proposer: FeatureProposer,
        *,
        evaluator: MetricsEvaluatorPort | None = None,
        executor: FeatureCodeExecutor | None = None,
        gate: CostBenefitGate | None = None,
        scorer_estimator: Callable[[TaskType], Any] | None = None,
        cv: int = 5,
        max_features: int = 5,
        random_state: int = 42,
    ) -> None:
        self._proposer = proposer
        self._evaluator = evaluator or _default_evaluator()
        self._executor = executor or FeatureCodeExecutor()
        self._gate = gate or CostBenefitGate()
        self._scorer_estimator_factory = scorer_estimator
        self._cv = cv
        self._max_features = max_features
        self._random_state = random_state

    def engineer(self, dataset: Dataset, *, max_features: int | None = None) -> EngineeringResult:
        task = dataset.task
        metric = self._evaluator.default_metric(task)
        scoring = self._evaluator.scoring_name(task, metric)
        baseline = self._cv_score(dataset.X, dataset.y, task, scoring)

        proposals = self._proposer.propose(dataset, max_features=max_features or self._max_features)
        working = dataset.X.copy()
        current = baseline
        accepted: list[AcceptedFeature] = []
        rejected: list[RejectedFeature] = []

        for proposal in proposals:
            try:
                candidate = self._executor.execute(proposal.code, working)
            except FeatureExecutionError as exc:
                rejected.append(RejectedFeature(proposal, str(exc)))
                continue
            candidate_score = self._cv_score(candidate, dataset.y, task, scoring)
            if self._gate.accepts(current, candidate_score):
                working = candidate
                accepted.append(AcceptedFeature(proposal, candidate_score, candidate_score - current))
                current = candidate_score
            else:
                rejected.append(
                    RejectedFeature(proposal, f"no lift ({candidate_score:.4f} <= {current:.4f})", candidate_score)
                )

        return EngineeringResult(
            dataset=dataset.with_features(working),
            accepted=accepted,
            rejected=rejected,
            baseline_score=baseline,
            final_score=current,
            metric=metric,
        )

    def _cv_score(self, X: Any, y: Any, task: TaskType, scoring: str) -> float:
        from sklearn.model_selection import cross_val_score

        estimator = self._pipeline(self._scorer_estimator(task), X)
        scores = cross_val_score(estimator, X, y, cv=self._cv, scoring=scoring)
        return float(scores.mean())

    def _scorer_estimator(self, task: TaskType) -> Any:
        if self._scorer_estimator_factory is not None:
            return self._scorer_estimator_factory(task)
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        cls = HistGradientBoostingClassifier if task in _CLASSIFICATION else HistGradientBoostingRegressor
        return cls(random_state=self._random_state)

    def _pipeline(self, estimator: Any, X: Any) -> Any:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        numeric = X.select_dtypes(include="number").columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]
        transformers: list[tuple[str, Any, list[str]]] = []
        if numeric:
            transformers.append(("num", SimpleImputer(strategy="median"), numeric))
        if categorical:
            cat_pipe = Pipeline(
                [("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]
            )
            transformers.append(("cat", cat_pipe, categorical))
        if not transformers:
            return Pipeline([("model", estimator)])
        return Pipeline([("prep", ColumnTransformer(transformers, remainder="drop")), ("model", estimator)])


class AgentFeatureProposer:
    """Proposes features using a ``FireflyAgent`` (reused from ``fireflyframework-agentic``)."""

    def __init__(self, *, model: Any = None, agent: Any = None, sample_rows: int = 5) -> None:
        self._sample_rows = sample_rows
        self._agent = agent if agent is not None else self._build_agent(model)

    def _build_agent(self, model: Any) -> Any:
        from fireflyframework_agentic.agents import FireflyAgent

        from fireflyframework_datascience.features._schema import FeatureList

        return FireflyAgent(
            name="feature_engineer",
            model=model or "openai:gpt-4o",
            instructions=_INSTRUCTIONS,
            output_type=FeatureList,
            default_middleware=False,
            auto_register=False,
        )

    def propose(self, dataset: Dataset, *, max_features: int = 5) -> list[FeatureProposal]:
        prompt = self._describe(dataset, max_features)
        result = self._agent.run_sync(prompt)
        features = getattr(result.output, "features", [])
        return [FeatureProposal(name=f.name, code=f.code, rationale=f.rationale) for f in features[:max_features]]

    def _describe(self, dataset: Dataset, max_features: int) -> str:
        head = dataset.X.head(self._sample_rows).to_dict(orient="records")
        dtypes = {c: str(t) for c, t in dataset.X.dtypes.items()}
        return (
            f"Task: {dataset.task.value}. Target: {dataset.target_name}. "
            f"Propose up to {max_features} new features.\n"
            f"Columns and dtypes: {dtypes}\n"
            f"Sample rows: {head}"
        )


def _default_evaluator() -> MetricsEvaluatorPort:
    from fireflyframework_datascience.evaluation.adapters import SklearnMetricsEvaluator

    return SklearnMetricsEvaluator()


__all__ = ["AgentFeatureProposer", "GenAIFeatureEngineer"]
