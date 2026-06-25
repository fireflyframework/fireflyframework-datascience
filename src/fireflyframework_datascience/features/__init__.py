# Copyright 2026 Firefly Software Foundation.
"""Feature engineering module — ports, the cost/benefit gate, and result types (import-light).

This module embodies the framework's core thesis: an LLM *proposes* feature code, a deterministic
classical engine *measures the cross-validation lift*, and a :class:`CostBenefitGate` accepts the
feature only if it beats the current baseline. The LLM never decides; the measured score does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fireflyframework_datascience.datasets import Dataset


@dataclass(frozen=True)
class FeatureProposal:
    """A single proposed feature: a name, the Python code that creates it, and a rationale."""

    name: str
    code: str
    rationale: str = ""


@dataclass(frozen=True)
class AcceptedFeature:
    """A proposal that improved the cross-validation score and was kept."""

    proposal: FeatureProposal
    score: float
    gain: float


@dataclass(frozen=True)
class RejectedFeature:
    """A proposal that was dropped (unsafe code, execution error, or no measured lift)."""

    proposal: FeatureProposal
    reason: str
    score: float = float("nan")


@dataclass
class EngineeringResult:
    """The engineered dataset plus the audit trail of accepted/rejected features."""

    dataset: Dataset
    accepted: list[AcceptedFeature] = field(default_factory=list)
    rejected: list[RejectedFeature] = field(default_factory=list)
    baseline_score: float = 0.0
    final_score: float = 0.0
    metric: str = ""

    @property
    def lift(self) -> float:
        return self.final_score - self.baseline_score

    def summary(self) -> str:
        return (
            f"GenAI feature engineering: {len(self.accepted)} accepted, {len(self.rejected)} rejected; "
            f"{self.metric} {self.baseline_score:.4f} -> {self.final_score:.4f} (lift {self.lift:+.4f})"
        )


class CostBenefitGate:
    """Accepts a GenAI contribution only if it improves the score by at least ``min_gain``.

    This is the governance primitive that keeps GenAI honest: a proposal that does not measurably beat
    the seeded classical baseline is rejected, and (optionally) the whole GenAI step can be disabled if
    nothing it produces clears the bar.
    """

    def __init__(self, min_gain: float = 0.0) -> None:
        self.min_gain = min_gain

    def accepts(self, current_score: float, candidate_score: float) -> bool:
        return (candidate_score - current_score) > self.min_gain


@runtime_checkable
class FeatureProposer(Protocol):
    """Proposes feature-engineering code for a dataset (LLM-backed or deterministic)."""

    def propose(self, dataset: Dataset, *, max_features: int = 5) -> list[FeatureProposal]: ...


class StaticFeatureProposer:
    """A deterministic proposer returning a fixed list — for known features or LLM-free pipelines."""

    def __init__(self, proposals: list[FeatureProposal]) -> None:
        self._proposals = list(proposals)

    def propose(self, dataset: Dataset, *, max_features: int = 5) -> list[FeatureProposal]:
        return self._proposals[:max_features]


@runtime_checkable
class FeatureEngineerPort(Protocol):
    """Engineers new features for a dataset, returning an engineered copy and an audit trail."""

    name: str

    def engineer(self, dataset: Dataset, *, max_features: int = 5) -> EngineeringResult: ...


__all__ = [
    "AcceptedFeature",
    "CostBenefitGate",
    "EngineeringResult",
    "FeatureEngineerPort",
    "FeatureProposal",
    "FeatureProposer",
    "RejectedFeature",
    "StaticFeatureProposer",
]
