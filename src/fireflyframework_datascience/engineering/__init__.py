# Copyright 2026 Firefly Software Foundation.
"""Agentic ML-engineering loop — ports and result types (import-light).

The loop realizes the SOTA agentic pattern grounded on a deterministic executor: an LLM *proposes* a
solution (model + hyperparameters), the classical engine *trains and cross-validates* it, and a
**Verifier** — a stage distinct from execution-success — decides whether the result is actually good
(beats a trivial baseline / is finite). Search is greedy with reflection over the attempt history,
bounded by an iteration budget. Concrete engine in :mod:`fireflyframework_datascience.engineering.loop`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fireflyframework_datascience.datasets import Dataset


@dataclass(frozen=True)
class SolutionCandidate:
    """A proposed solution: a trainer name, its hyperparameters, and a rationale."""

    trainer: str
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass(frozen=True)
class Verdict:
    """The verifier's judgement of an evaluated candidate (correctness, not just 'it ran')."""

    valid: bool
    reason: str
    score: float


@dataclass(frozen=True)
class AttemptRecord:
    """One iteration of the loop: the candidate, its CV score, and the verdict."""

    candidate: SolutionCandidate
    score: float
    verdict: Verdict


@dataclass
class EngineeringRun:
    """The outcome of an agentic ML-engineering run."""

    best_candidate: SolutionCandidate | None
    best_score: float
    model: Any
    metric: str
    baseline_score: float
    attempts: list[AttemptRecord] = field(default_factory=list)

    @property
    def n_iterations(self) -> int:
        return len(self.attempts)

    @property
    def valid_attempts(self) -> list[AttemptRecord]:
        return [a for a in self.attempts if a.verdict.valid]

    def summary(self) -> str:
        return (
            f"Agentic AutoML: {self.n_iterations} attempts ({len(self.valid_attempts)} verified); "
            f"best={self.best_candidate.trainer if self.best_candidate else 'none'} "
            f"{self.metric}={self.best_score:.4f} (baseline {self.baseline_score:.4f})"
        )


@runtime_checkable
class CandidateProposer(Protocol):
    """Proposes initial candidates and reflects on history to propose the next one."""

    def propose_initial(self, dataset: Dataset, trainers: list[str]) -> list[SolutionCandidate]: ...

    def propose_next(
        self, dataset: Dataset, history: list[AttemptRecord], trainers: list[str]
    ) -> SolutionCandidate | None: ...


@runtime_checkable
class Verifier(Protocol):
    """Judges whether an evaluated candidate is genuinely good (distinct from execution-success)."""

    def verify(self, dataset: Dataset, candidate: SolutionCandidate, score: float, baseline: float) -> Verdict: ...


@runtime_checkable
class AgenticLoopPort(Protocol):
    """Runs an agentic ML-engineering search over a dataset and returns the best verified solution."""

    name: str

    def solve(self, dataset: Dataset) -> EngineeringRun: ...


class SequenceProposer:
    """A deterministic proposer that yields a fixed candidate sequence — for tests / fixed strategies."""

    def __init__(self, candidates: list[SolutionCandidate]) -> None:
        self._initial = list(candidates)
        self._queue: list[SolutionCandidate] = []

    def propose_initial(self, dataset: Dataset, trainers: list[str]) -> list[SolutionCandidate]:
        if not self._initial:
            return []
        first, *rest = self._initial
        self._queue = rest
        return [first]

    def propose_next(
        self, dataset: Dataset, history: list[AttemptRecord], trainers: list[str]
    ) -> SolutionCandidate | None:
        return self._queue.pop(0) if self._queue else None


__all__ = [
    "AgenticLoopPort",
    "AttemptRecord",
    "CandidateProposer",
    "EngineeringRun",
    "SequenceProposer",
    "SolutionCandidate",
    "Verdict",
    "Verifier",
]
