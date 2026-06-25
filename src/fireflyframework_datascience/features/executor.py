# Copyright 2026 Firefly Software Foundation.
"""Secure execution of LLM-generated feature-engineering code.

Reuses ``fireflyframework_agentic.execution`` static safety analysis (deny imports / dunder access /
dangerous builtins), then runs the vetted snippet in a restricted namespace exposing only ``df`` (a
copy of the feature frame), ``pd``, and ``np``.

SECURITY: executing model-generated code is an attack surface. Defence in depth here is: (1) static
analysis rejects imports, dunder access, and dangerous builtins before anything runs; (2) ``exec`` runs
with a minimal ``__builtins__`` allowlist and no I/O; (3) the caller may require HITL approval and/or
route to a container sandbox via ``config.execution.sandbox`` for untrusted data. This is the CAAFE
pattern: pandas/numpy transforms only, never arbitrary capability.
"""

from __future__ import annotations

from typing import Any

from fireflyframework_datascience.core.exceptions import FireflyDataScienceError

# A deliberately small builtins allowlist — enough for arithmetic/aggregation feature code, nothing else.
_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs",
        "min",
        "max",
        "sum",
        "round",
        "len",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sorted",
        "float",
        "int",
        "bool",
        "str",
        "list",
        "dict",
        "tuple",
        "set",
        "pow",
    )
}


class FeatureExecutionError(FireflyDataScienceError):
    """Raised when proposed feature code is unsafe or fails to execute."""


class FeatureCodeExecutor:
    """Vets and executes a feature-engineering snippet against a copy of the dataframe."""

    def __init__(self) -> None:
        from fireflyframework_agentic.execution import SafetyPolicy

        self._policy = SafetyPolicy(
            denied_modules=frozenset(
                {"os", "sys", "subprocess", "shutil", "socket", "pathlib", "importlib", "builtins"}
            ),
            denied_builtins=frozenset(
                {
                    "eval",
                    "exec",
                    "compile",
                    "open",
                    "__import__",
                    "input",
                    "globals",
                    "locals",
                    "vars",
                    "getattr",
                    "setattr",
                }
            ),
            deny_dunder_access=True,
        )

    def execute(self, code: str, X: Any) -> Any:
        """Run ``code`` (which mutates ``df``) and return the resulting dataframe.

        Raises :class:`FeatureExecutionError` if the code is unsafe, errors, or adds no new column.
        """
        from fireflyframework_agentic.execution import analyze_code

        report = analyze_code(code, self._policy)
        if not report.safe:
            reasons = "; ".join(v.message for v in report.violations)
            raise FeatureExecutionError(f"Unsafe feature code rejected: {reasons}")

        import numpy as np
        import pandas as pd

        before = set(X.columns)
        namespace: dict[str, Any] = {"df": X.copy(), "pd": pd, "np": np}
        try:
            exec(compile(code, "<feature>", "exec"), {"__builtins__": _SAFE_BUILTINS}, namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001 - surface any runtime error as a typed rejection
            raise FeatureExecutionError(f"Feature code failed at runtime: {exc}") from exc

        result = namespace["df"]
        if not isinstance(result, pd.DataFrame):
            raise FeatureExecutionError("Feature code must leave a pandas DataFrame in `df`")
        new_columns = set(result.columns) - before
        if not new_columns:
            raise FeatureExecutionError("Feature code added no new column")
        # Reject non-numeric / non-finite new columns (would break downstream estimators).
        for col in new_columns:
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise FeatureExecutionError(f"New feature {col!r} is not numeric")
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)
        return result


__all__ = ["FeatureCodeExecutor", "FeatureExecutionError"]
