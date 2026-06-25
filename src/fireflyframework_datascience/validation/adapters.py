# Copyright 2026 Firefly Software Foundation.
"""Data validation adapters: a pandas-based basic validator and an optional Pandera validator."""

from __future__ import annotations

from typing import Any

from fireflyframework_datascience.validation import ValidationReport


class BasicValidator:
    """Pandas-based sanity checks: empty data, all-null / constant columns, duplicates, target nulls."""

    name = "basic"

    def validate(self, X: Any, y: Any = None) -> ValidationReport:
        report = ValidationReport()
        n_rows, n_cols = int(X.shape[0]), int(X.shape[1])
        report.stats = {"n_rows": n_rows, "n_cols": n_cols}

        if n_rows == 0:
            report.ok = False
            report.errors.append("Feature matrix has zero rows")
            return report

        null_frac = X.isna().mean()
        all_null = [c for c in X.columns if null_frac[c] >= 1.0]
        mostly_null = [c for c in X.columns if 0.5 <= null_frac[c] < 1.0]
        constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if all_null:
            report.ok = False
            report.errors.append(f"All-null columns: {all_null}")
        if mostly_null:
            report.warnings.append(f"Columns >=50% null: {mostly_null}")
        if constant:
            report.warnings.append(f"Constant columns (no signal): {constant}")

        dup = int(X.duplicated().sum())
        if dup:
            report.warnings.append(f"{dup} duplicate rows")
        report.stats["duplicate_rows"] = dup

        if y is not None:
            import pandas as pd

            y_null = int(pd.Series(y).isna().sum())
            if y_null:
                report.ok = False
                report.errors.append(f"Target has {y_null} null values")
        return report


class PanderaValidator:
    """Validates against an inferred or supplied Pandera schema. Requires the ``validation`` extra."""

    name = "pandera"

    def __init__(self, schema: Any = None) -> None:
        self._schema = schema

    def validate(self, X: Any, y: Any = None) -> ValidationReport:
        try:
            import pandera.pandas as pa
        except ImportError as exc:  # pragma: no cover
            from fireflyframework_datascience.core.exceptions import AdapterUnavailableError

            raise AdapterUnavailableError("PanderaValidator", "validation") from exc

        report = ValidationReport(stats={"n_rows": int(X.shape[0]), "n_cols": int(X.shape[1])})
        schema = self._schema or pa.infer_schema(X)
        try:
            schema.validate(X, lazy=True)
        except pa.errors.SchemaErrors as exc:
            report.ok = False
            report.errors.append(str(exc.failure_cases.shape[0]) + " schema failure cases")
        return report


__all__ = ["BasicValidator", "PanderaValidator"]
