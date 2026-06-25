# Copyright 2026 Firefly Software Foundation.
"""Version sanity tests (CalVer YY.MM.PATCH)."""

from __future__ import annotations

import re

import fireflyframework_datascience as ds


def test_version_is_calver() -> None:
    assert re.fullmatch(r"\d{2}\.\d{1,2}\.\d+", ds.__version__), ds.__version__


def test_pyproject_version_matches() -> None:
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    assert data["project"]["version"] == ds.__version__
