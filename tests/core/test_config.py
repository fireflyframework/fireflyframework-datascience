# Copyright 2026 Firefly Software Foundation.
"""Tests for configuration loading and precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from fireflyframework_datascience.core.config import FireflyDataScienceConfig


def test_defaults() -> None:
    config = FireflyDataScienceConfig.load(config_dir="/nonexistent-dir-xyz")
    assert config.default_ml_framework == "sklearn"
    assert config.genai.enabled is False
    assert config.execution.sandbox == "monty"
    assert config.execution.require_approval is True


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREFLY_DATASCIENCE_DEFAULT_ML_FRAMEWORK", "xgboost")
    monkeypatch.setenv("FIREFLY_DATASCIENCE_GENAI__ENABLED", "true")
    config = FireflyDataScienceConfig.load(config_dir="/nonexistent-dir-xyz")
    assert config.default_ml_framework == "xgboost"
    assert config.genai.enabled is True


def test_yaml_base(tmp_path: Path) -> None:
    (tmp_path / "firefly-datascience.yaml").write_text("default_ml_framework: lightgbm\ngenai:\n  enabled: true\n")
    config = FireflyDataScienceConfig.load(config_dir=tmp_path)
    assert config.default_ml_framework == "lightgbm"
    assert config.genai.enabled is True


def test_profile_overlay_beats_base(tmp_path: Path) -> None:
    (tmp_path / "firefly-datascience.yaml").write_text("default_ml_framework: lightgbm\n")
    (tmp_path / "firefly-datascience-prod.yaml").write_text("default_ml_framework: catboost\n")
    config = FireflyDataScienceConfig.load(config_dir=tmp_path, profiles=["prod"])
    assert config.default_ml_framework == "catboost"
    assert config.profiles == ["prod"]


def test_env_beats_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "firefly-datascience.yaml").write_text("default_ml_framework: lightgbm\n")
    monkeypatch.setenv("FIREFLY_DATASCIENCE_DEFAULT_ML_FRAMEWORK", "xgboost")
    config = FireflyDataScienceConfig.load(config_dir=tmp_path)
    assert config.default_ml_framework == "xgboost"
