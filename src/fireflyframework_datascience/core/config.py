# Copyright 2026 Firefly Software Foundation.
"""Configuration model and loader (mirrors pyfly's config DNA).

:class:`FireflyDataScienceConfig` is a ``pydantic-settings`` model. Values resolve with the precedence
(highest first): constructor kwargs → environment variables (``FIREFLY_DATASCIENCE_*``, nested via
``__``) → ``.env`` → profile YAML overlays → base ``firefly-datascience.yaml`` → field defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from fireflyframework_datascience.core.banner import BannerMode

_CONFIG_FILE = "firefly-datascience.yaml"


class BannerConfig(BaseModel):
    """Startup banner settings."""

    mode: BannerMode = BannerMode.TEXT


class GenAIConfig(BaseModel):
    """GenAI accelerator settings. Off by default — classical-first."""

    enabled: bool = False
    default_model: str = "openai:gpt-4o"
    cost_benefit_gate: bool = True
    budget_usd: float | None = None


class ExecutionConfig(BaseModel):
    """Secure code-execution settings for LLM-generated code."""

    sandbox: Literal["monty", "docker", "e2b", "local"] = "monty"
    timeout_seconds: int = 60
    require_approval: bool = True


class FireflyDataScienceConfig(BaseSettings):
    """Root configuration for a Firefly DataScience application."""

    model_config = SettingsConfigDict(
        env_prefix="FIREFLY_DATASCIENCE_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    # Resolved at load() time; consumed by settings_customise_sources.
    _config_dir: ClassVar[Path] = Path(".")
    _active_profiles: ClassVar[list[str]] = []

    app_name: str = "firefly-datascience-app"
    profiles: list[str] = Field(default_factory=list)
    default_ml_framework: str = "sklearn"
    default_dataset_backend: str = "pandas"
    tracking_enabled: bool = False
    model_registry_url: str | None = None
    feature_store_endpoint: str | None = None
    banner: BannerConfig = Field(default_factory=BannerConfig)
    genai: GenAIConfig = Field(default_factory=GenAIConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        files: list[Path] = []
        base = cls._config_dir / _CONFIG_FILE
        if base.exists():
            files.append(base)
        for profile in cls._active_profiles:
            overlay = cls._config_dir / f"firefly-datascience-{profile}.yaml"
            if overlay.exists():
                files.append(overlay)
        yaml_sources = [YamlConfigSettingsSource(settings_cls, yaml_file=f) for f in files]
        # Earlier == higher priority. Profile overlays (later files) outrank the base file.
        return (init_settings, env_settings, dotenv_settings, *reversed(yaml_sources), file_secret_settings)

    @classmethod
    def load(
        cls,
        *,
        config_dir: str | Path | None = None,
        profiles: list[str] | None = None,
    ) -> FireflyDataScienceConfig:
        """Load configuration, merging env, ``.env``, and YAML (base + profile overlays)."""
        cls._config_dir = (
            Path(config_dir) if config_dir is not None else Path(os.getenv("FIREFLY_DATASCIENCE_CONFIG_DIR", "."))
        )
        if profiles is not None:
            cls._active_profiles = list(profiles)
        else:
            raw = os.getenv("FIREFLY_DATASCIENCE_PROFILES", "")
            cls._active_profiles = [p.strip() for p in raw.split(",") if p.strip()]
        config = cls()
        if cls._active_profiles and not config.profiles:
            config = config.model_copy(update={"profiles": cls._active_profiles})
        return config
