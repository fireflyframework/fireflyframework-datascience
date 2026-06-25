# Copyright 2026 Firefly Software Foundation.
"""Pydantic models for the GenAI feature-proposal structured output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Feature(BaseModel):
    """One proposed feature."""

    name: str = Field(description="A short snake_case name for the feature")
    code: str = Field(description="pandas code that adds one numeric column to `df`")
    rationale: str = Field(default="", description="One line on why this feature should help")


class FeatureList(BaseModel):
    """The structured output of the feature-engineer agent."""

    features: list[Feature] = Field(default_factory=list)
