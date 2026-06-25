# Copyright 2026 Firefly Software Foundation.
"""Tests for the ``firefly-ds`` CLI."""

from __future__ import annotations

from click.testing import CliRunner

from fireflyframework_datascience import __version__
from fireflyframework_datascience.cli.main import cli


def test_version_command() -> None:
    result = CliRunner().invoke(cli, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_doctor_command() -> None:
    result = CliRunner().invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "doctor" in result.output
    assert "agentic" in result.output


def test_introspect_command() -> None:
    result = CliRunner().invoke(cli, ["introspect"])
    assert result.exit_code == 0
    assert "CoreAutoConfiguration" in result.output


def test_help() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Firefly DataScience" in result.output
