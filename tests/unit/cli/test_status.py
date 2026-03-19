"""Unit tests for the tractable status command."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from tractable.cli.main import app

runner = CliRunner()


# ── Tests ─────────────────────────────────────────────────────────────────────

# AC-5: tractable status exits 0 when no agents are registered
def test_status_no_agents_message() -> None:
    with patch(
        "tractable.cli.commands.status._fetch_contexts",
        new=AsyncMock(return_value=[]),
    ):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "No agents registered" in result.output


# AC-5: tractable status exits 0 and prints a table when agents exist
def test_status_with_agents_prints_table() -> None:
    fake_contexts = [
        {
            "agent_id": "agent-001",
            "base_template": "api_maintainer",
            "last_active": "2026-03-19 10:00:00+00:00",
            "last_known_head_sha": "abc123def456",
        }
    ]
    with patch(
        "tractable.cli.commands.status._fetch_contexts",
        new=AsyncMock(return_value=fake_contexts),
    ):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "agent-001" in result.output


# Missing DATABASE_URL → exit code 1
def test_status_missing_database_url_exits_1() -> None:
    with patch(
        "tractable.cli.commands.status._fetch_contexts",
        new=AsyncMock(side_effect=RuntimeError("DATABASE_URL environment variable is not set.")),
    ):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 1


# Database connection error → exit code 1
def test_status_db_connection_error_exits_1() -> None:
    with patch(
        "tractable.cli.commands.status._fetch_contexts",
        new=AsyncMock(side_effect=OSError("connection refused")),
    ):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 1


# Multiple agents are listed
def test_status_multiple_agents() -> None:
    fake_contexts = [
        {
            "agent_id": "agent-001",
            "base_template": "api_maintainer",
            "last_active": "2026-03-18 09:00:00+00:00",
            "last_known_head_sha": "aaa111",
        },
        {
            "agent_id": "agent-002",
            "base_template": "frontend_maintainer",
            "last_active": "2026-03-19 12:00:00+00:00",
            "last_known_head_sha": "bbb222",
        },
    ]
    with patch(
        "tractable.cli.commands.status._fetch_contexts",
        new=AsyncMock(return_value=fake_contexts),
    ):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "agent-001" in result.output
    assert "agent-002" in result.output


# status --help exits 0
def test_status_help_exits_zero() -> None:
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0
