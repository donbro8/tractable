"""Unit tests for the tractable register command."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from tractable.cli.main import app


runner = CliRunner()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_config(tmp_path: Path) -> Path:
    """Write a minimal valid RepositoryRegistration config to a temp file."""
    config = tmp_path / "config.py"
    config.write_text(
        textwrap.dedent("""\
            from tractable.types.config import GitProviderConfig, RepositoryRegistration

            registration = RepositoryRegistration(
                name="acme/my-api",
                git_url="https://github.com/acme/my-api.git",
                git_provider=GitProviderConfig(
                    provider_type="github",
                    credentials_secret_ref="GITHUB_TOKEN",
                ),
                primary_language="python",
            )
        """),
        encoding="utf-8",
    )
    return config


def _invalid_config(tmp_path: Path) -> Path:
    """Write a config that triggers a Pydantic validation error."""
    config = tmp_path / "bad_config.py"
    config.write_text(
        textwrap.dedent("""\
            from tractable.types.config import GitProviderConfig, RepositoryRegistration

            registration = RepositoryRegistration(
                name=123,
                git_url="https://github.com/acme/my-api.git",
                git_provider=GitProviderConfig(
                    provider_type="invalid_provider",
                    credentials_secret_ref="TOKEN",
                ),
                primary_language="python",
            )
        """),
        encoding="utf-8",
    )
    return config


def _no_registration_config(tmp_path: Path) -> Path:
    """Write a config file that has no RepositoryRegistration instance."""
    config = tmp_path / "empty_config.py"
    config.write_text("x = 42\n", encoding="utf-8")
    return config


# ── Tests ─────────────────────────────────────────────────────────────────────

# AC-1: --help lists register and status
def test_help_lists_register_and_status() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "register" in result.output
    assert "status" in result.output


def test_register_help_exits_zero() -> None:
    result = runner.invoke(app, ["register", "--help"])
    assert result.exit_code == 0


# AC-3: nonexistent path → exit code 1 + error on stderr
def test_register_nonexistent_path_exits_1() -> None:
    result = runner.invoke(app, ["register", "/nonexistent/path.py"])
    assert result.exit_code == 1
    assert "not found" in (result.stderr or result.output).lower() or "error" in (result.stderr or result.output).lower()


# AC-4: Pydantic validation error → exit code 1
def test_register_pydantic_validation_error(tmp_path: Path) -> None:
    config = _invalid_config(tmp_path)
    result = runner.invoke(app, ["register", str(config)])
    assert result.exit_code == 1


# No RepositoryRegistration instance → exit code 1
def test_register_no_registration_in_file(tmp_path: Path) -> None:
    config = _no_registration_config(tmp_path)
    result = runner.invoke(app, ["register", str(config)])
    assert result.exit_code == 1


# Successful registration (mocked ingest)
def test_register_success_mocked(tmp_path: Path) -> None:
    config = _valid_config(tmp_path)

    fake_result = {
        "files_parsed": 10,
        "entities_created": 42,
        "relationships_created": 18,
        "duration_seconds": 2.5,
        "errors": [],
    }

    with patch(
        "tractable.cli.commands.register._run_ingest",
        new=AsyncMock(return_value=fake_result),
    ):
        result = runner.invoke(app, ["register", str(config)])

    assert result.exit_code == 0
    assert "Registration complete" in result.output
    assert "10" in result.output  # files_parsed


# Registration with ingest errors should still succeed (non-fatal warnings)
def test_register_with_parse_warnings_mocked(tmp_path: Path) -> None:
    config = _valid_config(tmp_path)

    fake_result = {
        "files_parsed": 5,
        "entities_created": 20,
        "relationships_created": 8,
        "duration_seconds": 1.0,
        "errors": ["src/broken.py: SyntaxError"],
    }

    with patch(
        "tractable.cli.commands.register._run_ingest",
        new=AsyncMock(return_value=fake_result),
    ):
        result = runner.invoke(app, ["register", str(config)])

    assert result.exit_code == 0
    assert "Warning" in result.output or "warning" in result.output.lower()


# Ingest failure → exit code 1
def test_register_ingest_failure_exits_1(tmp_path: Path) -> None:
    config = _valid_config(tmp_path)

    with patch(
        "tractable.cli.commands.register._run_ingest",
        new=AsyncMock(side_effect=RuntimeError("FalkorDB connection refused")),
    ):
        result = runner.invoke(app, ["register", str(config)])

    assert result.exit_code == 1


# Registration summary is printed
def test_register_prints_summary(tmp_path: Path) -> None:
    config = _valid_config(tmp_path)

    fake_result = {
        "files_parsed": 3,
        "entities_created": 7,
        "relationships_created": 4,
        "duration_seconds": 0.5,
        "errors": [],
    }

    with patch(
        "tractable.cli.commands.register._run_ingest",
        new=AsyncMock(return_value=fake_result),
    ):
        result = runner.invoke(app, ["register", str(config)])

    assert result.exit_code == 0
    assert "acme/my-api" in result.output


# Path to a directory → exit code 1
def test_register_path_is_directory(tmp_path: Path) -> None:
    result = runner.invoke(app, ["register", str(tmp_path)])
    assert result.exit_code == 1
