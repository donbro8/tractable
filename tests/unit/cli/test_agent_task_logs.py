"""Unit tests for tractable agent, task, and logs CLI commands.

Acceptance criteria covered:
    AC1 — agent list shows table with agent_id and repo
    AC2 — agent context prints non-empty string ending with pinned instructions
    AC3 — agent edit with EDITOR=cat saves pinned instructions unchanged
    AC4 — task submit prints UUID task_id and exits 0
    AC5 — task submit --repo nonexistent exits 1 with "not found" message
    AC6 — logs --agent <id> prints one JSON line per audit entry
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from tractable.cli.main import app
from tractable.errors import FatalError
from tractable.types.agent import AgentContext, AuditEntry

runner = CliRunner()

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_context(
    agent_id: str = "agent-001",
    repo: str = "my-api",
    system_prompt: str = "You are a coding agent.",
    pinned_instructions: list[str] | None = None,
) -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        repo=repo,
        base_template="api_maintainer",
        system_prompt=system_prompt,
        repo_architectural_summary="",
        pinned_instructions=pinned_instructions or [],
    )


def _make_audit_entry(
    agent_id: str = "agent-001",
    action: str = "file_written",
) -> AuditEntry:
    return AuditEntry(
        timestamp=datetime(2026, 3, 20, 12, 0, 0, tzinfo=UTC),
        agent_id=agent_id,
        task_id="task-001",
        action=action,
        detail={"path": "src/foo.py"},
        outcome="success",
    )


# ── AC1: agent list ───────────────────────────────────────────────────────────


def test_agent_list_two_agents_shows_table() -> None:
    """AC1: two agents in store → table with two rows, correct agent_id and repo."""
    fake_rows = [
        {
            "agent_id": "agent-001",
            "repo": "my-api",
            "status": "dormant",
            "last_active": "2026-03-19T10:00:00+00:00",
            "current_task_id": "—",
        },
        {
            "agent_id": "agent-002",
            "repo": "frontend-app",
            "status": "idle",
            "last_active": "2026-03-20T08:00:00+00:00",
            "current_task_id": "—",
        },
    ]
    with patch(
        "tractable.cli.commands.agent._list_agents_data",
        new=AsyncMock(return_value=fake_rows),
    ):
        result = runner.invoke(app, ["agent", "list"])

    assert result.exit_code == 0, result.output
    assert "agent-001" in result.output
    assert "agent-002" in result.output
    assert "my-api" in result.output
    assert "frontend-app" in result.output


def test_agent_list_no_agents_prints_message() -> None:
    """agent list with empty store prints a helpful message."""
    with patch(
        "tractable.cli.commands.agent._list_agents_data",
        new=AsyncMock(return_value=[]),
    ):
        result = runner.invoke(app, ["agent", "list"])

    assert result.exit_code == 0
    assert "No agents registered" in result.output


def test_agent_list_db_error_exits_1() -> None:
    """agent list with DB error exits 1."""
    with patch(
        "tractable.cli.commands.agent._list_agents_data",
        new=AsyncMock(side_effect=FatalError("DATABASE_URL not set.")),
    ):
        result = runner.invoke(app, ["agent", "list"])

    assert result.exit_code == 1
    assert isinstance(result.exception, FatalError)


# ── AC2: agent context ────────────────────────────────────────────────────────


def test_agent_context_prints_system_prompt_and_pinned() -> None:
    """AC2: agent context prints non-empty string ending with pinned instructions."""
    expected_text = "You are a coding agent.\n[pinned] always write tests"
    with patch(
        "tractable.cli.commands.agent._get_context_text",
        new=AsyncMock(return_value=expected_text),
    ):
        result = runner.invoke(app, ["agent", "context", "agent-001"])

    assert result.exit_code == 0, result.output
    assert "You are a coding agent." in result.output
    assert "[pinned] always write tests" in result.output
    # Ends with pinned instruction
    assert result.output.strip().endswith("[pinned] always write tests")


def test_agent_context_not_found_exits_1() -> None:
    """agent context for unknown agent exits 1 with 'not found' message."""
    with patch(
        "tractable.cli.commands.agent._get_context_text",
        new=AsyncMock(side_effect=FatalError("Agent unknown-agent not found.")),
    ):
        result = runner.invoke(app, ["agent", "context", "unknown-agent"])

    assert result.exit_code == 1
    assert isinstance(result.exception, FatalError)
    assert "not found" in str(result.exception).lower()


# ── AC3: agent edit ───────────────────────────────────────────────────────────


def test_agent_edit_cat_editor_saves_unchanged() -> None:
    """AC3: EDITOR=cat (no-op) saves pinned instructions unchanged."""
    original_pinned = ["run all tests before commit", "follow PEP 8"]

    saved_pinned: list[list[str]] = []

    async def fake_edit_pinned(*_args: object) -> None:
        # Simulate: editor reads temp file (unchanged), save is called
        saved_pinned.append(original_pinned)

    with (
        patch("tractable.cli.commands.agent._edit_pinned", new=fake_edit_pinned),
        patch.dict("os.environ", {"EDITOR": "cat"}),
    ):
        result = runner.invoke(app, ["agent", "edit", "agent-001"])

    assert result.exit_code == 0, result.output
    assert saved_pinned == [original_pinned]


def test_agent_edit_saves_unchanged_via_store_mock() -> None:
    """AC3 (store-level): EDITOR=cat writes temp file, reads it back, save_agent_context called."""
    original_pinned = ["always write tests"]
    captured_save: list[tuple[str, list[str]]] = []

    mock_store = MagicMock()
    mock_store.get_agent_context = AsyncMock(
        return_value=_make_context(pinned_instructions=original_pinned)
    )

    async def _fake_save(agent_id: str, ctx: AgentContext) -> None:
        captured_save.append((agent_id, list(ctx.pinned_instructions)))

    mock_store.save_agent_context = AsyncMock(side_effect=_fake_save)

    with (
        patch(
            "tractable.state.store.PostgreSQLAgentStateStore.from_env",
            return_value=mock_store,
        ),
        patch("tractable.cli.commands.agent.subprocess.run"),
        patch.dict("os.environ", {"EDITOR": "cat"}),
    ):
        result = runner.invoke(app, ["agent", "edit", "agent-001"])

    assert result.exit_code == 0, result.output
    assert len(captured_save) == 1
    saved_agent_id, saved_pinned = captured_save[0]
    assert saved_agent_id == "agent-001"
    assert saved_pinned == original_pinned


# ── AC4: task submit success ──────────────────────────────────────────────────


def test_task_submit_prints_uuid_and_exits_0() -> None:
    """AC4: task submit prints UUID task_id and exits 0."""
    fake_task_id = str(uuid.uuid4())
    with patch(
        "tractable.cli.commands.task._submit_task_async",
        new=AsyncMock(return_value=fake_task_id),
    ):
        result = runner.invoke(app, ["task", "submit", "Fix null pointer", "--repo", "my-api"])

    assert result.exit_code == 0, result.output
    # task_id is a UUID — validate format
    output = result.output.strip()
    parsed = uuid.UUID(output)
    assert str(parsed) == output


# ── AC5: task submit repo not found ──────────────────────────────────────────


def test_task_submit_repo_not_found_exits_1() -> None:
    """AC5: task submit for unregistered repo exits 1 with 'not found' message."""
    with patch(
        "tractable.cli.commands.task._submit_task_async",
        new=AsyncMock(side_effect=FatalError("Repo nonexistent not found.")),
    ):
        result = runner.invoke(app, ["task", "submit", "Fix null pointer", "--repo", "nonexistent"])

    assert result.exit_code == 1
    assert isinstance(result.exception, FatalError)
    assert "nonexistent" in str(result.exception)
    assert "not found" in str(result.exception).lower()


# ── AC6: logs --agent ─────────────────────────────────────────────────────────


def test_logs_agent_filter_prints_json_lines() -> None:
    """AC6: logs --agent <id> prints at least one JSON line per audit entry."""
    entries = [
        _make_audit_entry("agent-001", "file_written"),
        _make_audit_entry("agent-001", "graph_query"),
    ]
    with patch(
        "tractable.cli.commands.logs._fetch_log",
        new=AsyncMock(return_value=entries),
    ):
        result = runner.invoke(app, ["logs", "--agent", "agent-001"])

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.strip().splitlines() if ln.strip()]
    assert len(lines) >= len(entries)
    # Each line must be valid JSON
    for line in lines:
        data = json.loads(line)
        assert data["agent_id"] == "agent-001"


def test_logs_no_filter_uses_default_limit() -> None:
    """logs without filters fetches last 50 entries."""
    entries = [_make_audit_entry(f"agent-{i:03d}") for i in range(3)]
    with patch(
        "tractable.cli.commands.logs._fetch_log",
        new=AsyncMock(return_value=entries),
    ) as mock_fetch:
        result = runner.invoke(app, ["logs"])

    assert result.exit_code == 0, result.output
    mock_fetch.assert_awaited_once()
    call_kwargs = mock_fetch.call_args
    assert call_kwargs.kwargs.get("limit") == 50 or call_kwargs.args[3] == 50


def test_logs_db_error_exits_1() -> None:
    """logs with DB error exits 1."""
    with patch(
        "tractable.cli.commands.logs._fetch_log",
        new=AsyncMock(side_effect=FatalError("DATABASE_URL not set.")),
    ):
        result = runner.invoke(app, ["logs"])

    assert result.exit_code == 1
    assert isinstance(result.exception, FatalError)
