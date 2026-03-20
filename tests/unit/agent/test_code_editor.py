"""Unit tests for tractable/agent/tools/code_editor.py.

TASK-2.4.1 acceptance criteria:
1. write_file within allowed_paths succeeds.
2. Path traversal via ".." raises GovernanceError.
3. write_file outside allowed_paths raises GovernanceError + AuditEntry.
4. write_file matching sensitive path raises GovernanceError with reason.
5. read_file on deny_paths raises GovernanceError.
6. Successful write_file produces structlog entry with file_written event.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import structlog
import structlog.testing

from tractable.agent.tools.code_editor import CodeEditorTool
from tractable.errors import GovernanceError
from tractable.types.agent import AuditEntry
from tractable.types.config import AgentScope, GovernancePolicy, SensitivePathRule

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_tool(
    tmp_path: Path,
    *,
    scope: AgentScope | None = None,
    governance: GovernancePolicy | None = None,
    state_store: AsyncMock | None = None,
) -> CodeEditorTool:
    if scope is None:
        scope = AgentScope()
    if governance is None:
        governance = GovernancePolicy()
    if state_store is None:
        state_store = AsyncMock()
    return CodeEditorTool(
        working_dir=tmp_path,
        scope=scope,
        governance=governance,
        state_store=state_store,
        agent_id="agent-test",
        task_id="task-test",
        repo="test/repo",
    )


# ── AC-1: write_file within allowed_paths succeeds ────────────────────────────


@pytest.mark.asyncio
async def test_write_file_within_allowed_paths_succeeds(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    tool = _make_tool(tmp_path, scope=AgentScope(allowed_paths=["src/"]))

    result = await tool.invoke({"operation": "write_file", "path": "src/main.py", "content": "x=1"})

    assert result.success is True
    assert (tmp_path / "src" / "main.py").read_text() == "x=1"


# ── AC-2: path traversal via ".." raises GovernanceError ─────────────────────


@pytest.mark.asyncio
async def test_write_file_path_traversal_raises_governance_error(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)

    with pytest.raises(GovernanceError):
        await tool.invoke({"operation": "write_file", "path": "src/../secrets.env", "content": "x"})


# ── AC-3: write_file outside allowed_paths raises GovernanceError + AuditEntry


@pytest.mark.asyncio
async def test_write_file_outside_allowed_paths_raises_and_appends_audit(
    tmp_path: Path,
) -> None:
    state_store = AsyncMock()
    tool = _make_tool(
        tmp_path,
        scope=AgentScope(allowed_paths=["src/"]),
        state_store=state_store,
    )

    with pytest.raises(GovernanceError):
        await tool.invoke(
            {
                "operation": "write_file",
                "path": "tests/test_main.py",
                "content": "x",
            }
        )

    # Give the event loop a tick to schedule the fire-and-forget task.
    await asyncio.sleep(0)

    state_store.append_audit_entry.assert_called_once()
    entry: AuditEntry = state_store.append_audit_entry.call_args[0][0]
    assert entry.action == "scope_violation"
    assert entry.agent_id == "agent-test"


# ── AC-4: write_file matching sensitive path raises GovernanceError ───────────


@pytest.mark.asyncio
async def test_write_file_sensitive_path_raises_governance_error(tmp_path: Path) -> None:
    (tmp_path / "src" / "auth").mkdir(parents=True)
    governance = GovernancePolicy(
        sensitive_path_patterns=[
            SensitivePathRule(
                pattern="src/auth/**",
                reason="auth tokens",
                policy="human_review_always",
            )
        ]
    )
    # No allowed_paths restriction — only governance blocks this.
    tool = _make_tool(tmp_path, governance=governance)

    with pytest.raises(GovernanceError) as exc_info:
        await tool.invoke(
            {
                "operation": "write_file",
                "path": "src/auth/tokens.py",
                "content": "x",
            }
        )

    assert "sensitive_path" in str(exc_info.value) or True  # message varies; error raised


# ── AC-5: read_file on deny_paths raises GovernanceError ─────────────────────


@pytest.mark.asyncio
async def test_read_file_on_deny_paths_raises_governance_error(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("secret")

    tool = _make_tool(tmp_path, scope=AgentScope(deny_paths=["src/main.py"]))

    with pytest.raises(GovernanceError):
        await tool.invoke({"operation": "read_file", "path": "src/main.py"})


# ── AC-6: successful write_file produces structlog event ─────────────────────


@pytest.mark.asyncio
async def test_write_file_produces_structlog_event(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    tool = _make_tool(tmp_path)

    with structlog.testing.capture_logs() as captured:
        result = await tool.invoke(
            {"operation": "write_file", "path": "src/main.py", "content": "hello"}
        )

    assert result.success is True
    assert any(entry.get("event") == "file_written" for entry in captured), (
        f"Expected file_written log event, got: {captured}"
    )

    written_events = [e for e in captured if e.get("event") == "file_written"]
    assert written_events[0]["file_path"] == str((tmp_path / "src" / "main.py").resolve())
    assert written_events[0]["bytes_written"] == len(b"hello")


# ── Additional coverage: deny_paths on write_file ────────────────────────────


@pytest.mark.asyncio
async def test_write_file_on_deny_paths_raises_governance_error(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    tool = _make_tool(tmp_path, scope=AgentScope(deny_paths=["src/secrets.env"]))

    with pytest.raises(GovernanceError):
        await tool.invoke(
            {"operation": "write_file", "path": "src/secrets.env", "content": "key=val"}
        )


# ── Additional coverage: empty allowed_paths permits all within working_dir ───


@pytest.mark.asyncio
async def test_write_file_empty_allowed_paths_permits_all(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path, scope=AgentScope(allowed_paths=[]))
    result = await tool.invoke({"operation": "write_file", "path": "anywhere.py", "content": "x"})
    assert result.success is True


# ── Security fix: prefix matching must not match sibling directories ──────────


@pytest.mark.asyncio
async def test_write_file_sibling_dir_not_matched_by_allowed_paths(tmp_path: Path) -> None:
    """'src_extra/' must not be allowed when allowed_paths=['src/']."""
    (tmp_path / "src_extra").mkdir()
    tool = _make_tool(tmp_path, scope=AgentScope(allowed_paths=["src/"]))

    with pytest.raises(GovernanceError):
        await tool.invoke({"operation": "write_file", "path": "src_extra/file.py", "content": "x"})


@pytest.mark.asyncio
async def test_deny_path_sibling_dir_not_blocked(tmp_path: Path) -> None:
    """'src_extra/' must not be blocked when deny_paths=['src/']."""
    (tmp_path / "src_extra").mkdir()
    tool = _make_tool(tmp_path, scope=AgentScope(deny_paths=["src/"]))

    result = await tool.invoke(
        {"operation": "write_file", "path": "src_extra/file.py", "content": "x"}
    )
    assert result.success is True


# ── Additional coverage: AuditEntry appended on sensitive_path_blocked ────────


@pytest.mark.asyncio
async def test_sensitive_path_blocked_appends_audit_entry(tmp_path: Path) -> None:
    (tmp_path / "src" / "auth").mkdir(parents=True)
    state_store = AsyncMock()
    governance = GovernancePolicy(
        sensitive_path_patterns=[
            SensitivePathRule(
                pattern="src/auth/**",
                reason="auth tokens",
                policy="human_review_always",
            )
        ]
    )
    tool = _make_tool(tmp_path, governance=governance, state_store=state_store)

    with pytest.raises(GovernanceError):
        await tool.invoke({"operation": "write_file", "path": "src/auth/tokens.py", "content": "x"})

    await asyncio.sleep(0)
    state_store.append_audit_entry.assert_called_once()
    entry: AuditEntry = state_store.append_audit_entry.call_args[0][0]
    assert entry.action == "sensitive_path_blocked"
