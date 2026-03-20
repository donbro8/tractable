"""Unit tests for PostgreSQLAgentStateStore.

All tests mock the SQLAlchemy async session — no live PostgreSQL required.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tractable.errors import RecoverableError, TransientError
from tractable.protocols.agent_state_store import AgentStateStore
from tractable.state.models import AgentCheckpointORM, AgentContextORM, AuditEntryORM
from tractable.state.store import PostgreSQLAgentStateStore, _orm_to_context
from tractable.types.agent import AgentCheckpoint, AgentContext, AuditEntry
from tractable.types.enums import TaskPhase

# ── Fixtures ──────────────────────────────────────────────────────────────────

NOW = datetime(2026, 3, 19, 10, 0, 0, tzinfo=UTC)


def make_context(agent_id: str = "agent-1") -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        base_template="api_maintainer",
        system_prompt="You are a coding agent.",
        repo_architectural_summary="REST API backed by PostgreSQL.",
        known_patterns=["use dependency injection"],
        pinned_instructions=["never break the public API"],
        user_overrides={"verbosity": "low"},
        last_refreshed=NOW,
    )


def make_checkpoint() -> AgentCheckpoint:
    return AgentCheckpoint(
        task_id="task-42",
        phase=TaskPhase.EXECUTING,
        progress_summary="halfway done",
        files_modified=["src/api.py"],
        pending_actions=["run tests"],
        conversation_summary="Started refactor.",
        token_usage=1500,
        created_at=NOW,
    )


def make_audit_entry(agent_id: str = "agent-1") -> AuditEntry:
    return AuditEntry(
        timestamp=NOW,
        agent_id=agent_id,
        task_id="task-42",
        action="file_write",
        detail={"path": "src/api.py"},
        outcome="success",
    )


def make_context_orm(agent_id: str = "agent-1") -> AgentContextORM:
    row = AgentContextORM()
    row.agent_id = agent_id
    row.repo = "my-api"
    row.base_template = "api_maintainer"
    row.system_prompt = "You are a coding agent."
    row.repo_architectural_summary = "REST API backed by PostgreSQL."
    row.known_patterns = ["use dependency injection"]
    row.pinned_instructions = ["never break the public API"]
    row.user_overrides = {"verbosity": "low"}
    row.last_refreshed = NOW
    row.last_active = None
    row.last_known_head_sha = None
    row.recent_changes_digest = ""
    row.updated_at = NOW
    return row


def make_store() -> tuple[PostgreSQLAgentStateStore, MagicMock]:
    """Return a store wired to a mock session_factory."""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_begin = AsyncMock()
    mock_begin.__aenter__ = AsyncMock(return_value=None)
    mock_begin.__aexit__ = AsyncMock(return_value=None)
    mock_session.begin = MagicMock(return_value=mock_begin)

    session_factory = MagicMock(return_value=mock_session)
    store = PostgreSQLAgentStateStore(session_factory)  # type: ignore[arg-type]
    return store, mock_session


# ── _orm_to_context ───────────────────────────────────────────────────────────


class TestOrmToContext:
    def test_round_trip_fields(self) -> None:
        orm = make_context_orm()
        ctx = _orm_to_context(orm)
        assert ctx.agent_id == "agent-1"
        assert ctx.base_template == "api_maintainer"
        assert ctx.known_patterns == ["use dependency injection"]
        assert ctx.user_overrides == {"verbosity": "low"}
        assert ctx.last_refreshed == NOW


# ── get_agent_context ─────────────────────────────────────────────────────────


class TestGetAgentContext:
    @pytest.mark.asyncio
    async def test_returns_context_when_found(self) -> None:
        store, mock_session = make_store()
        mock_session.get = AsyncMock(return_value=make_context_orm())
        result = await store.get_agent_context("agent-1")
        assert result.agent_id == "agent-1"
        mock_session.get.assert_awaited_once_with(AgentContextORM, "agent-1")

    @pytest.mark.asyncio
    async def test_raises_recoverable_error_when_missing(self) -> None:
        store, mock_session = make_store()
        mock_session.get = AsyncMock(return_value=None)
        with pytest.raises(RecoverableError, match="agent-1"):
            await store.get_agent_context("agent-1")


# ── Error Mapping ─────────────────────────────────────────────────────────────


class TestDbErrorMapping:
    @pytest.mark.asyncio
    async def test_operational_error_mapped_to_transient(self) -> None:
        from sqlalchemy.exc import OperationalError

        store, mock_session = make_store()
        mock_session.execute = AsyncMock(side_effect=OperationalError("select", "params", "orig"))
        with pytest.raises(TransientError, match="Database connection lost"):
            await store.get_checkpoint("agent-1", "task-1")

    @pytest.mark.asyncio
    async def test_timeout_error_mapped_to_transient(self) -> None:
        store, mock_session = make_store()
        mock_session.execute = AsyncMock(side_effect=TimeoutError("timeout"))
        with pytest.raises(TransientError, match="timed out"):
            await store.get_checkpoint("agent-1", "task-1")

    @pytest.mark.asyncio
    async def test_transient_error_on_db_unreachable(self) -> None:
        """AC-2: get_agent_context raises TransientError when DB is unreachable."""
        from sqlalchemy.exc import OperationalError

        store, mock_session = make_store()
        mock_session.get = AsyncMock(side_effect=OperationalError("connect", "params", "orig"))
        with pytest.raises(TransientError, match="unreachable"):
            await store.get_agent_context("agent-1")

    @pytest.mark.asyncio
    async def test_integrity_error_mapped_to_recoverable(self) -> None:
        from sqlalchemy.exc import IntegrityError

        store, mock_session = make_store()
        mock_session.execute = AsyncMock(side_effect=IntegrityError("insert", "params", "orig"))
        with pytest.raises(RecoverableError, match="integrity constraint violated"):
            await store.save_agent_context("agent-1", make_context())


# ── save_agent_context ────────────────────────────────────────────────────────


class TestSaveAgentContext:
    @pytest.mark.asyncio
    async def test_executes_upsert(self) -> None:
        store, mock_session = make_store()
        mock_session.execute = AsyncMock(return_value=None)
        ctx = make_context()
        await store.save_agent_context("agent-1", ctx)
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_uses_agent_id_from_param(self) -> None:
        store, mock_session = make_store()
        execute_calls: list[Any] = []
        mock_session.execute = AsyncMock(side_effect=lambda stmt: execute_calls.append(stmt))
        ctx = make_context(agent_id="agent-other")
        await store.save_agent_context("agent-99", ctx)
        assert len(execute_calls) == 1


# ── get_checkpoint ────────────────────────────────────────────────────────────


class TestGetCheckpoint:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self) -> None:
        store, mock_session = make_store()
        scalar_result = MagicMock()
        scalar_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=scalar_result)
        result = await store.get_checkpoint("agent-1", "task-42")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_checkpoint_when_found(self) -> None:
        store, mock_session = make_store()
        orm_row = AgentCheckpointORM()
        orm_row.task_id = "task-42"
        orm_row.phase = "executing"
        orm_row.progress_summary = "halfway"
        orm_row.files_modified = ["src/api.py"]
        orm_row.pending_actions = ["run tests"]
        orm_row.conversation_summary = "started"
        orm_row.token_usage = 1500
        orm_row.created_at = NOW

        scalar_result = MagicMock()
        scalar_result.scalar_one_or_none = MagicMock(return_value=orm_row)
        mock_session.execute = AsyncMock(return_value=scalar_result)

        result = await store.get_checkpoint("agent-1", "task-42")
        assert result is not None
        assert result.task_id == "task-42"
        assert result.token_usage == 1500


# ── save_checkpoint ───────────────────────────────────────────────────────────


class TestSaveCheckpoint:
    @pytest.mark.asyncio
    async def test_adds_row_to_session_and_logs(self) -> None:
        store, mock_session = make_store()
        added: list[Any] = []
        mock_session.add = MagicMock(side_effect=lambda row: added.append(row))
        cp = make_checkpoint()
        with patch("tractable.state.store.log.info") as mock_info:
            await store.save_checkpoint("agent-1", "task-42", cp)

        assert len(added) == 1
        assert isinstance(added[0], AgentCheckpointORM)
        assert added[0].task_id == "task-42"

        mock_info.assert_called_once_with(
            "checkpoint_saved",
            agent_id="agent-1",
            task_id="task-42",
            phase="executing",
        )


# ── append_audit_entry ────────────────────────────────────────────────────────


class TestAppendAuditEntry:
    @pytest.mark.asyncio
    async def test_adds_audit_row_and_logs(self) -> None:
        store, mock_session = make_store()
        added: list[Any] = []
        mock_session.add = MagicMock(side_effect=lambda row: added.append(row))
        entry = make_audit_entry()
        with patch("tractable.state.store.log.info") as mock_info:
            await store.append_audit_entry(entry)

        assert len(added) == 1
        assert isinstance(added[0], AuditEntryORM)
        assert added[0].outcome == "success"
        assert added[0].action == "file_write"

        mock_info.assert_called_once_with(
            "audit_entry_appended",
            agent_id="agent-1",
            entry_type="file_write",
        )


# ── get_audit_log ─────────────────────────────────────────────────────────────


class TestGetAuditLog:
    @pytest.mark.asyncio
    async def test_returns_entries(self) -> None:
        store, mock_session = make_store()
        orm_entry = AuditEntryORM()
        orm_entry.timestamp = NOW
        orm_entry.agent_id = "agent-1"
        orm_entry.task_id = "task-42"
        orm_entry.action = "file_write"
        orm_entry.detail = {"path": "src/api.py"}
        orm_entry.outcome = "success"

        scalars_result = MagicMock()
        scalars_result.__iter__ = MagicMock(return_value=iter([orm_entry]))
        execute_result = MagicMock()
        execute_result.scalars = MagicMock(return_value=scalars_result)
        mock_session.execute = AsyncMock(return_value=execute_result)

        results = await store.get_audit_log(agent_id="agent-1", limit=10)
        assert len(results) == 1
        assert results[0].agent_id == "agent-1"
        assert results[0].outcome == "success"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_entries(self) -> None:
        store, mock_session = make_store()
        scalars_result = MagicMock()
        scalars_result.__iter__ = MagicMock(return_value=iter([]))
        execute_result = MagicMock()
        execute_result.scalars = MagicMock(return_value=scalars_result)
        mock_session.execute = AsyncMock(return_value=execute_result)

        results = await store.get_audit_log(agent_id="agent-1")
        assert results == []


# ── Protocol conformance ──────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_isinstance_check_passes(self) -> None:
        store, _ = make_store()
        assert isinstance(store, AgentStateStore)
