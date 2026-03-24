"""Integration tests for PostgreSQLAgentStateStore against live PostgreSQL.

Requires the Docker Compose stack:
  docker compose -f deploy/docker-compose.yml up -d postgres

Run with:
  uv run pytest tests/integration/state/test_store_live.py

DATABASE_URL must be set in the environment (see .env.example).
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest

from tractable.errors import RecoverableError
from tractable.protocols.agent_state_store import AgentStateStore
from tractable.state.store import PostgreSQLAgentStateStore
from tractable.types.agent import AgentCheckpoint, AgentContext, AuditEntry
from tractable.types.enums import TaskPhase

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://tractable:tractable_dev@localhost:5433/tractable",
)

NOW = datetime(2026, 3, 19, 10, 0, 0, tzinfo=UTC)


@pytest.fixture()
def store() -> PostgreSQLAgentStateStore:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    engine = create_async_engine(_DATABASE_URL, pool_pre_ping=True)
    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(engine, expire_on_commit=False)
    return PostgreSQLAgentStateStore(factory)


def _context(agent_id: str) -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        base_template="api_maintainer",
        system_prompt="You are a coding agent.",
        repo_architectural_summary="REST API.",
        known_patterns=["use DI"],
        pinned_instructions=["never break the API"],
        user_overrides={"verbosity": "low"},
        last_refreshed=NOW,
    )


# ── AC2: save_agent_context → get_agent_context round-trip ───────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_and_get_agent_context_round_trip(
    store: PostgreSQLAgentStateStore,
) -> None:
    agent_id = f"test-agent-{uuid.uuid4()}"
    ctx = _context(agent_id)

    await store.save_agent_context(agent_id, ctx)
    result = await store.get_agent_context(agent_id)

    assert result.agent_id == agent_id
    assert result.base_template == ctx.base_template
    assert result.system_prompt == ctx.system_prompt
    assert result.known_patterns == ctx.known_patterns
    assert result.user_overrides == ctx.user_overrides


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_agent_context_upsert(store: PostgreSQLAgentStateStore) -> None:
    """Saving twice must update, not duplicate."""
    agent_id = f"test-agent-{uuid.uuid4()}"
    await store.save_agent_context(agent_id, _context(agent_id))

    updated = AgentContext(
        agent_id=agent_id,
        base_template="infra_maintainer",
        system_prompt="Updated prompt.",
        repo_architectural_summary="Updated.",
    )
    await store.save_agent_context(agent_id, updated)
    result = await store.get_agent_context(agent_id)
    assert result.base_template == "infra_maintainer"
    assert result.system_prompt == "Updated prompt."


# ── AC3: append_audit_entry → get_audit_log ──────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_append_and_get_audit_entry(store: PostgreSQLAgentStateStore) -> None:
    agent_id = f"test-agent-{uuid.uuid4()}"

    # Must save context first (FK constraint)
    await store.save_agent_context(agent_id, _context(agent_id))

    entry = AuditEntry(
        timestamp=NOW,
        agent_id=agent_id,
        task_id="task-1",
        action="file_write",
        detail={"path": "src/api.py"},
        outcome="success",
    )
    await store.append_audit_entry(entry)

    log = await store.get_audit_log(agent_id=agent_id, limit=10)
    assert len(log) >= 1
    found = next((e for e in log if e.agent_id == agent_id), None)
    assert found is not None
    assert found.outcome == "success"
    assert found.action == "file_write"


# ── AC4: get_agent_context on missing agent raises KeyError ───────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_agent_context_missing_raises_recoverable_error(
    store: PostgreSQLAgentStateStore,
) -> None:
    with pytest.raises(RecoverableError):
        await store.get_agent_context(f"nonexistent-{uuid.uuid4()}")


# ── AC5: isinstance(store, AgentStateStore) == True ──────────────────────────


@pytest.mark.integration
def test_protocol_isinstance(store: PostgreSQLAgentStateStore) -> None:
    assert isinstance(store, AgentStateStore)


# ── Checkpoint round-trip ─────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_and_get_checkpoint(store: PostgreSQLAgentStateStore) -> None:
    agent_id = f"test-agent-{uuid.uuid4()}"
    await store.save_agent_context(agent_id, _context(agent_id))

    cp = AgentCheckpoint(
        task_id="task-99",
        phase=TaskPhase.EXECUTING,
        progress_summary="halfway",
        files_modified=["src/api.py"],
        pending_actions=["run tests"],
        conversation_summary="Started.",
        token_usage=1000,
        created_at=NOW,
    )
    await store.save_checkpoint(agent_id, "task-99", cp)
    result = await store.get_checkpoint(agent_id, "task-99")
    assert result is not None
    assert result.task_id == "task-99"
    assert result.token_usage == 1000


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_checkpoint_missing_returns_none(
    store: PostgreSQLAgentStateStore,
) -> None:
    result = await store.get_checkpoint(f"no-such-{uuid.uuid4()}", "no-task")
    assert result is None
