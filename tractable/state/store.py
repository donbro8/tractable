"""PostgreSQL-backed AgentStateStore implementation.

Implements the AgentStateStore Protocol (tech-spec.py §2.3) using SQLAlchemy
async sessions backed by asyncpg.

Connection string is loaded exclusively from the ``DATABASE_URL`` environment
variable — never hardcoded.

Usage:
    store = PostgreSQLAgentStateStore.from_env()
    await store.save_agent_context("agent-1", context)
    ctx = await store.get_agent_context("agent-1")
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError, OperationalError, TimeoutError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tractable.errors import FatalError, RecoverableError, TransientError
from tractable.state.models import AgentCheckpointORM, AgentContextORM, AuditEntryORM
from tractable.types.agent import AgentCheckpoint, AgentContext, AuditEntry
from tractable.types.enums import TaskPhase

log = structlog.get_logger(__name__)

def _now() -> datetime:
    return datetime.now(tz=UTC)


@asynccontextmanager
async def _catch_db_errors() -> AsyncGenerator[None, None]:
    """Map SQLAlchemy exceptions to Tractable error taxonomy."""
    try:
        yield
    except OperationalError as exc:
        raise TransientError("Database connection lost or unreachable", retry_after=5) from exc
    except TimeoutError as exc:
        raise TransientError(
            "Database operation or pool acquisition timed out", retry_after=5
        ) from exc
    except IntegrityError as exc:
        raise RecoverableError(f"Database integrity constraint violated: {exc}") from exc


class PostgreSQLAgentStateStore:
    """PostgreSQL-backed implementation of the AgentStateStore Protocol.

    All database credentials are read from the ``DATABASE_URL`` environment
    variable (format: ``postgresql+asyncpg://user:pass@host:port/db``).
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    @classmethod
    def from_env(cls) -> PostgreSQLAgentStateStore:
        """Construct from the ``DATABASE_URL`` environment variable."""
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise FatalError("DATABASE_URL environment variable is not set.")

        pool_size = int(os.environ.get("TRACTABLE_PG_POOL_SIZE", "5"))
        max_overflow = int(os.environ.get("TRACTABLE_PG_POOL_MAX_OVERFLOW", "10"))

        engine = create_async_engine(
            url,
            pool_pre_ping=True,
            pool_size=pool_size,
            max_overflow=max_overflow,
        )
        factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            engine, expire_on_commit=False
        )
        return cls(factory)

    # ── AgentContext ──────────────────────────────────────────────────────────

    async def get_agent_context(self, agent_id: str) -> AgentContext:
        """Load agent context; raise ``RecoverableError`` if not found."""
        async with _catch_db_errors(), self._session_factory() as session:
            row = await session.get(AgentContextORM, agent_id)
            if row is None:
                raise RecoverableError(f"Agent context not found: {agent_id!r}")
            return _orm_to_context(row)

    async def save_agent_context(
        self,
        agent_id: str,
        context: AgentContext,
    ) -> None:
        """Upsert agent context (INSERT … ON CONFLICT UPDATE)."""
        values: dict[str, Any] = {
            "agent_id": agent_id,
            "base_template": context.base_template,
            "system_prompt": context.system_prompt,
            "repo_architectural_summary": context.repo_architectural_summary,
            "known_patterns": list(context.known_patterns),
            "pinned_instructions": list(context.pinned_instructions),
            "user_overrides": dict(context.user_overrides),
            "last_refreshed": context.last_refreshed,
            "updated_at": _now(),
        }
        stmt = (
            pg_insert(AgentContextORM)
            .values(**values)
            .on_conflict_do_update(
                index_elements=["agent_id"],
                set_={k: v for k, v in values.items() if k != "agent_id"},
            )
        )
        async with _catch_db_errors(), self._session_factory() as session, session.begin():
            await session.execute(stmt)

    # ── AgentCheckpoint ───────────────────────────────────────────────────────

    async def get_checkpoint(
        self,
        agent_id: str,
        task_id: str,
    ) -> AgentCheckpoint | None:
        """Return the most recent checkpoint for an agent+task pair."""
        async with _catch_db_errors(), self._session_factory() as session:
            result = await session.execute(
                select(AgentCheckpointORM)
                .where(
                    AgentCheckpointORM.agent_id == agent_id,
                    AgentCheckpointORM.task_id == task_id,
                )
                .order_by(AgentCheckpointORM.created_at.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            return _orm_to_checkpoint(row) if row is not None else None

    async def save_checkpoint(
        self,
        agent_id: str,
        task_id: str,
        checkpoint: AgentCheckpoint,
    ) -> None:
        """Append a new checkpoint row (all rows retained)."""
        row = AgentCheckpointORM(
            agent_id=agent_id,
            task_id=task_id,
            phase=str(checkpoint.phase),
            progress_summary=checkpoint.progress_summary,
            files_modified=list(checkpoint.files_modified),
            pending_actions=list(checkpoint.pending_actions),
            conversation_summary=checkpoint.conversation_summary,
            token_usage=checkpoint.token_usage,
            created_at=checkpoint.created_at,
        )
        async with _catch_db_errors(), self._session_factory() as session, session.begin():
            session.add(row)
        
        log.info(
            "checkpoint_saved",
            agent_id=agent_id,
            task_id=task_id,
            phase=str(checkpoint.phase),
        )

    # ── AuditEntry ────────────────────────────────────────────────────────────

    async def append_audit_entry(self, entry: AuditEntry) -> None:
        """Append to the immutable audit log (INSERT only)."""
        row = AuditEntryORM(
            timestamp=entry.timestamp,
            agent_id=entry.agent_id,
            task_id=entry.task_id,
            action=entry.action,
            detail=dict(entry.detail),
            outcome=entry.outcome,
        )
        async with _catch_db_errors(), self._session_factory() as session, session.begin():
            session.add(row)

        log.info(
            "audit_entry_appended",
            agent_id=entry.agent_id,
            entry_type=str(entry.action),
        )

    async def get_audit_log(
        self,
        agent_id: str | None = None,
        task_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> Sequence[AuditEntry]:
        """Query the audit log with optional filters, ordered by timestamp DESC."""
        stmt = select(AuditEntryORM).order_by(AuditEntryORM.timestamp.desc())
        if agent_id is not None:
            stmt = stmt.where(AuditEntryORM.agent_id == agent_id)
        if task_id is not None:
            stmt = stmt.where(AuditEntryORM.task_id == task_id)
        if since is not None:
            stmt = stmt.where(AuditEntryORM.timestamp >= since)
        stmt = stmt.limit(limit)

        async with _catch_db_errors(), self._session_factory() as session:
            result = await session.execute(stmt)
            return [_orm_to_audit_entry(row) for row in result.scalars()]


# ── Mapping helpers ───────────────────────────────────────────────────────────


def _orm_to_context(row: AgentContextORM) -> AgentContext:
    return AgentContext(
        agent_id=row.agent_id,
        base_template=row.base_template,
        system_prompt=row.system_prompt,
        repo_architectural_summary=row.repo_architectural_summary,
        known_patterns=list(row.known_patterns),
        pinned_instructions=list(row.pinned_instructions),
        user_overrides=dict(row.user_overrides),
        last_refreshed=row.last_refreshed,
    )


def _orm_to_checkpoint(row: AgentCheckpointORM) -> AgentCheckpoint:
    return AgentCheckpoint(
        task_id=row.task_id,
        phase=TaskPhase(row.phase),
        progress_summary=row.progress_summary,
        files_modified=list(row.files_modified),
        pending_actions=list(row.pending_actions),
        conversation_summary=row.conversation_summary,
        token_usage=row.token_usage,
        created_at=row.created_at,
    )


def _orm_to_audit_entry(row: AuditEntryORM) -> AuditEntry:
    return AuditEntry(
        timestamp=row.timestamp,
        agent_id=row.agent_id,
        task_id=row.task_id,
        action=row.action,
        detail=dict(row.detail),
        outcome=row.outcome,  # type: ignore[arg-type]
    )
