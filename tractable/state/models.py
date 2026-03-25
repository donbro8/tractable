"""SQLAlchemy ORM models for the AgentStateStore.

Tables:
- agent_contexts  — one row per agent (upserted on save)
- agent_checkpoints — append-only per-task snapshots
- audit_log       — immutable audit trail

Source: tech-spec.py §2.3
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AgentContextORM(Base):
    """Persistent agent identity and accumulated knowledge."""

    __tablename__ = "agent_contexts"

    agent_id: Mapped[str] = mapped_column(String, primary_key=True)
    repo: Mapped[str] = mapped_column(Text, nullable=False, server_default="", default="")
    base_template: Mapped[str] = mapped_column(Text, nullable=False, default="")
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    repo_architectural_summary: Mapped[str] = mapped_column(
        Text, nullable=False, default=""
    )
    known_patterns: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    pinned_instructions: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list
    )
    user_overrides: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    last_refreshed: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_active: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_known_head_sha: Mapped[str | None] = mapped_column(String)
    recent_changes_digest: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AgentCheckpointORM(Base):
    """Mid-task snapshot — all rows retained (append-only per task)."""

    __tablename__ = "agent_checkpoints"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(
        String, ForeignKey("agent_contexts.agent_id"), nullable=False
    )
    task_id: Mapped[str] = mapped_column(String, nullable=False)
    phase: Mapped[str] = mapped_column(String, nullable=False)
    progress_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    files_modified: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    pending_actions: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list
    )
    conversation_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    token_usage: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    workflow_state: Mapped[str] = mapped_column(Text, nullable=False, server_default="{}")


class RepoPollStateORM(Base):
    """Tracks the last-polled commit SHA for repos using the polling fallback."""

    __tablename__ = "repo_poll_state"

    repo_id: Mapped[str] = mapped_column(String, primary_key=True)
    last_polled_sha: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditEntryORM(Base):
    """Immutable audit record — INSERT only, no UPDATE or DELETE."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    task_id: Mapped[str | None] = mapped_column(String)
    action: Mapped[str] = mapped_column(String, nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    outcome: Mapped[str] = mapped_column(String, nullable=False)
