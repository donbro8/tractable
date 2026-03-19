"""Initial schema: agent_contexts, agent_checkpoints, audit_log.

Revision ID: 0001
Revises:
Create Date: 2026-03-19

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_contexts",
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("base_template", sa.Text(), nullable=False, server_default=""),
        sa.Column("system_prompt", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "repo_architectural_summary",
            sa.Text(),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "known_patterns",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "pinned_instructions",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "user_overrides",
            sa.JSON(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("last_refreshed", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_active", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_known_head_sha", sa.String(), nullable=True),
        sa.Column(
            "recent_changes_digest",
            sa.Text(),
            nullable=False,
            server_default="",
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("agent_id"),
    )

    op.create_table(
        "agent_checkpoints",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("task_id", sa.String(), nullable=False),
        sa.Column("phase", sa.String(), nullable=False),
        sa.Column(
            "progress_summary", sa.Text(), nullable=False, server_default=""
        ),
        sa.Column(
            "files_modified",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "pending_actions",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "conversation_summary", sa.Text(), nullable=False, server_default=""
        ),
        sa.Column("token_usage", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["agent_id"], ["agent_contexts.agent_id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "audit_log",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("task_id", sa.String(), nullable=True),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("outcome", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("agent_checkpoints")
    op.drop_table("agent_contexts")
