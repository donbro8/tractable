"""Add workflow_state column to agent_checkpoints.

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-19

Adds the ``workflow_state`` TEXT column introduced by TASK-2.5.1.  The column
stores the full ``AgentWorkflowState`` dict serialised as JSON so that
``resume_task()`` can reconstruct the LangGraph state on restart without
re-executing completed phases.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_checkpoints",
        sa.Column(
            "workflow_state",
            sa.Text(),
            nullable=False,
            server_default="{}",
        ),
    )


def downgrade() -> None:
    op.drop_column("agent_checkpoints", "workflow_state")
