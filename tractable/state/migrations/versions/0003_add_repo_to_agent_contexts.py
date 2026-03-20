"""Add repo column to agent_contexts.

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-20

Adds the ``repo`` TEXT column introduced by TASK-2.6.3.  The column stores
the registered repository name so that ``tractable agent list`` can display
it without needing to re-assemble the full agent context.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_contexts",
        sa.Column(
            "repo",
            sa.Text(),
            nullable=False,
            server_default="",
        ),
    )


def downgrade() -> None:
    op.drop_column("agent_contexts", "repo")
