"""Add repo_poll_state table for ChangePoller.

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-25

Adds the ``repo_poll_state`` table introduced by TASK-3.2.2.  The table
stores the last commit SHA seen by ``ChangePoller`` for each polled repo,
enabling the poller to detect new commits without re-processing unchanged
repos.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "repo_poll_state",
        sa.Column("repo_id", sa.String(), nullable=False),
        sa.Column("last_polled_sha", sa.String(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("repo_id"),
    )


def downgrade() -> None:
    op.drop_table("repo_poll_state")
