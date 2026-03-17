"""Task-related value types (stubs for Phase 2).

Models are defined now so other modules can import them; task execution
logic is implemented in Phase 2.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from tractable.types.enums import TaskPhase

TaskStatus = Literal["submitted", "in_progress", "completed", "failed", "cancelled"]


class TaskHandle(BaseModel):
    """Lightweight reference to a running or completed task."""

    task_id: str
    agent_id: str
    repo_name: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime


class Task(BaseModel):
    """Full task record."""

    task_id: str
    description: str
    repo_name: str
    agent_id: str
    phase: TaskPhase
    created_at: datetime
    result_pr_url: str | None = None
