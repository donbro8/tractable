"""CI/CD pipeline event types (stubs for Phase 2).

Models are defined now so other modules can import them; pipeline
integration logic is implemented in Phase 2.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

CheckRunStatus = Literal["queued", "in_progress", "completed", "action_required"]


class PipelineEvent(BaseModel):
    """A CI/CD pipeline status event."""

    event_id: str
    repo_name: str
    branch: str
    commit_sha: str
    check_name: str
    status: CheckRunStatus
    conclusion: str | None = None
    timestamp: datetime


class PipelineFailureEvent(PipelineEvent):
    """A pipeline event that represents a failure with diagnostic detail."""

    failure_log_url: str
    failure_category: Literal["flaky", "agent_caused", "environment"]
