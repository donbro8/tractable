"""Agent state and context value types for the Tractable framework.

Sources:
- tech-spec.py §2.3 — AgentContext, AgentCheckpoint, AuditEntry
- realtime-temporal-spec.py §E — TemporalAgentContext, ChangeVelocity
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from tractable.types.enums import AgentStatus, TaskPhase
from tractable.types.temporal import AgentReactivityConfig, ChangeNotification


class AgentContext(BaseModel):
    """
    The AGENT.md equivalent — stored as structured data, not as a file in the
    repository. The agent's identity, instructions, and accumulated knowledge.

    Assembled from three layers (lowest → highest priority):
    base template → registration overrides → human-pinned instructions.
    """

    agent_id: str
    repo: str = ""
    base_template: str
    system_prompt: str
    repo_architectural_summary: str
    known_patterns: list[str] = Field(default_factory=list)
    pinned_instructions: list[str] = Field(default_factory=list)
    user_overrides: dict[str, Any] = Field(default_factory=dict)
    last_refreshed: datetime | None = None
    reactivity_config: AgentReactivityConfig = Field(default_factory=AgentReactivityConfig)


class AgentCheckpoint(BaseModel):
    """Serialisable snapshot of an agent's mid-task state.

    ``workflow_state`` holds the full ``AgentWorkflowState`` dict serialised as
    a JSON string.  This is the data ``resume_task()`` uses to reconstruct the
    LangGraph initial state on restart.  Defaults to ``"{}"`` (empty JSON
    object) for checkpoints that pre-date TASK-2.5.1.
    """

    task_id: str
    phase: TaskPhase
    progress_summary: str
    files_modified: list[str]
    pending_actions: list[str]
    conversation_summary: str
    token_usage: int
    created_at: datetime
    workflow_state: str = "{}"


class AuditEntry(BaseModel):
    """Append-only audit record for governance and compliance."""

    timestamp: datetime
    agent_id: str
    task_id: str | None = None
    action: str
    detail: dict[str, Any] = Field(default_factory=dict)
    outcome: Literal["success", "failure", "escalated"]


class ChangeVelocity(BaseModel):
    """How fast things are changing in an agent's domain."""

    commits_last_24h: int
    commits_last_7d: int
    entities_changed_last_24h: int
    entities_changed_last_7d: int
    cross_repo_changes_last_7d: int
    hotspot_files: list[str] = Field(default_factory=list)
    hotspot_entities: list[str] = Field(default_factory=list)


class TemporalAgentContext(BaseModel):
    """
    Extended agent context with temporal awareness.

    Includes all ``AgentContext`` fields plus temporal state so agents can
    quickly catch up on what changed while they were dormant.
    """

    agent_id: str
    base_template: str
    system_prompt: str
    repo_architectural_summary: str
    known_patterns: list[str] = Field(default_factory=list)
    pinned_instructions: list[str] = Field(default_factory=list)
    user_overrides: dict[str, Any] = Field(default_factory=dict)
    last_refreshed: datetime | None = None

    last_active: datetime | None = None
    last_known_head_sha: str | None = None
    recent_changes_digest: str = ""
    pending_notifications: list[ChangeNotification] = []
    change_velocity: ChangeVelocity | None = None


# Re-export AgentStatus so downstream modules can import agent types
# from a single location without needing to reach into enums directly.
__all__ = [
    "AgentContext",
    "AgentCheckpoint",
    "AgentReactivityConfig",
    "AgentStatus",
    "AuditEntry",
    "ChangeVelocity",
    "TaskPhase",
    "TemporalAgentContext",
]
