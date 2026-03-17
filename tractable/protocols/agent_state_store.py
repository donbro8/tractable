"""AgentStateStore Protocol — persistent store for agent state.

Source: tech-spec.py §2.3
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from tractable.types.agent import AgentCheckpoint, AgentContext, AuditEntry


@runtime_checkable
class AgentStateStore(Protocol):
    """
    Persistent store for agent contexts, conversation history, and
    checkpoint state. Agent state lives here — EXTERNAL to any repository
    the agent works on.
    """

    async def get_agent_context(self, agent_id: str) -> AgentContext:
        """Load the full context document for an agent."""
        ...

    async def save_agent_context(
        self,
        agent_id: str,
        context: AgentContext,
    ) -> None:
        """Persist updated agent context."""
        ...

    async def get_checkpoint(
        self,
        agent_id: str,
        task_id: str,
    ) -> AgentCheckpoint | None:
        """Load a task checkpoint for crash recovery."""
        ...

    async def save_checkpoint(
        self,
        agent_id: str,
        task_id: str,
        checkpoint: AgentCheckpoint,
    ) -> None:
        """Persist task progress for crash recovery."""
        ...

    async def append_audit_entry(self, entry: AuditEntry) -> None:
        """Append to the immutable audit log."""
        ...

    async def get_audit_log(
        self,
        agent_id: str | None = None,
        task_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> Sequence[AuditEntry]:
        """Query the audit log."""
        ...
