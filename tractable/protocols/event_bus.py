"""EventBus Protocol — async message bus for inter-agent coordination.

Source: tech-spec.py §2.5
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# ── Supporting value type ──────────────────────────────────────────────


class AgentEvent(BaseModel):
    """A coordination message published to the event bus."""

    event_id: str
    timestamp: datetime
    source_agent_id: str
    target_agent_id: str | None = None  # None = broadcast
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    affected_entity_ids: list[str] = Field(default_factory=list)
    requires_response: bool = False


# ── Protocol ───────────────────────────────────────────────────────────


@runtime_checkable
class EventBus(Protocol):
    """
    Async message bus for inter-agent coordination.
    Messages are typed and routed by topic.
    """

    async def publish(self, topic: str, event: AgentEvent) -> None:
        """Publish an event to a topic."""
        ...

    async def subscribe(
        self,
        topic: str,
        agent_id: str,
    ) -> AsyncIterator[AgentEvent]:
        """Subscribe to a topic and yield incoming events."""
        ...
