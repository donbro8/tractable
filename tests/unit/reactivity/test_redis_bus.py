"""Unit tests for RedisEventBus (TASK-2.6.2).

Covers:
- AC-2: publish() with Redis unavailable raises TransientError.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from tractable.errors import TransientError
from tractable.events.redis_bus import RedisEventBus
from tractable.protocols.event_bus import AgentEvent


def _sample_event() -> AgentEvent:
    return AgentEvent(
        event_id="ev-unit-001",
        timestamp=datetime.now(tz=UTC),
        source_agent_id="agent-a",
        target_agent_id="agent-b",
        event_type="test.ping",
    )


# ── AC-2: ConnectionError → TransientError ────────────────────────────────


@pytest.mark.asyncio
async def test_publish_connection_error_raises_transient_error() -> None:
    """AC-2: Redis ConnectionError during publish is wrapped in TransientError."""
    redis = MagicMock()
    redis.publish = AsyncMock(side_effect=ConnectionError("Redis is down"))

    bus = RedisEventBus(redis)

    with pytest.raises(TransientError, match="Redis publish failed"):
        await bus.publish("some-topic", _sample_event())


@pytest.mark.asyncio
async def test_publish_success_does_not_raise() -> None:
    """publish() with a working Redis does not raise."""
    redis = MagicMock()
    redis.publish = AsyncMock(return_value=1)  # 1 subscriber received the message

    bus = RedisEventBus(redis)
    # Should not raise.
    await bus.publish("some-topic", _sample_event())
    redis.publish.assert_called_once()
