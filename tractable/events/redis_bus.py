"""Redis Pub/Sub EventBus implementation.

TASK-2.6.2 — Implement Redis EventBus and NotificationRouter.

``RedisEventBus`` satisfies the ``EventBus`` Protocol from
``tractable/protocols/event_bus.py``.  It uses Redis Pub/Sub for
real-time event delivery between agents.

Topics follow the convention:
  - ``agent.{agent_id}.notifications``  — per-agent change notifications
  - ``repo.{repo_name}.changes``         — repo-level change events
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Protocol

import structlog
from redis.exceptions import RedisError

from tractable.errors import TransientError
from tractable.protocols.event_bus import AgentEvent

_log = structlog.get_logger()

# Topic naming conventions.
TOPIC_AGENT_NOTIFICATIONS = "agent.{agent_id}.notifications"
TOPIC_REPO_CHANGES = "repo.{repo_name}.changes"


# ── Internal Redis protocol stubs ──────────────────────────────────────────


class _AsyncPubSub(Protocol):
    """Minimal async Redis PubSub protocol used internally."""

    async def subscribe(self, *channels: str) -> None:
        ...

    async def unsubscribe(self, *channels: str) -> None:
        ...

    def listen(self) -> AsyncIterator[dict[str, Any]]:
        """Yield raw pub/sub message dicts as they arrive."""
        ...

    async def aclose(self) -> None:
        ...


class _AsyncRedis(Protocol):
    """Minimal async Redis client protocol required by ``RedisEventBus``."""

    async def publish(self, channel: str, message: bytes) -> int:
        """Publish *message* to *channel*. Returns the subscriber count."""
        ...

    def pubsub(self) -> _AsyncPubSub:
        """Return a new PubSub object bound to this connection."""
        ...


# ── EventBus implementation ────────────────────────────────────────────────


class RedisEventBus:
    """Redis Pub/Sub implementation of the ``EventBus`` Protocol.

    Parameters
    ----------
    redis:
        An async Redis client.  Must expose ``publish(channel, message)``
        and ``pubsub()``.  Compatible with ``redis.asyncio.Redis``.
    """

    def __init__(self, redis: _AsyncRedis) -> None:
        self._redis = redis

    async def publish(self, topic: str, event: AgentEvent) -> None:
        """Serialise *event* to JSON and publish it on *topic*.

        Raises ``TransientError`` if the Redis connection is unavailable.
        """
        data: bytes = event.model_dump_json().encode()
        try:
            await self._redis.publish(topic, data)
        except (ConnectionError, RedisError) as exc:
            raise TransientError(
                f"Redis publish failed on topic {topic!r}: {exc}"
            ) from exc
        _log.debug(
            "event_published",
            topic=topic,
            event_id=event.event_id,
            event_type=event.event_type,
        )

    async def subscribe(
        self, topic: str, agent_id: str
    ) -> AsyncIterator[AgentEvent]:
        """Subscribe to *topic* and return an async iterator of ``AgentEvent``s.

        The Redis subscription is established before returning, so any event
        published after this call is guaranteed to be received.

        Raises ``TransientError`` if the Redis connection is unavailable.
        """
        pubsub = self._redis.pubsub()
        try:
            await pubsub.subscribe(topic)
        except (ConnectionError, RedisError) as exc:
            raise TransientError(
                f"Redis subscribe failed on topic {topic!r}: {exc}"
            ) from exc
        _log.debug("subscribed", topic=topic, agent_id=agent_id)
        return _listen_to_pubsub(pubsub, topic)


async def _listen_to_pubsub(
    pubsub: _AsyncPubSub,
    topic: str,
) -> AsyncGenerator[AgentEvent, None]:
    """Async generator that yields ``AgentEvent``s from a Redis PubSub stream.

    Skips non-message frames (e.g. subscribe confirmations).  Logs a warning
    and skips any message whose payload cannot be deserialised.

    The PubSub subscription is cleaned up in the ``finally`` block so the
    caller can break out of iteration at any time without leaking resources.
    """
    try:
        async for raw in pubsub.listen():
            if raw.get("type") != "message":
                continue
            try:
                event = AgentEvent.model_validate_json(raw["data"])
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "event_deserialize_failed", topic=topic, error=str(exc)
                )
                continue
            _log.debug(
                "event_received",
                topic=topic,
                event_id=event.event_id,
                event_type=event.event_type,
            )
            yield event
    finally:
        await pubsub.unsubscribe(topic)
        await pubsub.aclose()
