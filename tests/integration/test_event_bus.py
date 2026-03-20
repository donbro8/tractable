"""Integration tests for RedisEventBus against a live Redis instance.

Requires the Docker Compose stack running:
    docker compose -f deploy/docker-compose.yml up -d

Marked with ``pytest.mark.integration`` so they are skipped in unit-only runs.

Covers:
- AC-1: publish then subscribe delivers the same event to the handler.
- AC-2: publish with Redis unavailable raises TransientError.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

import pytest
import pytest_asyncio

from tractable.errors import TransientError
from tractable.events.redis_bus import RedisEventBus
from tractable.protocols.event_bus import AgentEvent

_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

pytestmark = pytest.mark.integration


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest_asyncio.fixture()
async def redis_client() -> object:
    """Live async Redis client; skips test if Redis is unreachable."""
    try:
        import redis.asyncio as aioredis  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("redis package not installed")

    client = aioredis.from_url(_REDIS_URL, decode_responses=False)
    try:
        await client.ping()
    except Exception:  # noqa: BLE001
        await client.aclose()
        pytest.skip(f"Redis not reachable at {_REDIS_URL}")

    yield client
    await client.aclose()


def _sample_event(suffix: str = "") -> AgentEvent:
    return AgentEvent(
        event_id=f"test-event-{suffix}",
        timestamp=datetime.now(tz=UTC),
        source_agent_id="test-source",
        target_agent_id="test-target",
        event_type="test.ping",
        payload={"hello": "world"},
    )


# ── AC-1: pub/sub round-trip ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_subscribe_roundtrip(redis_client: object) -> None:
    """AC-1: publish then subscribe delivers exactly the same event."""
    bus = RedisEventBus(redis_client)  # type: ignore[arg-type]
    topic = "tractable-test:pubsub-roundtrip"
    event = _sample_event("roundtrip")

    # Subscribe first so the Redis channel is registered before publishing.
    sub_iter = await bus.subscribe(topic, "test-agent")

    # Give Redis a moment to confirm the subscription.
    await asyncio.sleep(0.05)

    # Publish the event.
    await bus.publish(topic, event)

    # Receive the first event from the subscription with a timeout.
    received: AgentEvent | None = None
    try:
        received = await asyncio.wait_for(sub_iter.__anext__(), timeout=5.0)
    except TimeoutError:
        pytest.fail("Timed out waiting for published event on subscription")

    assert received is not None
    assert received.event_id == event.event_id
    assert received.event_type == event.event_type
    assert received.source_agent_id == event.source_agent_id
    assert received.payload == event.payload


# ── AC-2: TransientError when Redis is unavailable ────────────────────────


@pytest.mark.asyncio
async def test_publish_unavailable_redis_raises_transient_error() -> None:
    """AC-2: publish() raises TransientError when Redis connection fails."""
    try:
        import redis.asyncio as aioredis  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("redis package not installed")

    # Point at a port where nothing is listening.
    bad_client = aioredis.from_url(
        "redis://localhost:19999", decode_responses=False, socket_connect_timeout=1
    )
    bus = RedisEventBus(bad_client)  # type: ignore[arg-type]
    event = _sample_event("bad")

    with pytest.raises(TransientError):
        await bus.publish("some-topic", event)

    await bad_client.aclose()
