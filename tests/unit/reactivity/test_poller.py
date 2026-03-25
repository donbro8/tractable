"""Unit tests for ChangePoller.

Covers all acceptance criteria from TASK-3.2.2:
- First poll stores last_polled_sha and emits no event
- Second poll with same SHA emits no event
- Second poll with new SHA emits exactly one event
- stop() cancels the polling task within 1 second
- start() for a repo with webhook_secret raises RecoverableError
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tractable.errors import RecoverableError
from tractable.reactivity.poller import ChangePoller
from tractable.types.git import CommitEntry

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_commit(sha: str, message: str = "test commit") -> CommitEntry:
    return CommitEntry(
        sha=sha,
        message=message,
        author="bot",
        timestamp=datetime.now(UTC),
        files_changed=["src/main.py"],
    )


def _make_poller(
    commits: list[CommitEntry] | None = None,
    last_sha: str | None = None,
    poll_interval: int = 0,
) -> tuple[ChangePoller, MagicMock, AsyncMock, AsyncMock]:
    """Build a ChangePoller with fully mocked dependencies."""
    provider = MagicMock()
    provider.get_commit_history = AsyncMock(return_value=commits or [_make_commit("abc123")])

    published_events: list[tuple[str, Any]] = []

    async def _publish(topic: str, event: object) -> None:
        published_events.append((topic, event))

    event_bus = MagicMock()
    event_bus.publish = _publish

    state_store = MagicMock()
    state_store.get_last_polled_sha = AsyncMock(return_value=last_sha)
    state_store.set_last_polled_sha = AsyncMock()

    poller = ChangePoller(
        provider=provider,
        event_bus=event_bus,
        state_store=state_store,
        poll_interval_seconds=poll_interval,
    )
    return poller, provider, state_store, published_events  # type: ignore[return-value]


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_poll_stores_sha_emits_no_event() -> None:
    """First poll: stores last_polled_sha, publishes nothing."""
    poller, provider, state_store, published = _make_poller(
        commits=[_make_commit("sha-v1")],
        last_sha=None,  # No prior SHA
    )
    await poller._poll_once("owner/repo")

    state_store.set_last_polled_sha.assert_awaited_once_with("owner/repo", "sha-v1")
    assert published == []


@pytest.mark.asyncio
async def test_second_poll_same_sha_emits_no_event() -> None:
    """Second poll with the same SHA: no event published."""
    poller, provider, state_store, published = _make_poller(
        commits=[_make_commit("sha-v1")],
        last_sha="sha-v1",  # Same as current head
    )
    await poller._poll_once("owner/repo")

    state_store.set_last_polled_sha.assert_not_awaited()
    assert published == []


@pytest.mark.asyncio
async def test_second_poll_new_sha_emits_exactly_one_event() -> None:
    """Second poll with a new SHA: exactly one event published and SHA updated."""
    poller, provider, state_store, published = _make_poller(
        commits=[_make_commit("sha-v2")],
        last_sha="sha-v1",
    )
    await poller._poll_once("owner/repo")

    assert len(published) == 1
    topic, agent_event = published[0]
    assert topic == "repo.owner/repo.change"
    assert agent_event.event_type == "repository_change"
    # Payload contains the RepositoryChangeEvent fields
    assert agent_event.payload["after_sha"] == "sha-v2"
    assert agent_event.payload["before_sha"] == "sha-v1"

    state_store.set_last_polled_sha.assert_awaited_once_with("owner/repo", "sha-v2")


@pytest.mark.asyncio
async def test_stop_cancels_task_within_one_second() -> None:
    """stop() removes and cancels the polling task."""
    poller, _, _, _ = _make_poller()

    # Inject a long-running task into the poller directly so we can test stop()
    # without racing against the polling loop's sleep duration.
    async def _forever() -> None:
        await asyncio.sleep(9999)

    loop = asyncio.get_event_loop()
    task: asyncio.Task[None] = loop.create_task(_forever())
    poller._tasks["owner/repo"] = task

    assert not task.done()
    poller.stop("owner/repo")

    # Wait up to 1 second for the cancellation to propagate
    with contextlib.suppress(asyncio.CancelledError, TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=1.0)

    assert task.cancelled() or task.done()
    assert "owner/repo" not in poller._tasks


@pytest.mark.asyncio
async def test_start_with_webhook_secret_raises_recoverable_error() -> None:
    """start() for a repo with webhook_secret raises RecoverableError."""
    poller, _, _, _ = _make_poller()

    with pytest.raises(RecoverableError, match="repo has webhook configured"):
        poller.start("owner/repo", webhook_secret="s3cr3t")

    # No task should have been created
    assert "owner/repo" not in poller._tasks


@pytest.mark.asyncio
async def test_start_no_webhook_secret_succeeds() -> None:
    """start() with no webhook_secret creates a polling task."""
    poller, _, _, _ = _make_poller(poll_interval=9999)

    poller.start("owner/repo")
    # Yield briefly; the loop will block on asyncio.sleep(9999)
    await asyncio.sleep(0)

    assert "owner/repo" in poller._tasks
    poller.stop("owner/repo")


@pytest.mark.asyncio
async def test_poll_loop_ignores_provider_exception() -> None:
    """Provider failure in _poll_once logs a warning and does not raise."""
    poller, provider, _, published = _make_poller()
    provider.get_commit_history = AsyncMock(side_effect=RuntimeError("network error"))

    # Should not raise
    await poller._poll_once("owner/repo")
    assert published == []
