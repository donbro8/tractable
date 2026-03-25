"""Integration test for ChangePoller — uses mocked GitProvider and EventBus.

AC-2 (TASK-3.2.2): ChangePoller publishes a RepositoryChangeEvent to the event
bus when it detects a new commit SHA on the second poll call.

asyncio time-mocking is used so the test completes instantly rather than
waiting 90 real seconds.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tractable.reactivity.poller import ChangePoller
from tractable.types.git import CommitEntry


def _make_commit(sha: str) -> CommitEntry:
    return CommitEntry(
        sha=sha,
        message=f"commit {sha}",
        author="ci-bot",
        timestamp=datetime.now(UTC),
        files_changed=["src/app.py"],
    )


@pytest.mark.asyncio
async def test_poller_delivers_change_notification() -> None:
    """ChangePoller publishes a RepositoryChangeEvent when new commits appear.

    Simulates the first poll (records SHA, no event) followed by a second poll
    with a new SHA (emits event).  asyncio.sleep is patched to zero so the loop
    runs without real-time delays.
    """
    repo_id = "owner/my-repo"

    # The provider returns sha-v1 on the first call, sha-v2 on subsequent calls.
    call_count = 0

    async def _get_history(
        rid: str,
        path: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[CommitEntry]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [_make_commit("sha-initial")]
        return [_make_commit("sha-new")]

    provider = MagicMock()
    provider.get_commit_history = _get_history

    published: list[tuple[str, Any]] = []

    async def _publish(topic: str, event: Any) -> None:
        published.append((topic, event))

    event_bus = MagicMock()
    event_bus.publish = _publish

    # In-memory state store
    _store: dict[str, str | None] = {}

    async def _get_sha(rid: str) -> str | None:
        return _store.get(rid)

    async def _set_sha(rid: str, sha: str) -> None:
        _store[rid] = sha

    state_store = MagicMock()
    state_store.get_last_polled_sha = _get_sha
    state_store.set_last_polled_sha = _set_sha

    poller = ChangePoller(
        provider=provider,
        event_bus=event_bus,
        state_store=state_store,
        poll_interval_seconds=0,
    )

    # Patch asyncio.sleep so the loop doesn't actually wait
    sleep_call_count = 0

    async def _fast_sleep(seconds: float) -> None:
        nonlocal sleep_call_count
        sleep_call_count += 1
        # After two iterations (first poll + one change-detecting poll), stop
        if sleep_call_count >= 2:
            raise asyncio.CancelledError

    with patch("tractable.reactivity.poller.asyncio.sleep", side_effect=_fast_sleep):
        poller.start(repo_id)
        # Let the polling loop run until cancelled
        task = poller._tasks[repo_id]
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    # Assert that at least one RepositoryChangeEvent was published
    assert len(published) >= 1, (
        f"Expected at least one change event; got {len(published)}"
    )
    topic, agent_event = published[0]
    assert topic == f"repo.{repo_id}.change"
    assert agent_event.event_type == "repository_change"
    assert agent_event.payload["after_sha"] == "sha-new"
    assert agent_event.payload["before_sha"] == "sha-initial"
