"""ChangePoller — polling fallback for repos without webhook support.

Polls a registered repo's commit history on a configurable interval and
publishes ``RepositoryChangeEvent`` objects (wrapped in ``AgentEvent``) to
the ``EventBus`` when new commits are detected.

Source: realtime-temporal-spec.py §C — ChangePoller Protocol (polling fallback)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog

from tractable.errors import RecoverableError
from tractable.protocols.agent_state_store import AgentStateStore
from tractable.protocols.event_bus import AgentEvent, EventBus
from tractable.protocols.git_provider import GitProvider
from tractable.protocols.reactivity import RepositoryChangeEvent, WebhookCommit
from tractable.types.git import CommitEntry

log = structlog.get_logger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

_POLLER_AGENT_ID = "tractable.poller"


def _commit_to_webhook_commit(entry: CommitEntry) -> WebhookCommit:
    """Convert a ``CommitEntry`` to the ``WebhookCommit`` shape."""
    return WebhookCommit(
        sha=entry.sha,
        message=entry.message,
        author=entry.author,
        timestamp=entry.timestamp,
        # CommitEntry.files_changed doesn't distinguish add/modify/remove;
        # treat all as modified_files to avoid false classifications.
        modified_files=list(entry.files_changed),
    )


def _build_change_event(
    repo_id: str,
    commits: list[CommitEntry],
    before_sha: str | None,
) -> RepositoryChangeEvent:
    """Construct a ``RepositoryChangeEvent`` from polled commit entries."""
    head = commits[0]
    return RepositoryChangeEvent(
        event_id=f"poll-{repo_id}-{head.sha[:12]}-{uuid.uuid4().hex[:8]}",
        repo_name=repo_id,
        provider="polling",
        event_type="push",
        ref="refs/heads/main",
        before_sha=before_sha,
        after_sha=head.sha,
        commits=[_commit_to_webhook_commit(c) for c in commits],
        author=head.author,
        timestamp=head.timestamp,
    )


def _wrap_in_agent_event(change_event: RepositoryChangeEvent) -> AgentEvent:
    """Wrap a ``RepositoryChangeEvent`` inside an ``AgentEvent`` for transport."""
    payload: dict[str, Any] = change_event.model_dump(mode="json")
    return AgentEvent(
        event_id=change_event.event_id,
        timestamp=datetime.now(UTC),
        source_agent_id=_POLLER_AGENT_ID,
        event_type="repository_change",
        payload=payload,
    )


# ── ChangePoller ──────────────────────────────────────────────────────────────


class ChangePoller:
    """Polls registered repos for new commits and publishes change events.

    Usage::

        poller = ChangePoller(provider, event_bus, state_store)
        poller.start("owner/my-repo")
        # … later …
        poller.stop("owner/my-repo")

    Each ``start()`` call spawns an ``asyncio.Task`` that loops indefinitely
    at the configured interval.  ``stop()`` cancels that task.

    Repos with a ``webhook_secret`` configured should NOT be polled — call
    ``start()`` for those repos will raise ``RecoverableError``.
    """

    def __init__(
        self,
        provider: GitProvider,
        event_bus: EventBus,
        state_store: AgentStateStore,
        poll_interval_seconds: int = 60,
    ) -> None:
        self._provider = provider
        self._event_bus = event_bus
        self._state_store = state_store
        self._poll_interval = poll_interval_seconds
        self._tasks: dict[str, asyncio.Task[None]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, repo_id: str, webhook_secret: str | None = None) -> None:
        """Launch the polling loop for *repo_id*.

        Raises ``RecoverableError`` if *webhook_secret* is set — such repos
        should use the ``WebhookReceiver`` path instead.  Calling ``start()``
        twice for the same repo is a no-op (existing task is reused).
        """
        if webhook_secret:
            raise RecoverableError(
                "repo has webhook configured; polling not started"
            )
        if repo_id in self._tasks and not self._tasks[repo_id].done():
            log.debug("poller.already_running", repo=repo_id)
            return
        task = asyncio.get_event_loop().create_task(
            self._poll_loop(repo_id),
            name=f"poller:{repo_id}",
        )
        self._tasks[repo_id] = task
        log.info("poller.started", repo=repo_id, interval=self._poll_interval)

    def stop(self, repo_id: str) -> None:
        """Cancel the polling task for *repo_id*.

        Safe to call even if the repo is not being polled.
        """
        task = self._tasks.pop(repo_id, None)
        if task is not None and not task.done():
            task.cancel()
            log.info("poller.stopped", repo=repo_id)

    # ── Polling loop ──────────────────────────────────────────────────────────

    async def _poll_loop(self, repo_id: str) -> None:
        """Infinite polling loop — runs until the task is cancelled."""
        log.debug("poller.loop.start", repo=repo_id)
        try:
            while True:
                await self._poll_once(repo_id)
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            log.debug("poller.loop.cancelled", repo=repo_id)
            raise

    async def _poll_once(self, repo_id: str) -> None:
        """Execute a single poll cycle for *repo_id*."""
        try:
            commits = list(
                await self._provider.get_commit_history(repo_id, limit=10)
            )
        except Exception as exc:
            log.warning("poller.fetch_failed", repo=repo_id, error=str(exc))
            return

        if not commits:
            log.debug("poller.no_commits", repo=repo_id)
            return

        head_sha = commits[0].sha
        last_sha = await self._state_store.get_last_polled_sha(repo_id)

        if last_sha is None:
            # First poll — record the current head but emit no event.
            await self._state_store.set_last_polled_sha(repo_id, head_sha)
            log.info("poller.first_poll", repo=repo_id, sha=head_sha)
            return

        if head_sha == last_sha:
            log.debug("poller.no_change", repo=repo_id, sha=head_sha)
            return

        # New commits detected.
        change_event = _build_change_event(repo_id, commits, before_sha=last_sha)
        agent_event = _wrap_in_agent_event(change_event)
        topic = f"repo.{repo_id}.change"

        try:
            await self._event_bus.publish(topic, agent_event)
        except Exception as exc:
            log.warning(
                "poller.publish_failed", repo=repo_id, topic=topic, error=str(exc)
            )
            return

        await self._state_store.set_last_polled_sha(repo_id, head_sha)
        log.info(
            "poller.change_published",
            repo=repo_id,
            before=last_sha,
            after=head_sha,
            topic=topic,
        )
