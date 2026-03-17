"""Reactivity Protocols — webhook receiver, change ingestion, polling, lifecycle.

Sources:
- realtime-temporal-spec.py §C — WebhookReceiver, ChangeIngestionPipeline, ChangePoller
- realtime-temporal-spec.py §D — AgentLifecycleManager
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from tractable.types.temporal import ChangeNotification, TemporalMutationResult

# ── Supporting value types ─────────────────────────────────────────────


class WebhookCommit(BaseModel):
    """A single commit as reported in a webhook payload."""

    sha: str
    message: str
    author: str
    timestamp: datetime
    added_files: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    removed_files: list[str] = Field(default_factory=list)

    @property
    def all_affected_files(self) -> list[str]:
        return [*self.added_files, *self.modified_files, *self.removed_files]


class RepositoryChangeEvent(BaseModel):
    """
    Normalised representation of a repository change event, regardless of
    whether it came from a GitHub webhook, GitLab webhook, or polling.
    """

    event_id: str
    repo_name: str
    provider: str
    event_type: Literal[
        "push",
        "pull_request_merged",
        "pull_request_opened",
        "branch_created",
        "branch_deleted",
        "tag_created",
        "force_push",
    ]
    ref: str
    before_sha: str | None = None
    after_sha: str
    commits: list[WebhookCommit] = []
    author: str
    timestamp: datetime
    is_agent_authored: bool = False
    agent_id: str | None = None


class ChangeIngestionResult(BaseModel):
    """Result of processing a single repository change event."""

    event_id: str
    repo_name: str
    commit_sha: str

    files_added: int
    files_modified: int
    files_removed: int
    parse_duration_ms: int

    graph_mutations: TemporalMutationResult

    notifications_sent: list[ChangeNotification] = []
    warnings: list[str] = Field(default_factory=list)


class SyncResult(BaseModel):
    """Result of syncing an agent's local repo clone to a new ref."""

    success: bool
    strategy_used: str  # "pull" | "rebase" | "merge" | "abort"
    conflicts: list[str] = Field(default_factory=list)
    files_updated: int
    new_head_sha: str | None = None


# ── Protocols ──────────────────────────────────────────────────────────


@runtime_checkable
class WebhookReceiver(Protocol):
    """
    Receives raw webhook payloads from git providers and normalises them
    into ``RepositoryChangeEvent`` objects.
    """

    async def handle_webhook(
        self,
        headers: dict[str, str],
        body: bytes,
        provider: str,
    ) -> RepositoryChangeEvent | None:
        """
        Parse and validate a webhook payload.
        Returns ``None`` if the event should be ignored.
        """
        ...

    async def verify_signature(
        self,
        headers: dict[str, str],
        body: bytes,
        secret: str,
    ) -> bool:
        """Verify webhook HMAC signature for security."""
        ...


@runtime_checkable
class ChangeIngestionPipeline(Protocol):
    """
    The core reactive pipeline.

    Receives a ``RepositoryChangeEvent`` and: re-parses changed files,
    computes the graph diff, applies temporal mutations, identifies affected
    agents, and publishes notifications to the event bus.
    """

    async def process_change(
        self,
        event: RepositoryChangeEvent,
    ) -> ChangeIngestionResult:
        """Process a single repository change event end-to-end."""
        ...


@runtime_checkable
class ChangePoller(Protocol):
    """
    Fallback for git providers without webhook support, or as a consistency
    check to catch missed webhooks.
    """

    async def poll_repo(
        self,
        repo_name: str,
    ) -> RepositoryChangeEvent | None:
        """Check one repo for changes. Returns ``None`` if unchanged."""
        ...

    async def poll_all(self) -> Sequence[RepositoryChangeEvent]:
        """Check all registered repos. Returns only those with changes."""
        ...


@runtime_checkable
class AgentLifecycleManager(Protocol):
    """
    Manages agent wake/sleep cycles in response to real-time events.

    Lifecycle: IDLE → debounce → WAKE → assess → act or back to IDLE.
    """

    async def notify_agent(
        self,
        agent_id: str,
        notification: ChangeNotification,
    ) -> None:
        """Deliver a change notification to an agent."""
        ...

    async def wake_agent(self, agent_id: str, reason: str) -> None:
        """Bring an idle/dormant agent to active state."""
        ...

    async def sync_agent_repo(
        self,
        agent_id: str,
        to_ref: str,
    ) -> SyncResult:
        """
        Update the agent's local repo clone to a specific ref.
        Handles rebase/merge of any in-progress agent branch.
        """
        ...

    async def get_agent_last_active(
        self,
        agent_id: str,
    ) -> datetime | None:
        """When was this agent last active? Used for catch-up queries."""
        ...
