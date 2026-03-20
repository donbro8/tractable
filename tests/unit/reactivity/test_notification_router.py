"""Unit tests for NotificationRouter (TASK-2.6.2).

Covers:
- AC-3: Two DIRECT agents both receive ChangeNotification.
- AC-4: Agent with wake_on_direct_change=False is filtered out.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from tractable.protocols.reactivity import ChangeIngestionResult
from tractable.reactivity.notification_router import NotificationRouter
from tractable.types.agent import AgentContext, AgentReactivityConfig
from tractable.types.enums import ChangeRelevance
from tractable.types.temporal import ChangeNotification, TemporalMutationResult

# ── Helpers ────────────────────────────────────────────────────────────────


def _noop_mutation_result() -> TemporalMutationResult:
    return TemporalMutationResult(
        entities_created=0,
        entities_updated=0,
        entities_deleted=0,
        edges_created=0,
        edges_deleted=0,
        timestamp=datetime.now(tz=UTC),
    )


def _make_ingestion_result(repo: str = "my-api") -> ChangeIngestionResult:
    return ChangeIngestionResult(
        event_id="ev-001",
        repo_name=repo,
        commit_sha="abc123",
        files_added=1,
        files_modified=2,
        files_removed=0,
        parse_duration_ms=50,
        graph_mutations=_noop_mutation_result(),
    )


def _make_agent_context(
    agent_id: str,
    repo: str = "my-api",
    wake_on_direct: bool = True,
) -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        repo=repo,
        base_template="base",
        system_prompt="prompt",
        repo_architectural_summary="summary",
        reactivity_config=AgentReactivityConfig(
            wake_on_direct_change=wake_on_direct,
        ),
    )


def _make_graph(agent_rows: list[dict[str, object]]) -> AsyncMock:
    """Return a mocked TemporalCodeGraph.

    The first ``query_current`` call returns *agent_rows* (direct agents);
    subsequent calls (cross-repo queries) return empty lists.
    """
    graph = AsyncMock()

    call_count: list[int] = [0]

    async def _query_current(
        cypher: str, params: dict[str, object] | None = None
    ) -> list[dict[str, object]]:
        call_count[0] += 1
        if call_count[0] == 1:
            return agent_rows
        return []

    graph.query_current = _query_current
    return graph


# ── AC-3: Two direct agents both receive notifications ─────────────────────


@pytest.mark.asyncio
async def test_two_direct_agents_both_notified() -> None:
    """AC-3: Two agents registered against 'my-api' both receive DIRECT notifications."""
    event_bus = AsyncMock()
    event_bus.publish = AsyncMock()

    graph = _make_graph([{"agent_id": "agent-1"}, {"agent_id": "agent-2"}])

    state_store = AsyncMock()
    state_store.get_agent_context = AsyncMock(side_effect=lambda aid: _make_agent_context(aid))

    router = NotificationRouter(event_bus)
    result = await router.route(_make_ingestion_result(), graph, state_store)

    assert len(result) == 2
    agent_ids = {n.target_agent_id for n in result}
    assert agent_ids == {"agent-1", "agent-2"}

    for notification in result:
        assert notification.relevance == ChangeRelevance.DIRECT
        assert notification.repo_name == "my-api"
        assert notification.commit_sha == "abc123"

    assert event_bus.publish.call_count == 2


# ── AC-4: Agent with wake_on_direct_change=False is filtered ──────────────


@pytest.mark.asyncio
async def test_agent_with_direct_wake_disabled_not_notified() -> None:
    """AC-4: Agent with wake_on_direct_change=False receives no notification."""
    event_bus = AsyncMock()
    event_bus.publish = AsyncMock()

    graph = _make_graph([{"agent_id": "agent-wake"}, {"agent_id": "agent-sleep"}])

    def _context(agent_id: str) -> AgentContext:
        return _make_agent_context(agent_id, wake_on_direct=agent_id != "agent-sleep")

    state_store = AsyncMock()
    state_store.get_agent_context = AsyncMock(side_effect=_context)

    router = NotificationRouter(event_bus)
    result = await router.route(_make_ingestion_result(), graph, state_store)

    # Only the wake-enabled agent should receive a notification.
    assert len(result) == 1
    assert result[0].target_agent_id == "agent-wake"
    assert result[0].relevance == ChangeRelevance.DIRECT

    # Publish called once (for agent-wake only).
    event_bus.publish.assert_called_once()
    topic_arg: str = event_bus.publish.call_args.args[0]
    assert "agent-wake" in topic_arg


# ── Additional: empty agent list produces no notifications ─────────────────


@pytest.mark.asyncio
async def test_no_agents_returns_empty_list() -> None:
    """No agents registered against repo → empty notification list."""
    event_bus = AsyncMock()
    graph = _make_graph([])
    state_store = AsyncMock()

    router = NotificationRouter(event_bus)
    result = await router.route(_make_ingestion_result(), graph, state_store)

    assert result == []
    event_bus.publish.assert_not_called()


# ── Additional: returned list contains ChangeNotification instances ────────


@pytest.mark.asyncio
async def test_returned_objects_are_change_notifications() -> None:
    """route() returns a list of ChangeNotification Pydantic models."""
    event_bus = AsyncMock()
    graph = _make_graph([{"agent_id": "agent-x"}])
    state_store = AsyncMock()
    state_store.get_agent_context = AsyncMock(side_effect=lambda aid: _make_agent_context(aid))

    router = NotificationRouter(event_bus)
    result = await router.route(_make_ingestion_result(), graph, state_store)

    assert len(result) == 1
    assert isinstance(result[0], ChangeNotification)
