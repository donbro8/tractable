"""NotificationRouter — routes change ingestion results to affected agents.

TASK-2.6.2 — Implement Redis EventBus and NotificationRouter.

``NotificationRouter.route()`` determines which agents should be notified
about a repository change, filters out agents below their wake threshold,
and publishes a ``ChangeNotification`` (wrapped in an ``AgentEvent``) to
each qualifying agent's notification topic.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog

from tractable.protocols.agent_state_store import AgentStateStore
from tractable.protocols.code_graph import TemporalCodeGraph
from tractable.protocols.event_bus import AgentEvent, EventBus
from tractable.protocols.reactivity import ChangeIngestionResult
from tractable.types.enums import ChangeRelevance
from tractable.types.temporal import ChangeNotification

_log = structlog.get_logger()


class NotificationRouter:
    """Routes a ``ChangeIngestionResult`` to all affected agents via the event bus.

    Parameters
    ----------
    event_bus:
        The event bus used to publish ``ChangeNotification`` payloads.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def route(
        self,
        ingestion_result: ChangeIngestionResult,
        graph: TemporalCodeGraph,
        state_store: AgentStateStore,
    ) -> list[ChangeNotification]:
        """Compute and publish notifications for *ingestion_result*.

        Steps:
        1. Query the graph for agents registered against the changed repo
           (``ChangeRelevance.DIRECT``).
        2. Query for cross-repo agents via dependency/consumer graph edges
           (``ChangeRelevance.DEPENDENCY`` / ``ChangeRelevance.CONSUMER``).
        3. Load each candidate agent's ``AgentReactivityConfig`` from the
           state store; filter out agents whose wake threshold is not met.
        4. Publish a ``ChangeNotification`` to
           ``agent.{agent_id}.notifications`` for each qualifying agent.
        5. Return the list of sent notifications.
        """
        repo = ingestion_result.repo_name
        commit_sha = ingestion_result.commit_sha

        # ── Step 1: direct agents ──────────────────────────────────────────
        direct_rows = await graph.query_current(
            "MATCH (a:Agent {repo: $repo}) RETURN a.agent_id AS agent_id",
            {"repo": repo},
        )
        seen: set[str] = set()
        candidates: list[tuple[str, ChangeRelevance]] = []

        for row in direct_rows:
            agent_id = str(row["agent_id"])
            if agent_id not in seen:
                seen.add(agent_id)
                candidates.append((agent_id, ChangeRelevance.DIRECT))

        # ── Step 2: cross-repo agents ──────────────────────────────────────
        # Agents in repos that DEPEND ON the changed repo → DEPENDENCY.
        dep_rows = await graph.query_current(
            "MATCH (a:Agent)-[:DEPENDS_ON]->(r:Repo {name: $repo})"
            " RETURN a.agent_id AS agent_id",
            {"repo": repo},
        )
        for row in dep_rows:
            agent_id = str(row["agent_id"])
            if agent_id not in seen:
                seen.add(agent_id)
                candidates.append((agent_id, ChangeRelevance.DEPENDENCY))

        # Agents in repos CONSUMED BY the changed repo → CONSUMER.
        con_rows = await graph.query_current(
            "MATCH (a:Agent)<-[:DEPENDS_ON]-(r:Repo {name: $repo})"
            " RETURN a.agent_id AS agent_id",
            {"repo": repo},
        )
        for row in con_rows:
            agent_id = str(row["agent_id"])
            if agent_id not in seen:
                seen.add(agent_id)
                candidates.append((agent_id, ChangeRelevance.CONSUMER))

        # ── Step 3: apply reactivity config filter ─────────────────────────
        notifications: list[ChangeNotification] = []

        for agent_id, relevance in candidates:
            context = await state_store.get_agent_context(agent_id)
            cfg = context.reactivity_config

            wake = (
                (relevance == ChangeRelevance.DIRECT and cfg.wake_on_direct_change)
                or (
                    relevance == ChangeRelevance.DEPENDENCY
                    and cfg.wake_on_dependency_change
                )
                or (
                    relevance == ChangeRelevance.CONSUMER
                    and cfg.wake_on_consumer_change
                )
                or (
                    relevance == ChangeRelevance.TRANSITIVE
                    and cfg.wake_on_transitive_change
                )
            )
            if not wake:
                _log.debug(
                    "agent_skipped_by_reactivity_config",
                    agent_id=agent_id,
                    relevance=relevance,
                )
                continue

            # ── Step 4: construct and publish notification ─────────────────
            notification = ChangeNotification(
                target_agent_id=agent_id,
                repo_name=repo,
                relevance=relevance,
                change_summary=(
                    f"{ingestion_result.files_added} added, "
                    f"{ingestion_result.files_modified} modified, "
                    f"{ingestion_result.files_removed} removed in {repo}"
                ),
                affected_entity_ids=[],
                cross_repo_edges_affected=[],
                commit_sha=commit_sha,
                requires_action=True,
            )

            topic = f"agent.{agent_id}.notifications"
            agent_event = AgentEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(tz=UTC),
                source_agent_id="system",
                target_agent_id=agent_id,
                event_type="change_notification",
                payload=notification.model_dump(),
            )
            await self._event_bus.publish(topic, agent_event)

            _log.info(
                "change_notification_sent",
                agent_id=agent_id,
                repo=repo,
                relevance=relevance,
                commit_sha=commit_sha,
            )
            notifications.append(notification)

        return notifications
