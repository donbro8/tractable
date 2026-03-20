"""Tractable registry service entry point.

Exposes a uvicorn-compatible ``app`` instance.  Start with::

    uvicorn tractable.registry.main:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /health          — liveness probe; returns ``{"status": "ok"}``
POST /webhooks/github — GitHub push webhook receiver (TASK-2.6.1)

On startup the service initialises a PostgreSQL-backed ``AgentStateStore``
and wires it into a ``_WakeOnWebhookPipeline`` that, for every incoming
webhook, updates ``last_active`` for every agent registered against the
changed repository.  This satisfies the EC3 integration-test requirement
(``AgentContext.last_active`` updated within 35 seconds of a webhook POST).
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import structlog
from fastapi import FastAPI

from tractable.protocols.agent_state_store import AgentStateStore
from tractable.protocols.reactivity import (
    ChangeIngestionResult,
    RepositoryChangeEvent,
)
from tractable.reactivity.webhook_receiver import create_webhook_router
from tractable.types.temporal import TemporalMutationResult

_log = structlog.get_logger()


# ── Minimal ChangeIngestionPipeline ──────────────────────────────────────────


class _WakeOnWebhookPipeline:
    """ChangeIngestionPipeline that wakes matching agents when a webhook arrives.

    Does not perform graph mutations or file parsing — it reads the agent
    registry from PostgreSQL and sets ``last_active`` / ``status`` for every
    agent whose ``repo`` field matches the incoming event's ``repo_name``.
    """

    def __init__(self, state_store: AgentStateStore) -> None:
        self._state_store = state_store

    async def process_change(
        self, event: RepositoryChangeEvent
    ) -> ChangeIngestionResult:
        noop_mutations = TemporalMutationResult(
            entities_created=0,
            entities_updated=0,
            entities_deleted=0,
            edges_created=0,
            edges_deleted=0,
            timestamp=datetime.now(tz=UTC),
        )
        try:
            agents = await self._state_store.list_agents()
            for agent in agents:
                if agent.repo == event.repo_name:
                    agent.user_overrides["status"] = "working"
                    agent.user_overrides["last_active"] = datetime.now(
                        tz=UTC
                    ).isoformat()
                    await self._state_store.save_agent_context(
                        agent.agent_id, agent
                    )
                    _log.info(
                        "agent_woke",
                        agent_id=agent.agent_id,
                        task_id=None,
                        repo=event.repo_name,
                        level="info",
                        reason="webhook",
                        event="agent_woke",
                    )
        except Exception:
            _log.warning(
                "wake_on_webhook_failed",
                repo=event.repo_name,
                exc_info=True,
            )
        return ChangeIngestionResult(
            event_id=event.event_id,
            repo_name=event.repo_name,
            commit_sha=event.after_sha,
            files_added=0,
            files_modified=0,
            files_removed=0,
            parse_duration_ms=0,
            graph_mutations=noop_mutations,
        )


# ── Application factory ───────────────────────────────────────────────────────


def _build_app() -> FastAPI:
    """Build and return the FastAPI application.

    Uses a ``_PipelineProxy`` to defer to the real pipeline once it has been
    initialised during the lifespan startup handler.  This allows the webhook
    router to be added at module-import time while the database connection is
    deferred until the ASGI server starts.
    """
    _holder: list[_WakeOnWebhookPipeline] = []

    class _PipelineProxy:
        """Forwards ``process_change`` to the real pipeline once initialised."""

        async def process_change(
            self, event: RepositoryChangeEvent
        ) -> ChangeIngestionResult:
            if _holder:
                return await _holder[0].process_change(event)
            # Fallback: return a no-op result if pipeline not yet ready.
            return ChangeIngestionResult(
                event_id=event.event_id,
                repo_name=event.repo_name,
                commit_sha=event.after_sha,
                files_added=0,
                files_modified=0,
                files_removed=0,
                parse_duration_ms=0,
                graph_mutations=TemporalMutationResult(
                    entities_created=0,
                    entities_updated=0,
                    entities_deleted=0,
                    edges_created=0,
                    edges_deleted=0,
                    timestamp=datetime.now(tz=UTC),
                ),
            )

    proxy = _PipelineProxy()
    webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
    router = create_webhook_router(proxy, webhook_secret)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        try:
            from tractable.state.store import PostgreSQLAgentStateStore

            store = PostgreSQLAgentStateStore.from_env()
            _holder.append(_WakeOnWebhookPipeline(store))
            _log.info(
                "tractable_service_started",
                task_id=None,
                agent_id="system",
                repo="tractable",
                level="info",
            )
        except Exception:
            _log.warning(
                "state_store_init_failed",
                exc_info=True,
            )
        yield
        _holder.clear()

    application = FastAPI(title="Tractable Registry", lifespan=lifespan)

    @application.get("/health")
    async def health() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        """Liveness probe — returns HTTP 200 with ``{"status": "ok"}``."""
        return {"status": "ok"}

    application.include_router(router)
    return application


app = _build_app()
