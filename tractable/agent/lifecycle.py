"""AgentLifecycleManager implementation.

TASK-2.3.3: Manages agent wake/sleep cycles in response to real-time events.

Methods implemented
-------------------
- notify_agent   — debounced wake scheduling when a ChangeNotification arrives
- wake_agent     — DORMANT/IDLE → WORKING state transition in the state store
- sync_agent_repo — git pull --rebase; raises RecoverableError on conflict
- get_agent_last_active — reads last_active from AgentContext.user_overrides

Debounce mechanism
------------------
A per-agent ``asyncio.Handle`` is stored in ``_timers``.  Each
``notify_agent`` call cancels any existing handle and schedules a new one
at ``debounce_seconds`` (default 30 s).  When the timer fires it calls
``wake_agent`` via ``loop.create_task``.

State storage
-------------
``AgentContext`` does not expose ``status`` or ``last_active`` fields
directly; they are stored in ``AgentContext.user_overrides`` under the
keys ``"status"`` and ``"last_active"`` (ISO-8601 string).
"""

from __future__ import annotations

import asyncio
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import structlog

from tractable.agent.context import assemble_context
from tractable.errors import RecoverableError
from tractable.protocols.agent_state_store import AgentStateStore
from tractable.protocols.code_graph import TemporalCodeGraph
from tractable.types.agent import AgentStatus
from tractable.types.config import RepositoryRegistration
from tractable.types.enums import ChangeRelevance
from tractable.types.temporal import AgentReactivityConfig, ChangeNotification, SyncResult

_log = structlog.get_logger()


class LifecycleManager:
    """Concrete ``AgentLifecycleManager`` implementation.

    Parameters
    ----------
    state_store:
        Persistent store for agent contexts.
    graph:
        Temporal code graph; passed to ``assemble_context`` on wake.
    registrations:
        Mapping of ``agent_id → RepositoryRegistration``.  Used to refresh
        the system prompt on wake.  If an agent has no entry, context
        refresh is skipped gracefully.
    working_dirs:
        Mapping of ``agent_id → Path`` to the agent's local repo clone.
        Required for ``sync_agent_repo``.
    reactivity_configs:
        Per-agent reactivity configuration.  Falls back to
        ``AgentReactivityConfig()`` defaults when absent.
    """

    def __init__(
        self,
        state_store: AgentStateStore,
        graph: TemporalCodeGraph,
        registrations: dict[str, RepositoryRegistration],
        working_dirs: dict[str, Path],
        reactivity_configs: dict[str, AgentReactivityConfig] | None = None,
    ) -> None:
        self._state_store = state_store
        self._graph = graph
        self._registrations = registrations
        self._working_dirs = working_dirs
        self._reactivity_configs = reactivity_configs or {}
        self._timers: dict[str, asyncio.Handle] = {}

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_reactivity_config(self, agent_id: str) -> AgentReactivityConfig:
        return self._reactivity_configs.get(agent_id, AgentReactivityConfig())

    def _should_wake(
        self, config: AgentReactivityConfig, notification: ChangeNotification
    ) -> bool:
        relevance = notification.relevance
        if relevance == ChangeRelevance.DIRECT:
            return config.wake_on_direct_change
        if relevance == ChangeRelevance.DEPENDENCY:
            return config.wake_on_dependency_change
        if relevance == ChangeRelevance.CONSUMER:
            return config.wake_on_consumer_change
        if relevance == ChangeRelevance.TRANSITIVE:
            return config.wake_on_transitive_change
        return False

    # ── Protocol methods ───────────────────────────────────────────────────

    async def notify_agent(
        self, agent_id: str, notification: ChangeNotification
    ) -> None:
        """Deliver a change notification and (re)start the debounce timer."""
        config = self._get_reactivity_config(agent_id)

        _log.info(
            "notification_received",
            agent_id=agent_id,
            relevance=notification.relevance,
            requires_action=notification.requires_action,
        )

        if not (notification.requires_action and self._should_wake(config, notification)):
            return

        # Cancel any existing debounce timer for this agent.
        existing = self._timers.get(agent_id)
        if existing is not None:
            existing.cancel()

        # Schedule a new debounce timer.
        loop = asyncio.get_event_loop()
        reason = str(notification.relevance)

        def _on_timer() -> None:
            loop.create_task(self.wake_agent(agent_id, reason))

        handle = loop.call_later(config.debounce_seconds, _on_timer)
        self._timers[agent_id] = handle

    async def wake_agent(self, agent_id: str, reason: str) -> None:
        """Transition the agent from DORMANT/IDLE to WORKING."""
        ctx = await self._state_store.get_agent_context(agent_id)
        current_status = ctx.user_overrides.get("status")

        if current_status == AgentStatus.WORKING:
            _log.warning("agent_already_working", agent_id=agent_id, reason=reason)
            return

        ctx.user_overrides["status"] = AgentStatus.WORKING
        ctx.user_overrides["last_active"] = datetime.now(tz=UTC).isoformat()

        # Refresh system prompt from the three-layer context assembly.
        registration = self._registrations.get(agent_id)
        if registration is not None:
            try:
                ctx.system_prompt = await assemble_context(
                    agent_id=agent_id,
                    state_store=self._state_store,
                    graph=self._graph,
                    registration=registration,
                )
            except Exception:
                _log.warning(
                    "context_refresh_failed",
                    agent_id=agent_id,
                    exc_info=True,
                )

        await self._state_store.save_agent_context(agent_id, ctx)
        _log.info("agent_woke", agent_id=agent_id, reason=reason)

    async def sync_agent_repo(self, agent_id: str, to_ref: str) -> SyncResult:
        """Run ``git pull --rebase origin {to_ref}`` in the agent's working dir."""
        working_dir = self._working_dirs.get(agent_id)
        if working_dir is None:
            raise RecoverableError(f"No working directory registered for agent {agent_id!r}")

        result = subprocess.run(
            ["git", "pull", "--rebase", "origin", to_ref],
            cwd=working_dir,
            capture_output=True,
            text=True,
        )

        strategy_used = "rebase"

        if result.returncode != 0:
            combined = result.stdout + result.stderr
            if "CONFLICT" in combined:
                raise RecoverableError(
                    f"git pull --rebase conflict in {working_dir}: {combined[:500]}"
                )
            raise RecoverableError(
                f"git pull --rebase failed (exit {result.returncode}): {combined[:500]}"
            )

        _log.info(
            "repo_synced",
            agent_id=agent_id,
            to_ref=to_ref,
            strategy_used=strategy_used,
        )

        return SyncResult(
            success=True,
            strategy_used=strategy_used,
            files_updated=0,
        )

    async def get_agent_last_active(self, agent_id: str) -> datetime | None:
        """Return when this agent was last active, or ``None`` if never."""
        ctx = await self._state_store.get_agent_context(agent_id)
        raw = ctx.user_overrides.get("last_active")
        if raw is None:
            return None
        return datetime.fromisoformat(str(raw))
