"""REVIEWING node — third node in the agent workflow.

TASK-2.3.1: Saves a checkpoint with phase=REVIEWING, enforces GovernancePolicy
gates (mocked in this milestone), and routes either back to EXECUTING on
failure or forward to COORDINATING on success.

Routing constants
-----------------
RETRY_EDGE
    Returned by the routing function when governance gates fail; LangGraph
    follows this edge back to the EXECUTING node.
DONE_EDGE
    Returned when gates pass; LangGraph follows this edge to COORDINATING.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from tractable.types.agent import AgentCheckpoint
from tractable.types.enums import TaskPhase

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from tractable.agent.state import AgentWorkflowState
    from tractable.protocols.agent_state_store import AgentStateStore
    from tractable.protocols.tool import Tool

_log = structlog.get_logger()

# Edge labels used by the conditional router in workflow.py
RETRY_EDGE = "retry"
DONE_EDGE = "done"

# ── Public factory ─────────────────────────────────────────────────────────


def make_reviewing_node(
    tools: dict[str, Tool],
    state_store: AgentStateStore,
) -> Callable[[AgentWorkflowState], Coroutine[Any, Any, dict[str, Any]]]:
    """Return an async REVIEWING node with injected dependencies.

    Parameters
    ----------
    tools:
        Tool name → Tool mapping; injected at workflow construction time.
    state_store:
        Used to persist the REVIEWING-phase checkpoint.
    """

    async def reviewing_node(state: AgentWorkflowState) -> dict[str, Any]:
        agent_id = state["agent_id"]
        task_id = state["task_id"]

        checkpoint = AgentCheckpoint(
            task_id=task_id,
            phase=TaskPhase.REVIEWING,
            progress_summary="Entering REVIEWING node",
            files_modified=list(state["files_changed"]),
            pending_actions=[],
            conversation_summary="",
            token_usage=state["token_count"],
            created_at=datetime.now(tz=UTC),
        )
        await state_store.save_checkpoint(agent_id, task_id, checkpoint)
        _log.info(
            "checkpoint_saved",
            agent_id=agent_id,
            task_id=task_id,
            phase=TaskPhase.REVIEWING,
        )

        # Enforce governance gates via mocked tools.
        # In Milestone 2.4 the real test_runner and linter tools replace mocks.
        gate_error: str | None = None

        if "test_runner" in tools:
            result = await tools["test_runner"].invoke({})
            if not result.success:
                gate_error = result.error or "test_runner gate failed"

        if gate_error is None and "linter" in tools:
            result = await tools["linter"].invoke({})
            if not result.success:
                gate_error = result.error or "linter gate failed"

        if gate_error is not None:
            _log.warning(
                "governance_gate_failed",
                agent_id=agent_id,
                task_id=task_id,
                reason=gate_error,
            )
            return {"error": gate_error}

        return {
            "phase": TaskPhase.COORDINATING,
            "error": None,
        }

    return reviewing_node


# ── Routing function ──────────────────────────────────────────────────────


def reviewing_router(state: AgentWorkflowState) -> str:
    """Conditional edge: DONE_EDGE when gates pass, RETRY_EDGE on failure."""
    return DONE_EDGE if state["error"] is None else RETRY_EDGE
