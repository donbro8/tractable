"""COORDINATING node — fourth and final node in the agent workflow.

TASK-2.3.1: Saves a checkpoint with phase=COORDINATING, then creates a branch,
commits changes, and opens a PR via the git_ops tool (mocked in this
milestone).  Real tool implementations are substituted in Milestone 2.4.
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

# ── Public factory ─────────────────────────────────────────────────────────


def make_coordinating_node(
    tools: dict[str, Tool],
    state_store: AgentStateStore,
) -> Callable[[AgentWorkflowState], Coroutine[Any, Any, dict[str, Any]]]:
    """Return an async COORDINATING node with injected dependencies.

    Parameters
    ----------
    tools:
        Tool name → Tool mapping; injected at workflow construction time.
    state_store:
        Used to persist the COORDINATING-phase checkpoint.
    """

    async def coordinating_node(state: AgentWorkflowState) -> dict[str, Any]:
        agent_id = state["agent_id"]
        task_id = state["task_id"]

        checkpoint = AgentCheckpoint(
            task_id=task_id,
            phase=TaskPhase.COORDINATING,
            progress_summary="Entering COORDINATING node",
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
            phase=TaskPhase.COORDINATING,
        )

        # Create branch, commit, and open PR via git_ops tool (mocked).
        pr_url: str | None = None
        if "git_ops" in tools:
            result = await tools["git_ops"].invoke(
                {
                    "action": "create_pr",
                    "task_id": task_id,
                    "files_changed": state["files_changed"],
                }
            )
            if result.success and result.output:
                pr_url = str(result.output)

        return {
            "phase": TaskPhase.COMPLETED,
            "pr_url": pr_url,
            "error": None,
        }

    return coordinating_node
