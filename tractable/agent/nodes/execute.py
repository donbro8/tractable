"""EXECUTING node — second node in the agent workflow.

TASK-2.3.1: Saves a checkpoint with phase=EXECUTING, then invokes tools
(mocked in this milestone) to implement each plan step.  Real tool
implementations are substituted in Milestone 2.4.

TASK-3.2.3: Enforces ``max_files_per_change`` governance limit at node entry.
Re-plan loop protection via ``replan_count``; escalates to FAILED when the
replan ceiling is reached.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from tractable.errors import GovernanceError
from tractable.types.agent import AgentCheckpoint
from tractable.types.enums import TaskPhase

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from tractable.agent.state import AgentWorkflowState
    from tractable.protocols.agent_state_store import AgentStateStore
    from tractable.protocols.tool import Tool
    from tractable.types.config import GovernancePolicy

_log = structlog.get_logger()

# ── Public factory ─────────────────────────────────────────────────────────


def make_executing_node(
    tools: dict[str, Tool],
    state_store: AgentStateStore,
    governance: GovernancePolicy | None = None,
) -> Callable[[AgentWorkflowState], Coroutine[Any, Any, dict[str, Any]]]:
    """Return an async EXECUTING node with injected dependencies.

    Parameters
    ----------
    tools:
        Tool name → Tool mapping; injected at workflow construction time.
    state_store:
        Used to persist the EXECUTING-phase checkpoint.
    governance:
        When supplied, ``max_files_per_change`` is checked at node entry and
        re-plan loop protection is enforced via ``replan_count``.
    """

    async def executing_node(state: AgentWorkflowState) -> dict[str, Any]:
        agent_id = state["agent_id"]
        task_id = state["task_id"]

        # ── Governance: max_files_per_change ───────────────────────────────
        if governance is not None:
            files_count = len(state["files_changed"])
            replan_count: int = state.get("replan_count", 0)  # type: ignore[call-overload]

            if files_count > governance.max_files_per_change:
                if replan_count >= governance.max_retries_on_failure:
                    _log.warning(
                        "governance_replan_limit_reached",
                        agent_id=agent_id,
                        task_id=task_id,
                        repo="",
                        replan_count=replan_count,
                        limit=governance.max_retries_on_failure,
                    )
                    return {
                        "phase": TaskPhase.FAILED,
                        "error": "governance replan limit reached",
                    }

                _log.warning(
                    "governance_violation",
                    agent_id=agent_id,
                    task_id=task_id,
                    repo="",
                    type="max_files_per_change",
                    files_changed=files_count,
                    limit=governance.max_files_per_change,
                )
                return {
                    "phase": TaskPhase.PLANNING,
                    "replan_count": replan_count + 1,
                    "error": (
                        "max_files_per_change exceeded; re-plan with narrower scope"
                    ),
                }

        checkpoint = AgentCheckpoint(
            task_id=task_id,
            phase=TaskPhase.EXECUTING,
            progress_summary="Entering EXECUTING node",
            files_modified=list(state["files_changed"]),
            pending_actions=list(state["plan"]),
            conversation_summary="",
            token_usage=state["token_count"],
            created_at=datetime.now(tz=UTC),
        )
        await state_store.save_checkpoint(agent_id, task_id, checkpoint)
        _log.info(
            "checkpoint_saved",
            agent_id=agent_id,
            task_id=task_id,
            phase=TaskPhase.EXECUTING,
        )

        # Iterate through plan steps, invoking tools for each.
        # In this milestone, tools dict may be empty (all mocked).
        files_changed: list[str] = list(state["files_changed"])

        for step in state["plan"]:
            if "code_editor" in tools:
                try:
                    result = await tools["code_editor"].invoke(
                        {"action": "write", "step": step}
                    )
                except GovernanceError as exc:
                    _log.warning(
                        "governance_error_in_executing",
                        agent_id=agent_id,
                        task_id=task_id,
                        error=str(exc),
                    )
                    return {
                        "phase": TaskPhase.REVIEWING,
                        "error": str(exc),
                    }
                if result.success and result.output:
                    files_changed.append(str(result.output))

        return {
            "phase": TaskPhase.REVIEWING,
            "files_changed": files_changed,
            "error": None,
        }

    return executing_node
