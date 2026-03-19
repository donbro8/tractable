"""PLANNING node — first node in the agent workflow.

TASK-2.3.1: Saves a checkpoint with phase=PLANNING, queries the graph for
relevant entities (with incomplete-neighbourhood fallback), and produces a
step-by-step plan.  In this milestone all tool invocations are mediated
through the injected ``tools`` dict; real implementations are substituted in
Milestone 2.4 without touching this module.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from tractable.logging import bind_context
from tractable.types.agent import AgentCheckpoint
from tractable.types.enums import TaskPhase

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from tractable.agent.state import AgentWorkflowState
    from tractable.protocols.agent_state_store import AgentStateStore
    from tractable.protocols.code_graph import CodeGraph
    from tractable.protocols.tool import Tool

_log = structlog.get_logger()

# ── Public factory ─────────────────────────────────────────────────────────


def make_planning_node(
    tools: dict[str, Tool],
    state_store: AgentStateStore,
    graph: CodeGraph,
) -> Callable[[AgentWorkflowState], Coroutine[Any, Any, dict[str, Any]]]:
    """Return an async PLANNING node with injected dependencies.

    Parameters
    ----------
    tools:
        Tool name → Tool mapping; injected at workflow construction time.
    state_store:
        Used to persist the PLANNING-phase checkpoint.
    graph:
        CodeGraph queried for relevant entities during planning.
    """

    async def planning_node(state: AgentWorkflowState) -> dict[str, Any]:
        agent_id = state["agent_id"]
        task_id = state["task_id"]
        bind_context(agent_id=agent_id, task_id=task_id)

        checkpoint = AgentCheckpoint(
            task_id=task_id,
            phase=TaskPhase.PLANNING,
            progress_summary="Entering PLANNING node",
            files_modified=[],
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
            phase=TaskPhase.PLANNING,
        )

        # Query graph for relevant entities; fall back to direct file reads
        # when the neighbourhood is incomplete (see phase-1-analysis.md §6.2).
        plan: list[str] = []
        try:
            summary = await graph.get_repo_summary(state.get("task_description", ""))
            if summary.total_entities == 0:
                _log.warning(
                    "graph_incomplete_fallback",
                    agent_id=agent_id,
                    task_id=task_id,
                    reason="empty_repo_summary",
                )
                # Fallback: delegate to code_editor tool if available.
                if "code_editor" in tools:
                    result = await tools["code_editor"].invoke(
                        {"action": "read_file", "path": "."}
                    )
                    if result.success:
                        plan = [f"Read key files: {result.output}"]
            else:
                plan = [f"Analyse entity graph: {summary.summary_text}"]
        except Exception:
            _log.warning(
                "graph_incomplete_fallback",
                agent_id=agent_id,
                task_id=task_id,
                reason="graph_query_error",
                exc_info=True,
            )

        if not plan:
            plan = [f"Complete task: {state['task_description']}"]

        return {
            "phase": TaskPhase.EXECUTING,
            "plan": plan,
            "error": None,
        }

    return planning_node
