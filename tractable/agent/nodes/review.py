"""REVIEWING node — third node in the agent workflow.

TASK-2.3.1: Saves a checkpoint with phase=REVIEWING, enforces GovernancePolicy
gates (mocked in this milestone), and routes either back to EXECUTING on
failure or forward to COORDINATING on success.

TASK-3.2.3: Enforces ``max_lines_per_change`` governance limit after tests
pass.  Accepts an optional ``_count_lines`` callable for testing without a
live git repository.

Routing constants
-----------------
RETRY_EDGE
    Returned by the routing function when governance gates fail; LangGraph
    follows this edge back to the EXECUTING node.
DONE_EDGE
    Returned when gates pass; LangGraph follows this edge to COORDINATING.
"""

from __future__ import annotations

import subprocess
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
    from tractable.types.config import GovernancePolicy

_log = structlog.get_logger()

# Edge labels used by the conditional router in workflow.py
RETRY_EDGE = "retry"
DONE_EDGE = "done"


def _git_diff_stat_lines(_files_changed: list[str]) -> int:
    """Return total lines changed via ``git diff --stat``.

    Parses the summary line of ``git diff --stat`` output, e.g.::

        3 files changed, 42 insertions(+), 7 deletions(-)

    Returns insertions + deletions.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout.strip()
        if not output:
            return 0
        summary = output.splitlines()[-1]
        insertions = 0
        deletions = 0
        for part in summary.split(","):
            part = part.strip()
            if "insertion" in part:
                insertions = int(part.split()[0])
            elif "deletion" in part:
                deletions = int(part.split()[0])
        return insertions + deletions
    except Exception:
        return 0


# ── Public factory ─────────────────────────────────────────────────────────


def make_reviewing_node(
    tools: dict[str, Tool],
    state_store: AgentStateStore,
    governance: GovernancePolicy | None = None,
    _count_lines: Callable[[list[str]], int] | None = None,
) -> Callable[[AgentWorkflowState], Coroutine[Any, Any, dict[str, Any]]]:
    """Return an async REVIEWING node with injected dependencies.

    Parameters
    ----------
    tools:
        Tool name → Tool mapping; injected at workflow construction time.
    state_store:
        Used to persist the REVIEWING-phase checkpoint.
    governance:
        When supplied, ``max_lines_per_change`` is checked after tests pass.
    _count_lines:
        Optional override for line counting; defaults to ``_git_diff_stat_lines``.
        Pass a stub in unit tests to avoid requiring a real git repository.
    """
    count_lines = _count_lines if _count_lines is not None else _git_diff_stat_lines

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

        # ── GovernanceError from EXECUTING: notify human and preserve error ──
        existing_error = state["error"]
        if existing_error is not None and existing_error.startswith("Sensitive path blocked"):
            pr_url = state["pr_url"] or ""
            body = (
                f"⚠️ Sensitive path blocked\n\n{existing_error}\n\n"
                "Please review and either approve the path write or update "
                "the task description to avoid this path."
            )
            if "git_ops" in tools:
                await tools["git_ops"].invoke(
                    {"operation": "pr_comment", "pr_url": pr_url, "body": body}
                )
            return {"error": existing_error}

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

        # ── Governance: max_lines_per_change ───────────────────────────────
        if governance is not None:
            lines_count = count_lines(list(state["files_changed"]))
            if lines_count > governance.max_lines_per_change:
                replan_count: int = state.get("replan_count", 0)  # type: ignore[call-overload]
                _log.warning(
                    "governance_violation",
                    agent_id=agent_id,
                    task_id=task_id,
                    repo="",
                    type="max_lines_per_change",
                    lines_changed=lines_count,
                    limit=governance.max_lines_per_change,
                )
                return {
                    "phase": TaskPhase.PLANNING,
                    "replan_count": replan_count + 1,
                    "error": (
                        "max_lines_per_change exceeded; "
                        "split this change into smaller increments"
                    ),
                }

        return {
            "phase": TaskPhase.COORDINATING,
            "error": None,
        }

    return reviewing_node


# ── Routing function ──────────────────────────────────────────────────────


def reviewing_router(state: AgentWorkflowState) -> str:
    """Conditional edge: DONE_EDGE when gates pass, RETRY_EDGE on failure."""
    return DONE_EDGE if state["error"] is None else RETRY_EDGE
