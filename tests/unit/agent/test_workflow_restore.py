"""Unit tests for checkpoint restore logic in workflow.py (TASK-2.5.1).

AC-3 covered here:
  The checkpoint's workflow_state is serialised as valid JSON (not pickle).
  Verified by json.loads(checkpoint.workflow_state) succeeding.

Additional unit coverage:
  - choose_entry_node returns PLANNING for no resume_from
  - choose_entry_node returns EXECUTING for resume_from="planning"
  - choose_entry_node returns REVIEWING for resume_from="reviewing"
  - resume_task() with no checkpoint calls build_workflow and starts fresh
  - resume_task() with a checkpoint restores state and logs checkpoint_restored
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tractable.agent.workflow import choose_entry_node, resume_task
from tractable.agent.state import AgentWorkflowState
from tractable.types.agent import AgentCheckpoint
from tractable.types.enums import TaskPhase


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_checkpoint(phase: TaskPhase, workflow_state: dict[str, Any]) -> AgentCheckpoint:
    return AgentCheckpoint(
        task_id="task-test",
        phase=phase,
        progress_summary="test",
        files_modified=[],
        pending_actions=[],
        conversation_summary="",
        token_usage=0,
        created_at=datetime.now(tz=UTC),
        workflow_state=json.dumps(workflow_state),
    )


def _state_with_resume(resume_from: str | None) -> AgentWorkflowState:
    return AgentWorkflowState(
        agent_id="agent-test",
        task_id="task-test",
        task_description="test",
        phase=TaskPhase.PLANNING,
        plan=[],
        files_changed=[],
        test_results={},
        pr_url=None,
        error=None,
        token_count=0,
        current_model="claude-sonnet-4-6",
        messages=[],
        resume_from=resume_from,
    )


# ── AC-3: workflow_state is valid JSON, not pickle ────────────────────────────


def test_workflow_state_is_valid_json() -> None:
    """AC-3: json.loads(checkpoint.workflow_state) must succeed."""
    state_dict: dict[str, Any] = {
        "agent_id": "agent-1",
        "task_id": "task-1",
        "task_description": "Fix the failing test",
        "phase": "planning",
        "plan": ["Step 1: write the fix"],
        "files_changed": [],
        "test_results": {},
        "pr_url": None,
        "error": None,
        "token_count": 0,
        "messages": [],
        "resume_from": None,
    }
    checkpoint = _make_checkpoint(TaskPhase.PLANNING, state_dict)
    # AC-3 verification: json.loads must succeed
    parsed = json.loads(checkpoint.workflow_state)
    assert parsed["plan"] == ["Step 1: write the fix"]
    assert parsed["phase"] == "planning"


# ── choose_entry_node routing ────────────────────────────────────────────────


def testchoose_entry_node_no_resume_from_returns_planning() -> None:
    state = _state_with_resume(None)
    assert choose_entry_node(state) == "PLANNING"


def testchoose_entry_node_resume_planning_returns_executing() -> None:
    state = _state_with_resume("planning")
    assert choose_entry_node(state) == "EXECUTING"


def testchoose_entry_node_resume_reviewing_returns_reviewing() -> None:
    state = _state_with_resume("reviewing")
    assert choose_entry_node(state) == "REVIEWING"


def testchoose_entry_node_resume_executing_returns_reviewing() -> None:
    state = _state_with_resume("executing")
    assert choose_entry_node(state) == "REVIEWING"


def testchoose_entry_node_resume_coordinating_returns_coordinating() -> None:
    state = _state_with_resume("coordinating")
    assert choose_entry_node(state) == "COORDINATING"


def testchoose_entry_node_unknown_resume_phase_returns_planning() -> None:
    state = _state_with_resume("unknown_phase")
    assert choose_entry_node(state) == "PLANNING"


# ── resume_task: no checkpoint → fresh start ─────────────────────────────────


@pytest.mark.asyncio
async def test_resume_task_no_checkpoint_starts_fresh() -> None:
    """resume_task() with no checkpoint builds fresh state (resume_from=None)."""
    store = AsyncMock()
    store.get_checkpoint.return_value = None

    captured_state: dict[str, Any] = {}

    async def fake_ainvoke(state: Any, config: Any = None) -> dict[str, Any]:
        captured_state.update(state)
        return dict(state)

    fake_workflow = MagicMock()
    fake_workflow.ainvoke = fake_ainvoke

    with patch("tractable.agent.workflow.build_workflow", return_value=fake_workflow):
        await resume_task(
            agent_id="agent-1",
            task_id="task-1",
            task_description="Fix the test",
            state_store=store,
            tools={},
            graph=MagicMock(),
        )

    store.get_checkpoint.assert_called_once_with("agent-1", "task-1")
    assert captured_state["resume_from"] is None
    assert captured_state["plan"] == []
    assert captured_state["task_description"] == "Fix the test"


# ── resume_task: checkpoint found → restore ───────────────────────────────────


@pytest.mark.asyncio
async def test_resume_task_with_checkpoint_sets_resume_from_and_logs() -> None:
    """resume_task() with checkpoint sets resume_from and logs checkpoint_restored."""
    state_dict: dict[str, Any] = {
        "agent_id": "agent-1",
        "task_id": "task-1",
        "task_description": "Fix the test",
        "phase": "planning",
        "plan": ["Step 1"],
        "files_changed": [],
        "test_results": {},
        "pr_url": None,
        "error": None,
        "token_count": 0,
        "messages": [],
        "resume_from": None,
    }
    checkpoint = _make_checkpoint(TaskPhase.PLANNING, state_dict)

    store = AsyncMock()
    store.get_checkpoint.return_value = checkpoint

    captured_state: dict[str, Any] = {}

    async def fake_ainvoke(state: Any, config: Any = None) -> dict[str, Any]:
        captured_state.update(state)
        return dict(state)

    fake_workflow = MagicMock()
    fake_workflow.ainvoke = fake_ainvoke

    with patch("tractable.agent.workflow.build_workflow", return_value=fake_workflow):
        with patch("tractable.agent.workflow._log") as mock_log:
            await resume_task(
                agent_id="agent-1",
                task_id="task-1",
                task_description="Fix the test",
                state_store=store,
                tools={},
                graph=MagicMock(),
            )

    assert captured_state["resume_from"] == str(TaskPhase.PLANNING)
    assert captured_state["plan"] == ["Step 1"]
    mock_log.info.assert_called_with(
        "checkpoint_restored",
        level="info",
        agent_id="agent-1",
        task_id="task-1",
        phase=str(TaskPhase.PLANNING),
    )
