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

TASK-3.1.2 additions:
  - test_checkpoint_resume_with_partial_write: crash-restore working directory
  - test_legacy_checkpoint_skips_restore_and_logs_warning: backwards compat
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tractable.agent.snapshot import create_snapshot
from tractable.agent.state import AgentWorkflowState
from tractable.agent.workflow import choose_entry_node, resume_task
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

    # LangGraph CompiledStateGraph.ainvoke: state: dict[str, Any], config: RunnableConfig | None
    async def fake_ainvoke(
        state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
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

    # LangGraph CompiledStateGraph.ainvoke: state: dict[str, Any], config: RunnableConfig | None
    async def fake_ainvoke(
        state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        captured_state.update(state)
        return dict(state)

    fake_workflow = MagicMock()
    fake_workflow.ainvoke = fake_ainvoke

    with (
        patch("tractable.agent.workflow.build_workflow", return_value=fake_workflow),
        patch("tractable.agent.workflow._log") as mock_log,
    ):
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


# ── TASK-3.1.2: crash-restore working directory ───────────────────────────────


@pytest.mark.asyncio
async def test_checkpoint_resume_with_partial_write() -> None:
    """resume_task() with working_dir restores the directory from snapshot.

    Sequence:
    (a) Create a working directory with a known file.
    (b) Create a snapshot of that initial state.
    (c) Write a partial file into the working directory (simulating a
        mid-EXECUTING crash that dirtied the filesystem).
    (d) Build a checkpoint that contains the snapshot_path and snapshot_hash.
    (e) Call resume_task() supplying that checkpoint and the working_dir.
    (f) Assert the working directory was restored to its pre-write state
        (original content, partial file removed) before the workflow ran.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        # (a) Initial working dir with a known file.
        working_dir = tmpdir / "workdir"
        working_dir.mkdir()
        original_file = working_dir / "source.py"
        original_file.write_text("# original content\n")

        snapshot_dir = tmpdir / "snapshots"

        # (b) Snapshot the clean state.
        archive_path, archive_hash = create_snapshot(working_dir, snapshot_dir)

        # (c) Simulate a partial write (crash mid-EXECUTING).
        original_file.write_text("# CORRUPTED PARTIAL WRITE\n")
        (working_dir / "partial_file.py").write_text("half-written\n")

        # (d) Build an AgentCheckpoint referencing the pre-write snapshot.
        workflow_state: dict[str, Any] = {
            "agent_id": "agent-1",
            "task_id": "task-restore",
            "task_description": "Fix the test",
            "phase": "executing",
            "plan": ["Step 1"],
            "files_changed": [],
            "test_results": {},
            "pr_url": None,
            "error": None,
            "token_count": 0,
            "messages": [],
            "resume_from": None,
            "snapshot_path": archive_path,
            "snapshot_hash": archive_hash,
        }
        checkpoint = AgentCheckpoint(
            task_id="task-restore",
            phase=TaskPhase.EXECUTING,
            progress_summary="mid-executing",
            files_modified=["source.py"],
            pending_actions=[],
            conversation_summary="",
            token_usage=0,
            created_at=datetime.now(tz=UTC),
            workflow_state=json.dumps(workflow_state),
            snapshot_path=archive_path,
            snapshot_hash=archive_hash,
        )

        store = AsyncMock()
        store.get_checkpoint.return_value = checkpoint

        # Track the state passed to the workflow to verify order of operations.
        state_at_invocation: dict[str, Any] = {}

        async def fake_ainvoke(
            state: dict[str, Any], config: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            # Record the working directory content at the moment of invocation.
            state_at_invocation["source_content"] = original_file.read_text()
            state_at_invocation["partial_exists"] = (
                working_dir / "partial_file.py"
            ).exists()
            return dict(state)

        fake_workflow = MagicMock()
        fake_workflow.ainvoke = fake_ainvoke

        # (e) Call resume_task with the working_dir.
        with patch("tractable.agent.workflow.build_workflow", return_value=fake_workflow):
            await resume_task(
                agent_id="agent-1",
                task_id="task-restore",
                task_description="Fix the test",
                state_store=store,
                tools={},
                graph=MagicMock(),
                working_dir=working_dir,
            )

        # (f) Assert working directory was restored before workflow ran.
        assert state_at_invocation["source_content"] == "# original content\n", (
            "Working directory should have been restored to the snapshot state "
            "before the workflow was invoked."
        )
        assert not state_at_invocation["partial_exists"], (
            "Partial file from the simulated crash should have been removed "
            "during snapshot restore."
        )


# ── TASK-3.1.2: legacy checkpoint (no snapshot) ──────────────────────────────


@pytest.mark.asyncio
async def test_legacy_checkpoint_skips_restore_and_logs_warning() -> None:
    """A checkpoint with no snapshot_path skips restore and logs snapshot_missing.

    This verifies backwards compatibility with Phase 2 checkpoints that were
    saved without a working-directory snapshot (AC-5).
    """
    legacy_workflow_state: dict[str, Any] = {
        "agent_id": "agent-1",
        "task_id": "task-legacy",
        "task_description": "Old task",
        "phase": "executing",
        "plan": ["Step 1"],
        "files_changed": [],
        "test_results": {},
        "pr_url": None,
        "error": None,
        "token_count": 0,
        "messages": [],
        "resume_from": None,
        # No snapshot_path or snapshot_hash — legacy format.
    }
    checkpoint = AgentCheckpoint(
        task_id="task-legacy",
        phase=TaskPhase.EXECUTING,
        progress_summary="",
        files_modified=[],
        pending_actions=[],
        conversation_summary="",
        token_usage=0,
        created_at=datetime.now(tz=UTC),
        workflow_state=json.dumps(legacy_workflow_state),
        # snapshot_path and snapshot_hash are None (default)
    )

    store = AsyncMock()
    store.get_checkpoint.return_value = checkpoint

    async def fake_ainvoke(
        state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return dict(state)

    fake_workflow = MagicMock()
    fake_workflow.ainvoke = fake_ainvoke

    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp) / "workdir"
        working_dir.mkdir()
        sentinel_file = working_dir / "sentinel.txt"
        sentinel_file.write_text("untouched\n")

        with (
            patch("tractable.agent.workflow.build_workflow", return_value=fake_workflow),
            patch("tractable.agent.workflow._log") as mock_log,
        ):
            await resume_task(
                agent_id="agent-1",
                task_id="task-legacy",
                task_description="Old task",
                state_store=store,
                tools={},
                graph=MagicMock(),
                working_dir=working_dir,
            )

        # Working directory must be untouched (no restore attempted).
        assert sentinel_file.read_text() == "untouched\n"

        # snapshot_missing warning must have been logged.
        mock_log.warning.assert_called_with(
            "snapshot_missing",
            level="warning",
            agent_id="agent-1",
            task_id="task-legacy",
            repo="",
            event="snapshot_missing",
        )
