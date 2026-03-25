"""Unit tests for governance limit enforcement (TASK-3.2.3).

Covers:
- AC-1: max_files_per_change triggers re-plan when file count exceeds limit.
- AC-2: max_lines_per_change triggers re-plan when line count exceeds limit.
- AC-3: replan_count >= max_retries_on_failure causes FAILED instead of re-plan.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog.testing

from tractable.agent.nodes.execute import make_executing_node
from tractable.agent.nodes.review import make_reviewing_node
from tractable.agent.state import AgentWorkflowState
from tractable.types.config import GovernancePolicy
from tractable.types.enums import TaskPhase

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_state(
    files_changed: list[str] | None = None,
    replan_count: int = 0,
) -> AgentWorkflowState:
    state = AgentWorkflowState(
        agent_id="agent-test",
        task_id="task-test",
        task_description="test task",
        phase=TaskPhase.EXECUTING,
        plan=[],
        files_changed=files_changed or [],
        test_results={},
        pr_url=None,
        error=None,
        token_count=0,
        current_model="claude-sonnet-4-6",
        messages=[],
        resume_from=None,
    )
    state["replan_count"] = replan_count  # type: ignore[typeddict-unknown-key]
    return state


def _stub_state_store() -> MagicMock:
    store = MagicMock()
    store.save_checkpoint = AsyncMock()
    return store


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_files_per_change_triggers_replan() -> None:
    """AC-1: EXECUTING node re-plans when files_changed exceeds max_files_per_change."""
    governance = GovernancePolicy(max_files_per_change=20)
    node = make_executing_node({}, _stub_state_store(), governance=governance)
    state = _make_state(files_changed=[f"file{i}.py" for i in range(21)])

    with structlog.testing.capture_logs() as logs:
        result = await node(state)

    assert result["phase"] == TaskPhase.PLANNING, (
        f"Expected PLANNING, got {result['phase']}"
    )
    violation_logs = [
        e for e in logs if e.get("event") == "governance_violation"
    ]
    assert len(violation_logs) >= 1, "Expected at least one governance_violation log entry"
    assert violation_logs[0].get("type") == "max_files_per_change"


@pytest.mark.asyncio
async def test_max_lines_per_change_triggers_replan() -> None:
    """AC-2: REVIEWING node re-plans when total lines changed exceeds max_lines_per_change."""
    governance = GovernancePolicy(max_lines_per_change=500)
    node = make_reviewing_node(
        {},
        _stub_state_store(),
        governance=governance,
        _count_lines=lambda _files: 501,
    )
    state = _make_state(files_changed=["main.py"])

    with structlog.testing.capture_logs() as logs:
        result = await node(state)

    assert result["phase"] == TaskPhase.PLANNING, (
        f"Expected PLANNING, got {result['phase']}"
    )
    violation_logs = [
        e for e in logs if e.get("event") == "governance_violation"
    ]
    assert len(violation_logs) >= 1, "Expected at least one governance_violation log entry"
    assert violation_logs[0].get("type") == "max_lines_per_change"


@pytest.mark.asyncio
async def test_replan_limit_reached_fails_task() -> None:
    """AC-3: EXECUTING node sets phase=FAILED when replan_count >= max_retries_on_failure."""
    governance = GovernancePolicy(max_files_per_change=20, max_retries_on_failure=3)
    node = make_executing_node({}, _stub_state_store(), governance=governance)
    # replan_count is already at the limit; files still over the limit
    state = _make_state(
        files_changed=[f"file{i}.py" for i in range(21)],
        replan_count=3,
    )

    with structlog.testing.capture_logs() as logs:
        result = await node(state)

    assert result["phase"] == TaskPhase.FAILED, (
        f"Expected FAILED, got {result['phase']}"
    )
    limit_logs = [
        e for e in logs if e.get("event") == "governance_replan_limit_reached"
    ]
    assert len(limit_logs) >= 1, (
        "Expected at least one governance_replan_limit_reached log entry"
    )
