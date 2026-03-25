"""Unit tests for governance limit enforcement (TASK-3.2.3, TASK-3.2.4).

Covers:
- AC-1: max_files_per_change triggers re-plan when file count exceeds limit.
- AC-2: max_lines_per_change triggers re-plan when line count exceeds limit.
- AC-3: replan_count >= max_retries_on_failure causes FAILED instead of re-plan.
- AC-3 (TASK-3.2.4): blocked write appends audit entry; get_audit_log() returns it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog.testing

from tractable.agent.nodes.execute import make_executing_node
from tractable.agent.nodes.review import make_reviewing_node
from tractable.agent.state import AgentWorkflowState
from tractable.agent.tools.code_editor import CodeEditorTool
from tractable.errors import GovernanceError
from tractable.types.agent import AgentCheckpoint, AgentContext, AuditEntry
from tractable.types.config import AgentScope, GovernancePolicy, SensitivePathRule
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


# ── In-memory store for audit log tests ────────────────────────────────────


class _InMemoryStateStore:
    """Minimal AgentStateStore stub that records audit entries in memory."""

    def __init__(self) -> None:
        self._audit_log: list[AuditEntry] = []

    async def get_agent_context(self, agent_id: str) -> AgentContext:
        raise NotImplementedError

    async def list_agents(self) -> Sequence[AgentContext]:
        return []

    async def save_agent_context(self, agent_id: str, context: AgentContext) -> None:
        pass

    async def get_checkpoint(
        self, agent_id: str, task_id: str
    ) -> AgentCheckpoint | None:
        return None

    async def save_checkpoint(
        self, agent_id: str, task_id: str, checkpoint: AgentCheckpoint
    ) -> None:
        pass

    async def append_audit_entry(self, entry: AuditEntry) -> None:
        self._audit_log.append(entry)

    async def get_audit_log(
        self,
        agent_id: str | None = None,
        task_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> Sequence[AuditEntry]:
        entries = self._audit_log
        if agent_id is not None:
            entries = [e for e in entries if e.agent_id == agent_id]
        if task_id is not None:
            entries = [e for e in entries if e.task_id == task_id]
        if since is not None:
            entries = [e for e in entries if e.timestamp >= since]
        return entries[:limit]

    async def get_last_polled_sha(self, repo_id: str) -> str | None:
        return None

    async def set_last_polled_sha(self, repo_id: str, sha: str) -> None:
        pass


# ── AC-3 (TASK-3.2.4) ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sensitive_path_blocked_appends_audit_entry(tmp_path: Path) -> None:
    """AC-3 (3.2.4): blocked write appends AuditEntry; get_audit_log() returns it."""
    (tmp_path / "db" / "migrations").mkdir(parents=True)
    store = _InMemoryStateStore()
    governance = GovernancePolicy(
        sensitive_path_patterns=[
            SensitivePathRule(
                pattern="**/migrations/**",
                reason="Database migrations require DBA review",
                policy="human_approval_required",
            )
        ]
    )
    tool = CodeEditorTool(
        working_dir=tmp_path,
        scope=AgentScope(),
        governance=governance,
        state_store=store,  # type: ignore[arg-type]
        agent_id="agent-test",
        task_id="task-test",
        repo="test/repo",
    )

    with pytest.raises(GovernanceError):
        await tool.invoke(
            {
                "operation": "write_file",
                "path": "db/migrations/002_add_col.sql",
                "content": "ALTER TABLE t ADD COLUMN x INT;",
            }
        )

    # Allow the fire-and-forget task to run.
    await asyncio.sleep(0)

    audit_log = await store.get_audit_log(agent_id="agent-test", task_id="task-test")
    matching = [e for e in audit_log if e.action == "sensitive_path_blocked"]
    assert len(matching) >= 1, (
        f"Expected sensitive_path_blocked audit entry; got: {[e.action for e in audit_log]}"
    )
    assert matching[0].detail.get("rule") == "**/migrations/**"
