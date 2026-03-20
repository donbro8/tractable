"""Integration tests: checkpoint restore on agent restart (TASK-2.5.1).

Acceptance criteria verified:
1. Checkpoint with phase=PLANNING → resume_task() logs checkpoint_restored,
   starts at EXECUTING (PLANNING node LLM logic not called).
   Verified against live PostgreSQL.
2. No checkpoint → resume_task() starts from PLANNING.
   Verified against live PostgreSQL.
4. Checkpoint with phase=REVIEWING → resume_task() skips PLANNING and
   EXECUTING, enters REVIEWING directly.
   Verified by node execution counts via mocked tools.

Requires:
    docker compose -f deploy/docker-compose.yml up -d postgres

Run with:
    uv run pytest tests/integration/agent/test_checkpoint_restore.py -m integration
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
import structlog.testing
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tractable.agent.workflow import resume_task
from tractable.protocols.tool import Tool, ToolResult
from tractable.state.store import PostgreSQLAgentStateStore
from tractable.types.agent import AgentCheckpoint, AgentContext
from tractable.types.enums import ChangeRisk, TaskPhase
from tractable.types.graph import (
    GraphEntity,
    ImpactReport,
    MutationResult,
    RepoGraphSummary,
    Subgraph,
)

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://tractable:tractable_dev@localhost:5433/tractable",
)


# ── Stubs ─────────────────────────────────────────────────────────────────────


class _StubCodeGraph:
    """Minimal CodeGraph stub with a call-tracking flag."""

    def __init__(self) -> None:
        self.repo_summary_called = False

    async def query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> Sequence[dict[str, Any]]:
        return []

    async def get_entity(self, entity_id: str) -> GraphEntity | None:
        return None

    async def get_neighborhood(
        self, entity_id: str, depth: int = 2, min_confidence: float = 0.7
    ) -> Subgraph:
        return Subgraph()

    async def impact_analysis(
        self,
        entity_ids: Sequence[str],
        depth: int = 3,
        min_confidence: float = 0.5,
    ) -> ImpactReport:
        return ImpactReport(risk_level=ChangeRisk.LOW)

    async def get_repo_boundary_edges(self, repo_name: str) -> Sequence[Any]:
        return []

    async def get_repo_summary(self, repo_name: str) -> RepoGraphSummary:
        self.repo_summary_called = True
        return RepoGraphSummary(repo_name=repo_name, total_entities=0, summary_text="stub")

    async def mutate(self, mutations: Sequence[Any]) -> MutationResult:
        return MutationResult(applied=0)


class _CallTrackingTool:
    """Tool that records every invoke() call for assertion purposes."""

    def __init__(self, name_: str) -> None:
        self._name = name_
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Tracking tool: {self._name}"

    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        self.call_count += 1
        return ToolResult(success=True)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_store() -> PostgreSQLAgentStateStore:
    engine = create_async_engine(_DATABASE_URL, pool_pre_ping=True)
    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(engine, expire_on_commit=False)
    return PostgreSQLAgentStateStore(factory)


async def _seed_agent(store: PostgreSQLAgentStateStore, agent_id: str) -> None:
    await store.save_agent_context(
        agent_id,
        AgentContext(
            agent_id=agent_id,
            base_template="api_maintainer",
            system_prompt="You are a coding agent.",
            repo_architectural_summary="REST API.",
        ),
    )


def _make_workflow_state(
    agent_id: str,
    task_id: str,
    *,
    plan: list[str] | None = None,
    files_changed: list[str] | None = None,
    phase: str = "planning",
) -> str:
    state: dict[str, Any] = {
        "agent_id": agent_id,
        "task_id": task_id,
        "task_description": "Fix the failing test in test_auth.py",
        "phase": phase,
        "plan": plan or ["Step 1: write the fix"],
        "files_changed": files_changed or [],
        "test_results": {},
        "pr_url": None,
        "error": None,
        "token_count": 0,
        "messages": [],
        "resume_from": None,
    }
    return json.dumps(state)


# ── AC-1: PLANNING checkpoint → skip to EXECUTING ─────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_restore_planning_checkpoint_logs_and_skips_planning() -> None:
    """AC-1: phase=PLANNING checkpoint → logs checkpoint_restored, goes to EXECUTING.

    The PLANNING node's LLM logic (get_repo_summary) must NOT be called.
    """
    store = _make_store()
    agent_id = f"integration-agent-{uuid.uuid4()}"
    task_id = f"integration-task-{uuid.uuid4()}"
    await _seed_agent(store, agent_id)

    # Pre-save a PLANNING checkpoint with a plan in workflow_state.
    checkpoint = AgentCheckpoint(
        task_id=task_id,
        phase=TaskPhase.PLANNING,
        progress_summary="PLANNING done",
        files_modified=[],
        pending_actions=[],
        conversation_summary="",
        token_usage=0,
        created_at=datetime.now(tz=UTC),
        workflow_state=_make_workflow_state(agent_id, task_id, plan=["step 1"]),
    )
    await store.save_checkpoint(agent_id, task_id, checkpoint)

    stub_graph = _StubCodeGraph()
    config = {"configurable": {"thread_id": task_id}}

    with structlog.testing.capture_logs() as captured:
        await resume_task(
            agent_id=agent_id,
            task_id=task_id,
            task_description="Fix the failing test",
            state_store=store,
            tools={},
            graph=stub_graph,
            config=config,
        )

    # AC-1a: checkpoint_restored was logged with correct phase
    restore_events = [e for e in captured if e.get("event") == "checkpoint_restored"]
    assert restore_events, f"No checkpoint_restored log entry found; captured: {captured}"
    assert restore_events[0]["phase"] == str(TaskPhase.PLANNING)
    assert restore_events[0]["agent_id"] == agent_id
    assert restore_events[0]["task_id"] == task_id

    # AC-1b: PLANNING node's LLM logic (get_repo_summary) was NOT called
    assert not stub_graph.repo_summary_called, (
        "PLANNING node ran its LLM logic despite restoring from a PLANNING checkpoint"
    )


# ── AC-2: no checkpoint → fresh start from PLANNING ──────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_checkpoint_starts_from_planning() -> None:
    """AC-2: no checkpoint → workflow starts from PLANNING (fresh run)."""
    store = _make_store()
    agent_id = f"integration-agent-{uuid.uuid4()}"
    task_id = f"integration-task-{uuid.uuid4()}"
    await _seed_agent(store, agent_id)

    stub_graph = _StubCodeGraph()
    config = {"configurable": {"thread_id": task_id}}

    with structlog.testing.capture_logs() as captured:
        await resume_task(
            agent_id=agent_id,
            task_id=task_id,
            task_description="Fix the failing test",
            state_store=store,
            tools={},
            graph=stub_graph,
            config=config,
        )

    # No checkpoint_restored event should appear
    restore_events = [e for e in captured if e.get("event") == "checkpoint_restored"]
    assert not restore_events, f"Unexpected checkpoint_restored log: {restore_events}"

    # PLANNING node ran (get_repo_summary was called)
    assert stub_graph.repo_summary_called, (
        "PLANNING node did not run for a fresh start (no checkpoint)"
    )


# ── AC-4: REVIEWING checkpoint → skip PLANNING and EXECUTING ─────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_restore_reviewing_checkpoint_skips_planning_and_executing() -> None:
    """AC-4: phase=REVIEWING checkpoint → agent skips PLANNING + EXECUTING.

    Verified by asserting that the code_editor tool (called in EXECUTING) was
    NOT invoked, while REVIEWING ran (test_runner/linter tools were invoked).
    """
    store = _make_store()
    agent_id = f"integration-agent-{uuid.uuid4()}"
    task_id = f"integration-task-{uuid.uuid4()}"
    await _seed_agent(store, agent_id)

    # Pre-save a REVIEWING checkpoint (PLANNING+EXECUTING already done).
    checkpoint = AgentCheckpoint(
        task_id=task_id,
        phase=TaskPhase.REVIEWING,
        progress_summary="REVIEWING starting",
        files_modified=["src/auth.py"],
        pending_actions=[],
        conversation_summary="",
        token_usage=0,
        created_at=datetime.now(tz=UTC),
        workflow_state=_make_workflow_state(
            agent_id,
            task_id,
            plan=["step 1"],
            files_changed=["src/auth.py"],
            phase="reviewing",
        ),
    )
    await store.save_checkpoint(agent_id, task_id, checkpoint)

    stub_graph = _StubCodeGraph()

    # Inject tracking tools to count node executions.
    code_editor = _CallTrackingTool("code_editor")
    test_runner = _CallTrackingTool("test_runner")
    linter = _CallTrackingTool("linter")

    tools: dict[str, Tool] = {
        "code_editor": code_editor,
        "test_runner": test_runner,
        "linter": linter,
    }
    config = {"configurable": {"thread_id": task_id}}

    await resume_task(
        agent_id=agent_id,
        task_id=task_id,
        task_description="Fix the failing test",
        state_store=store,
        tools=tools,
        graph=stub_graph,
        config=config,
    )

    # EXECUTING was skipped: code_editor should NOT have been called.
    assert code_editor.call_count == 0, (
        f"code_editor was called {code_editor.call_count} time(s); "
        "EXECUTING should have been skipped"
    )

    # PLANNING was skipped: get_repo_summary should NOT have been called.
    assert not stub_graph.repo_summary_called, (
        "PLANNING ran despite restoring from a REVIEWING checkpoint"
    )

    # REVIEWING ran: test_runner and/or linter were called.
    assert test_runner.call_count > 0 or linter.call_count > 0, (
        "Neither test_runner nor linter was called; REVIEWING may not have run"
    )
