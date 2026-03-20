"""Integration test: four-node LangGraph workflow saves checkpoints to live PostgreSQL.

Milestone 2.3 DoD item 3:
    A task submitted to a workflow instance with mocked tools progresses through
    all four nodes and saves a checkpoint after each; verified by integration
    test against a live PostgreSQL instance.

Requires:
    docker compose -f deploy/docker-compose.yml up -d postgres

Run with:
    uv run pytest tests/integration/agent/test_workflow_live.py -m integration
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tractable.agent.state import AgentWorkflowState
from tractable.agent.workflow import build_workflow
from tractable.state.models import AgentCheckpointORM
from tractable.state.store import PostgreSQLAgentStateStore
from tractable.types.agent import AgentContext
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


# ── Minimal CodeGraph stub ────────────────────────────────────────────────────


class _StubCodeGraph:
    """Minimal CodeGraph stub that satisfies the Protocol structurally.

    Returns empty/zero results so the PLANNING node falls back to the
    'complete task' plan path without requiring a live FalkorDB instance.
    """

    async def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        return []

    async def get_entity(self, entity_id: str) -> GraphEntity | None:
        return None

    async def get_neighborhood(
        self,
        entity_id: str,
        depth: int = 2,
        min_confidence: float = 0.7,
    ) -> Subgraph:
        return Subgraph()

    async def impact_analysis(
        self,
        entity_ids: Sequence[str],
        depth: int = 3,
        min_confidence: float = 0.5,
    ) -> ImpactReport:
        return ImpactReport(risk_level=ChangeRisk.LOW)

    async def get_repo_boundary_edges(
        self,
        repo_name: str,
    ) -> Sequence[Any]:
        return []

    async def get_repo_summary(self, repo_name: str) -> RepoGraphSummary:
        # total_entities=0 triggers the graph_incomplete_fallback path
        # in plan.py, which is the correct empty-graph behaviour.
        return RepoGraphSummary(
            repo_name=repo_name,
            total_entities=0,
            summary_text="stub — no live graph",
        )

    async def mutate(self, mutations: Sequence[Any]) -> MutationResult:
        return MutationResult(applied=0)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_store() -> PostgreSQLAgentStateStore:
    engine = create_async_engine(_DATABASE_URL, pool_pre_ping=True)
    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine, expire_on_commit=False
    )
    return PostgreSQLAgentStateStore(factory)


# ── Test ─────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_saves_all_four_checkpoints_to_postgres() -> None:
    """A 4-node workflow run saves PLANNING/EXECUTING/REVIEWING/COORDINATING
    checkpoints to live PostgreSQL — Milestone 2.3 DoD item 3."""
    store = _make_store()
    agent_id = f"integration-agent-{uuid.uuid4()}"
    task_id = f"integration-task-{uuid.uuid4()}"

    # Seed agent context row required by the FK on agent_checkpoints.
    await store.save_agent_context(
        agent_id,
        AgentContext(
            agent_id=agent_id,
            base_template="api_maintainer",
            system_prompt="You are a coding agent.",
            repo_architectural_summary="REST API.",
        ),
    )

    workflow = build_workflow(
        tools={},
        state_store=store,
        graph=_StubCodeGraph(),
    )

    initial_state = AgentWorkflowState(
        agent_id=agent_id,
        task_id=task_id,
        task_description="Fix the failing test in test_auth.py",
        phase=TaskPhase.PLANNING,
        plan=[],
        files_changed=[],
        test_results={},
        pr_url=None,
        error=None,
        token_count=0,
        current_model="claude-sonnet-4-6",
        messages=[],
        resume_from=None,
    )

    # LangGraph requires a thread_id in the config for MemorySaver.
    config = {"configurable": {"thread_id": task_id}}
    await workflow.ainvoke(initial_state, config=config)

    # Query ALL checkpoint rows for this agent+task directly from PostgreSQL.
    engine = create_async_engine(_DATABASE_URL, pool_pre_ping=True)
    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine, expire_on_commit=False
    )
    async with factory() as session:
        result = await session.execute(
            select(AgentCheckpointORM)
            .where(
                AgentCheckpointORM.agent_id == agent_id,
                AgentCheckpointORM.task_id == task_id,
            )
            .order_by(AgentCheckpointORM.created_at.asc())
        )
        rows = list(result.scalars())

    saved_phases = {TaskPhase(row.phase) for row in rows}

    assert TaskPhase.PLANNING in saved_phases, (
        f"PLANNING checkpoint missing; phases saved: {saved_phases}"
    )
    assert TaskPhase.EXECUTING in saved_phases, (
        f"EXECUTING checkpoint missing; phases saved: {saved_phases}"
    )
    assert TaskPhase.REVIEWING in saved_phases, (
        f"REVIEWING checkpoint missing; phases saved: {saved_phases}"
    )
    assert TaskPhase.COORDINATING in saved_phases, (
        f"COORDINATING checkpoint missing; phases saved: {saved_phases}"
    )
    assert len(rows) >= 4, (
        f"Expected at least 4 checkpoint rows; found {len(rows)}"
    )
