"""Unit tests for tractable/agent/workflow.py — TASK-2.3.1.

Covers:
- Workflow construction and import (AC-1)
- State progression through all four nodes (AC-2)
- Checkpoint saved with phase=PLANNING after planning node (AC-3)
- Checkpoint saved with phase=EXECUTING after executing node (AC-4)
- REVIEWING → EXECUTING retry edge when governance gate fails
- No hardcoded API keys in tractable/agent/ (AC-7, smoke check via import)
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pytest

from tractable.agent.state import AgentWorkflowState
from tractable.agent.workflow import build_workflow
from tractable.protocols.tool import ToolResult
from tractable.types.agent import AgentCheckpoint, AgentContext, AuditEntry
from tractable.types.enums import TaskPhase
from tractable.types.graph import (
    CrossRepoEdge,
    GraphEntity,
    ImpactReport,
    MutationResult,
    RepoGraphSummary,
    Subgraph,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_initial_state(
    agent_id: str = "agent-test",
    task_id: str = "task-test",
    task_description: str = "Fix the bug",
    current_model: str = "claude-sonnet-4-6",
) -> AgentWorkflowState:
    return AgentWorkflowState(
        agent_id=agent_id,
        task_id=task_id,
        task_description=task_description,
        phase=TaskPhase.PLANNING,
        plan=[],
        files_changed=[],
        test_results={},
        pr_url=None,
        error=None,
        token_count=0,
        current_model=current_model,
        messages=[],
        resume_from=None,
    )


class _MockStateStore:
    """Minimal AgentStateStore mock that records save_checkpoint calls."""

    def __init__(self) -> None:
        self.saved_checkpoints: list[AgentCheckpoint] = []

    async def get_agent_context(self, agent_id: str) -> AgentContext:
        return AgentContext(
            agent_id=agent_id,
            base_template="test",
            system_prompt="",
            repo_architectural_summary="",
        )

    async def save_agent_context(self, agent_id: str, context: AgentContext) -> None:
        pass

    async def get_checkpoint(self, agent_id: str, task_id: str) -> AgentCheckpoint | None:
        return None

    async def save_checkpoint(
        self, agent_id: str, task_id: str, checkpoint: AgentCheckpoint
    ) -> None:
        self.saved_checkpoints.append(checkpoint)

    async def append_audit_entry(self, entry: AuditEntry) -> None:
        pass

    async def get_audit_log(
        self,
        agent_id: str | None = None,
        task_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> Sequence[AuditEntry]:
        return []


class _MockGraph:
    """Minimal CodeGraph mock that returns a non-empty RepoGraphSummary."""

    async def query(
        self, cypher: str, params: dict[str, Any] | None = None
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
        return Subgraph(nodes=[], edges=[])

    async def impact_analysis(
        self,
        entity_ids: Sequence[str],
        depth: int = 3,
        min_confidence: float = 0.5,
    ) -> ImpactReport:
        from tractable.types.enums import ChangeRisk

        return ImpactReport(
            directly_affected=[],
            transitively_affected=[],
            affected_repos=[],
            cross_repo_edges=[],
            risk_level=ChangeRisk.LOW,
        )

    async def get_repo_boundary_edges(self, repo_name: str) -> Sequence[CrossRepoEdge]:
        return []

    async def get_repo_summary(self, repo_name: str) -> RepoGraphSummary:
        return RepoGraphSummary(
            repo_name=repo_name,
            total_entities=5,
            key_modules=["main.py"],
            public_interfaces=[],
            cross_repo_dependencies=[],
            summary_text="Test repo summary",
        )

    async def mutate(self, mutations: Sequence[Any]) -> MutationResult:
        return MutationResult(applied=0)


# ---------------------------------------------------------------------------
# AC-1: build_workflow imports and constructs without error
# ---------------------------------------------------------------------------


def test_build_workflow_imports() -> None:
    """AC-1: build_workflow can be imported and called with mocked deps."""
    store = _MockStateStore()
    graph = _MockGraph()
    compiled = build_workflow(tools={}, state_store=store, graph=graph)
    assert compiled is not None


def test_build_workflow_returns_compiled_graph() -> None:
    """AC-1: The return value is a compiled LangGraph StateGraph object."""
    from langgraph.graph.state import CompiledStateGraph

    store = _MockStateStore()
    graph = _MockGraph()
    compiled = build_workflow(tools={}, state_store=store, graph=graph)
    assert isinstance(compiled, CompiledStateGraph)


# ---------------------------------------------------------------------------
# AC-2: State progresses through all four nodes in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_phase_progression() -> None:
    """AC-2: Phase transitions PLANNING → EXECUTING → REVIEWING → COORDINATING → COMPLETED."""
    store = _MockStateStore()
    graph = _MockGraph()
    compiled = build_workflow(tools={}, state_store=store, graph=graph)

    initial = _make_initial_state()
    config = {"configurable": {"thread_id": "test-thread-1"}}

    final_state: dict[str, Any] = {}
    async for state in compiled.astream(initial, config=config):
        # Each stream event is a dict keyed by node name.
        # We track the most recent state.
        for node_output in state.values():
            if isinstance(node_output, dict):
                final_state.update(node_output)

    assert final_state.get("phase") == TaskPhase.COMPLETED


@pytest.mark.asyncio
async def test_workflow_node_order() -> None:
    """AC-2: All four nodes execute in the correct order."""
    store = _MockStateStore()
    graph = _MockGraph()
    compiled = build_workflow(tools={}, state_store=store, graph=graph)

    initial = _make_initial_state()
    config = {"configurable": {"thread_id": "test-thread-2"}}

    visited_nodes: list[str] = []
    async for event in compiled.astream(initial, config=config):
        visited_nodes.extend(event.keys())

    assert visited_nodes == ["PLANNING", "EXECUTING", "REVIEWING", "COORDINATING"]


# ---------------------------------------------------------------------------
# AC-3: save_checkpoint called with phase=PLANNING after PLANNING node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planning_node_saves_checkpoint() -> None:
    """AC-3: AgentStateStore.save_checkpoint called with phase=PLANNING after PLANNING."""
    store = _MockStateStore()
    graph = _MockGraph()
    compiled = build_workflow(tools={}, state_store=store, graph=graph)

    initial = _make_initial_state(agent_id="agent-a", task_id="task-a")
    config = {"configurable": {"thread_id": "test-thread-3"}}

    async for _ in compiled.astream(initial, config=config):
        pass

    planning_checkpoints = [c for c in store.saved_checkpoints if c.phase == TaskPhase.PLANNING]
    assert len(planning_checkpoints) >= 1
    assert planning_checkpoints[0].task_id == "task-a"


# ---------------------------------------------------------------------------
# AC-4: save_checkpoint called with phase=EXECUTING after EXECUTING node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executing_node_saves_checkpoint() -> None:
    """AC-4: AgentStateStore.save_checkpoint called with phase=EXECUTING after EXECUTING."""
    store = _MockStateStore()
    graph = _MockGraph()
    compiled = build_workflow(tools={}, state_store=store, graph=graph)

    initial = _make_initial_state(agent_id="agent-b", task_id="task-b")
    config = {"configurable": {"thread_id": "test-thread-4"}}

    async for _ in compiled.astream(initial, config=config):
        pass

    executing_checkpoints = [c for c in store.saved_checkpoints if c.phase == TaskPhase.EXECUTING]
    assert len(executing_checkpoints) >= 1
    assert executing_checkpoints[0].task_id == "task-b"


# ---------------------------------------------------------------------------
# Checkpoint phase ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_four_checkpoints_saved_in_order() -> None:
    """Checkpoints are saved for all four phases in workflow order."""
    store = _MockStateStore()
    graph = _MockGraph()
    compiled = build_workflow(tools={}, state_store=store, graph=graph)

    initial = _make_initial_state()
    config = {"configurable": {"thread_id": "test-thread-5"}}

    async for _ in compiled.astream(initial, config=config):
        pass

    phases = [c.phase for c in store.saved_checkpoints]
    assert TaskPhase.PLANNING in phases
    assert TaskPhase.EXECUTING in phases
    assert TaskPhase.REVIEWING in phases
    assert TaskPhase.COORDINATING in phases

    # Verify ordering
    p_idx = phases.index(TaskPhase.PLANNING)
    e_idx = phases.index(TaskPhase.EXECUTING)
    r_idx = phases.index(TaskPhase.REVIEWING)
    c_idx = phases.index(TaskPhase.COORDINATING)
    assert p_idx < e_idx < r_idx < c_idx


# ---------------------------------------------------------------------------
# REVIEWING → EXECUTING retry edge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewing_retries_when_gate_fails() -> None:
    """REVIEWING node routes back to EXECUTING when a tool gate fails."""
    # Create a test_runner tool that fails on first call, passes on second.
    call_count = 0

    class _FailOnceTool:
        @property
        def name(self) -> str:
            return "test_runner"

        @property
        def description(self) -> str:
            return "Fails on first invocation"

        async def invoke(self, params: dict[str, Any]) -> ToolResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ToolResult(success=False, error="tests failed")
            return ToolResult(success=True)

    store = _MockStateStore()
    graph = _MockGraph()
    compiled = build_workflow(
        tools={"test_runner": _FailOnceTool()},
        state_store=store,
        graph=graph,
    )

    initial = _make_initial_state()
    config = {"configurable": {"thread_id": "test-thread-6"}}

    async for _ in compiled.astream(initial, config=config):
        pass

    # EXECUTING should have been entered twice (initial + retry)
    executing_checkpoints = [c for c in store.saved_checkpoints if c.phase == TaskPhase.EXECUTING]
    assert len(executing_checkpoints) == 2
    # Workflow should still complete
    reviewing_after_retry = [c for c in store.saved_checkpoints if c.phase == TaskPhase.REVIEWING]
    assert len(reviewing_after_retry) == 2


# ---------------------------------------------------------------------------
# AC-7: No hardcoded API keys (smoke check)
# ---------------------------------------------------------------------------


def test_no_hardcoded_api_key_in_workflow() -> None:
    """AC-7: ANTHROPIC_API_KEY is not hardcoded in workflow module source."""
    import inspect

    from tractable.agent import workflow

    source = inspect.getsource(workflow)
    assert "sk-ant" not in source
    assert "ANTHROPIC_API_KEY =" not in source


def test_no_hardcoded_model_name_in_node_logic() -> None:
    """AC-7 (extended): No hardcoded model names in node logic modules.

    Model names live in workflow.py as configurable defaults (TASK-2.5.2);
    they must NOT be hardcoded in individual node logic files.
    """
    import inspect

    from tractable.agent.nodes import coordinate, execute, plan, review

    for mod in (plan, execute, review, coordinate):
        src = inspect.getsource(mod)
        assert "claude-sonnet" not in src, f"claude-sonnet hardcoded in {mod.__name__}"
        assert "claude-opus" not in src, f"claude-opus hardcoded in {mod.__name__}"


# ---------------------------------------------------------------------------
# AgentWorkflowState schema
# ---------------------------------------------------------------------------


def test_agent_workflow_state_fields() -> None:
    """AgentWorkflowState TypedDict has all required fields."""
    state = _make_initial_state()
    assert state["agent_id"] == "agent-test"
    assert state["task_id"] == "task-test"
    assert state["phase"] == TaskPhase.PLANNING
    assert state["plan"] == []
    assert state["files_changed"] == []
    assert state["pr_url"] is None
    assert state["error"] is None
    assert state["token_count"] == 0
    assert state["current_model"] == "claude-sonnet-4-6"
