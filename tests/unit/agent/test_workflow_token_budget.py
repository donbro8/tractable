"""Unit tests for token budget tracking and Sonnet → Opus escalation (TASK-2.5.2).

Covers:
- AC-1: Sonnet → Opus escalation after budget exceeded; third LLM call uses Opus.
- AC-2: Escalation produces structlog entry with event="model_escalated".
- AC-3: Workflow already on Opus that exceeds budget raises FatalError.
- AC-4: Last saved checkpoint is preserved after FatalError (not data-corrupting).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest

from tractable.agent.state import AgentWorkflowState
from tractable.agent.workflow import check_token_budget, resume_task
from tractable.errors import FatalError
from tractable.types.agent import AgentCheckpoint, AgentContext, AuditEntry
from tractable.types.config import GovernancePolicy
from tractable.types.enums import ChangeRisk, TaskPhase
from tractable.types.graph import (
    CrossRepoEdge,
    GraphEntity,
    ImpactReport,
    MutationResult,
    RepoGraphSummary,
    Subgraph,
)

# ── Helpers / fixtures ─────────────────────────────────────────────────────


def _make_state(
    token_count: int = 0,
    current_model: str = "claude-sonnet-4-6",
    agent_id: str = "agent-test",
    task_id: str = "task-test",
) -> AgentWorkflowState:
    return AgentWorkflowState(
        agent_id=agent_id,
        task_id=task_id,
        task_description="test task",
        phase=TaskPhase.PLANNING,
        plan=[],
        files_changed=[],
        test_results={},
        pr_url=None,
        error=None,
        token_count=token_count,
        current_model=current_model,
        messages=[],
        resume_from=None,
    )


class _StubGraph:
    """Minimal CodeGraph stub returning non-empty summaries."""

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
        entity_ids: Sequence[Any],
        depth: int = 3,
        min_confidence: float = 0.5,
    ) -> ImpactReport:
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
            total_entities=3,
            key_modules=["main.py"],
            public_interfaces=[],
            cross_repo_dependencies=[],
            summary_text="stub summary",
        )

    async def mutate(self, mutations: Sequence[Any]) -> MutationResult:
        return MutationResult(applied=0)


class _MockStateStore:
    """Minimal AgentStateStore mock that records checkpoints."""

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
        return None  # Always start fresh; override per test via AsyncMock

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


# ── check_token_budget unit tests ─────────────────────────────────────────


def test_check_token_budget_under_budget_returns_empty() -> None:
    """No escalation when token_count is within budget."""
    governance = GovernancePolicy(token_budget_per_task=500)
    state = _make_state(token_count=499)
    result = check_token_budget(state, governance, "claude-opus-4-6")
    assert result == {}


def test_check_token_budget_at_budget_returns_empty() -> None:
    """No escalation when token_count equals the budget (not exceeded)."""
    governance = GovernancePolicy(token_budget_per_task=500)
    state = _make_state(token_count=500)
    result = check_token_budget(state, governance, "claude-opus-4-6")
    assert result == {}


def test_check_token_budget_sonnet_exceeded_returns_model_update() -> None:
    """Budget exceeded on Sonnet → returns escalation dict without raising."""
    governance = GovernancePolicy(token_budget_per_task=500)
    state = _make_state(token_count=600, current_model="claude-sonnet-4-6")
    result = check_token_budget(state, governance, "claude-opus-4-6")
    assert result == {"current_model": "claude-opus-4-6"}


def test_check_token_budget_opus_exceeded_raises_fatal_error() -> None:
    """AC-3: Budget exceeded on Opus → raises FatalError."""
    governance = GovernancePolicy(token_budget_per_task=500)
    state = _make_state(token_count=600, current_model="claude-opus-4-6")
    with pytest.raises(FatalError) as exc_info:
        check_token_budget(state, governance, "claude-opus-4-6")
    assert "Token budget exhausted on Opus model" in str(exc_info.value)
    assert "Checkpoint preserved" in str(exc_info.value)


# ── AC-2: Escalation log event ─────────────────────────────────────────────


def test_escalation_logs_model_escalated_event() -> None:
    """AC-2: Escalation produces structlog entry with event='model_escalated'."""
    governance = GovernancePolicy(token_budget_per_task=500)
    state = _make_state(token_count=600, current_model="claude-sonnet-4-6")

    with patch("tractable.agent.workflow._log") as mock_log:
        check_token_budget(state, governance, "claude-opus-4-6")

    mock_log.info.assert_called_with(
        "model_escalated",
        agent_id="agent-test",
        task_id="task-test",
        **{"from": "claude-sonnet-4-6", "to": "claude-opus-4-6"},
    )


def test_escalation_logs_budget_exceeded_warning() -> None:
    """Budget exceeded on Sonnet logs 'token_budget_exceeded' at warning."""
    governance = GovernancePolicy(token_budget_per_task=500)
    state = _make_state(token_count=600, current_model="claude-sonnet-4-6")

    with patch("tractable.agent.workflow._log") as mock_log:
        check_token_budget(state, governance, "claude-opus-4-6")

    mock_log.warning.assert_called_with(
        "token_budget_exceeded",
        agent_id="agent-test",
        task_id="task-test",
        tokens_used=600,
        budget=500,
    )


# ── AC-1: Sonnet → Opus escalation via resume_task ─────────────────────────


@pytest.mark.asyncio
async def test_sonnet_to_opus_escalation_third_call_uses_opus() -> None:
    """AC-1: With budget=500 and 300 tokens/call, the 3rd LLM call uses Opus.

    Workflow progression:
      PLANNING:     token_count=0   < 500 → OK; llm_call(sonnet) → +300 → total=300
      EXECUTING:    token_count=300 < 500 → OK; llm_call(sonnet) → +300 → total=600
      REVIEWING:    token_count=600 > 500 → escalate to opus; llm_call(opus) → +300
      COORDINATING: token_count=900 > 500, model=opus → FatalError (expected)
    """
    llm_calls: list[str] = []

    def mock_llm(model: str) -> int:
        llm_calls.append(model)
        return 300

    governance = GovernancePolicy(token_budget_per_task=500)
    store = _MockStateStore()

    with pytest.raises(FatalError):
        await resume_task(
            agent_id="agent-ac1",
            task_id="task-ac1",
            task_description="AC-1 test task",
            state_store=store,
            tools={},
            graph=_StubGraph(),
            governance=governance,
            default_model="claude-sonnet-4-6",
            escalation_model="claude-opus-4-6",
            llm_call=mock_llm,
        )

    # At least 3 LLM calls must have been recorded.
    assert len(llm_calls) >= 3, f"Expected ≥3 llm calls, got {len(llm_calls)}: {llm_calls}"
    # First two calls with Sonnet (before budget exceeded).
    assert llm_calls[0] == "claude-sonnet-4-6"
    assert llm_calls[1] == "claude-sonnet-4-6"
    # Third call after escalation must use Opus.
    assert llm_calls[2] == "claude-opus-4-6"


# ── AC-3: Workflow starting on Opus exceeds budget → FatalError ─────────────


@pytest.mark.asyncio
async def test_workflow_on_opus_exceeds_budget_raises_fatal_error() -> None:
    """AC-3: Workflow starting on Opus with token_count > budget raises FatalError."""
    governance = GovernancePolicy(token_budget_per_task=100)
    store = _MockStateStore()

    with pytest.raises(FatalError) as exc_info:
        await resume_task(
            agent_id="agent-ac3",
            task_id="task-ac3",
            task_description="AC-3 test task",
            state_store=store,
            tools={},
            graph=_StubGraph(),
            governance=governance,
            default_model="claude-opus-4-6",  # Start on Opus directly
            escalation_model="claude-opus-4-6",
            llm_call=lambda _model: 200,  # 200 > 100 budget from first call
        )

    assert "Token budget exhausted on Opus model" in str(exc_info.value)


# ── AC-4: Checkpoint preserved after FatalError ────────────────────────────


@pytest.mark.asyncio
async def test_checkpoint_preserved_after_fatal_error() -> None:
    """AC-4: Last saved checkpoint remains after FatalError (not data-corrupting).

    When FatalError is raised during budget check at the start of a node,
    checkpoints saved by all previous nodes remain intact in the state store.
    """
    governance = GovernancePolicy(token_budget_per_task=500)
    store = _MockStateStore()

    with pytest.raises(FatalError):
        await resume_task(
            agent_id="agent-ac4",
            task_id="task-ac4",
            task_description="AC-4 checkpoint preservation",
            state_store=store,
            tools={},
            graph=_StubGraph(),
            governance=governance,
            default_model="claude-sonnet-4-6",
            escalation_model="claude-opus-4-6",
            llm_call=lambda _model: 300,  # 300 tokens/call; budget=500
        )

    # Checkpoints were saved before FatalError; at least PLANNING is present.
    assert len(store.saved_checkpoints) >= 1, (
        "Expected at least one checkpoint to be preserved after FatalError"
    )
    # The checkpoint task_id matches — not corrupted.
    assert store.saved_checkpoints[0].task_id == "task-ac4"
